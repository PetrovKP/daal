/* file: svm_train_boser_impl.i */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  SVM training algorithm implementation
//--
*/
/*
//  DESCRIPTION
//
//  Definition of the functions for training with SVM 2-class classifier.
//
//  REFERENCES
//
//  1. Rong-En Fan, Pai-Hsuen Chen, Chih-Jen Lin,
//     Working Set Selection Using Second Order Information
//     for Training Support Vector Machines,
//     Journal of Machine Learning Research 6 (2005), pp. 1889___1918
//  2. Bernard E. boser, Isabelle M. Guyon, Vladimir N. Vapnik,
//     A Training Algorithm for Optimal Margin Classifiers.
//  3. Thorsten Joachims, Making Large-Scale SVM Learning Practical,
//     Advances in Kernel Methods - Support Vector Learning
*/

#ifndef __SVM_TRAIN_GPU_IMPL_I__
#define __SVM_TRAIN_GPU_IMPL_I__

#include "externals/service_memory.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "service/kernel/service_utils.h"
#include "service/kernel/service_data_utils.h"
#include "externals/service_ittnotify.h"
#include "algorithms/kernel/svm/oneapi/cl_kernels/svm_train_oneapi.cl"

// TODO: DELETE
#include <algorithm>
#include <cstdlib>
#include <chrono>
using namespace std::chrono;
//

#include "algorithms/kernel/svm/oneapi/svm_train_cache.h"
#include "algorithms/kernel/svm/oneapi/svm_workset.h"

DAAL_ITTNOTIFY_DOMAIN(svm_train.default.batch);

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::oneapi::internal;

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, typename ParameterType>
services::Status SVMTrainOneAPI<algorithmFPType, ParameterType, boser>::initGrad(const services::Buffer<algorithmFPType> & y,
                                                                                 services::Buffer<algorithmFPType> & f, const size_t nVectors)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(initGrad);

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();

    services::Status status = HelperSVM::buildProgram(factory);
    DAAL_CHECK_STATUS_VAR(status);

    auto kernel = factory.getKernel("initGradient");

    KernelArguments args(2);
    args.set(0, y, AccessModeIds::read);
    args.set(1, f, AccessModeIds::write);

    KernelRange range(nVectors);

    context.run(range, kernel, args, &status);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

template <typename algorithmFPType, typename ParameterType>
services::Status SVMTrainOneAPI<algorithmFPType, ParameterType, boser>::compute(const NumericTablePtr & xTable, NumericTable & yTable,
                                                                                daal::algorithms::Model * r, const ParameterType * svmPar)
{
    services::Status status;

    auto & context       = services::Environment::getInstance()->getDefaultExecutionContext();
    const auto idType    = TypeIds::id<algorithmFPType>();
    const auto idTypeInt = TypeIds::id<int>();

    if (const char * env_p = std::getenv("SVM_VERBOSE"))
    {
        printf(">> VERBOSE MODE\n");
        verbose = true;
    }

    const algorithmFPType C(svmPar->C);
    const algorithmFPType eps(svmPar->accuracyThreshold);
    const algorithmFPType tau(svmPar->tau);
    const size_t maxIterations(svmPar->maxIterations);
    const size_t cacheSize(svmPar->cacheSize);
    kernel_function::KernelIfacePtr kernel = svmPar.kernel->clone();
    // TODO
    const size_t innerMaxIterations(100);

    const size_t nVectors = xTable->getNumberOfRows();

    // ai = 0
    auto alphaU = context.allocate(idType, nVectors, &status);
    context.fill(alphaU, 0.0, &status);
    DAAL_CHECK_STATUS_VAR(status);
    auto alphaBuff = alphaU.get<algorithmFPType>();

    // fi = -yi
    auto fU = context.allocate(idType, nVectors, &status);
    DAAL_CHECK_STATUS_VAR(status);
    auto fBuff = fU.get<algorithmFPType>();

    BlockDescriptor<algorithmFPType> yBD;
    yTable.getBlockOfRows(0, nVectors, ReadWriteMode::readOnly, yBD);
    auto yBuff = yBD.getBuffer();

    BlockDescriptor<algorithmFPType> xBD;
    xTable.getBlockOfRows(0, nVectors, ReadWriteMode::readOnly, xBD);
    auto xBuff = xBD.getBuffer();

    DAAL_CHECK_STATUS(status, initGrad(yBuff, fBuff, nVectors));

    TaskWorkingSet<algorithmFPType> workSet(nVectors, verbose);

    DAAL_CHECK_STATUS(status, workSet.init());

    const size_t nWS = workSet.getSize();

    SVMCacheOneAPIIface * cache = nullptr;

    if (cacheSize > nWS * nVectors * sizeof(algorithmFPType))
    {
        cache = SVMCacheOneAPI<noCache, algorithmFPType>::create(cacheSize, _nVectors, nWS, xTable, kernel, status);
    }
    else
    {
        // TODO!
        return status;
    }

    if (verbose)
    {
        printf(">>>> nVectors: %lu d: %lu nWS: %lu C: %f \n", nVectors, xTable->getNumberOfColumns(), nWS, C);
    }

    // TODO transfer on GPU

    // for (size_t iter = 0; iter < maxIterations; i++)
    {
        const auto t_0 = high_resolution_clock::now();

        DAAL_CHECK_STATUS(status, workSet.selectWS(yBuff, alphaBuff, fBuff, C));

        if (verbose)
        {
            const auto t_1           = high_resolution_clock::now();
            const float duration_sec = duration_cast<milliseconds>(t_1 - t_0).count();
            printf(">>>> SelectWS.compute time(ms) = %.1f\n", duration_sec);
        }
    }

    auto wsIndices = workSet.getWSIndeces();

    {
        const auto t_0 = high_resolution_clock::now();

        DAAL_CHECK_STATUS(status, cache->compute(xBuff, alphaBuff, fBuff, C));

        if (verbose)
        {
            const auto t_1           = high_resolution_clock::now();
            const float duration_sec = duration_cast<milliseconds>(t_1 - t_0).count();
            printf(">>>> Kernel.compute time(ms) = %.1f\n", duration_sec);
        }
    }

    auto kernelWS = cache->getSetRowsBlock();

    DAAL_CHECK_STATUS(status, yTable.releaseBlockOfRows(yBD));
    DAAL_CHECK_STATUS(status, xTable.releaseBlockOfRows(xBD));

    delete cache;

    return status;

    // return s.ok() ? task.setResultsToModel(*xTable, *static_cast<Model *>(r), svmPar->C) : s;
}

// inline Size MaxPow2(Size nVectors) {
//     if (!(n & (n - 1))) {
//         return nVectors;
//     }

//     Size count = 0;
//     while (n > 1) {
//         nVectors >>= 1;
//         count++;
//     }
//     return 1 << count;
// }

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
