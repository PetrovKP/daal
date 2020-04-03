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
#include "service/kernel/oneapi/blas_gpu.h"

// TODO: DELETE
#include <algorithm>
#include <cstdlib>
#include <chrono>
using namespace std::chrono;
//

#include "algorithms/kernel/svm/oneapi/svm_train_cache.h"
#include "algorithms/kernel/svm/oneapi/svm_train_workset.h"
#include "algorithms/kernel/svm/oneapi/svm_train_result.h"

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

    services::Status status = Helper::buildProgram(factory);
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
services::Status SVMTrainOneAPI<algorithmFPType, ParameterType, boser>::updateGrad(const services::Buffer<algorithmFPType> & kernelWS,
                                                                                   const services::Buffer<algorithmFPType> & deltaalpha,
                                                                                   services::Buffer<algorithmFPType> & grad, const size_t nVectors,
                                                                                   const size_t nWS)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(updateGrad);
    return BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::Trans, math::Transpose::NoTrans, nVectors, 1, nWS,
                                           algorithmFPType(1), kernelWS, nVectors, 0, deltaalpha, 1, 0, algorithmFPType(1), grad, 1, 0);
}

template <typename algorithmFPType, typename ParameterType>
services::Status SVMTrainOneAPI<algorithmFPType, ParameterType, boser>::smoKernel(
    const services::Buffer<algorithmFPType> & y, const services::Buffer<algorithmFPType> & kernelWsRows, const services::Buffer<int> & wsIndices,
    const int ldK, const services::Buffer<algorithmFPType> & f, const algorithmFPType C, const algorithmFPType eps, const algorithmFPType tau,
    const int maxInnerIteration, services::Buffer<algorithmFPType> & alpha, services::Buffer<algorithmFPType> & deltaalpha,
    services::Buffer<algorithmFPType> & resinfo, const size_t nWS)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(smoKernel);

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();

    services::Status status = Helper::buildProgram(factory);
    DAAL_CHECK_STATUS_VAR(status);

    auto kernel = factory.getKernel("smoKernel");

    KernelArguments args(12);
    args.set(0, y, AccessModeIds::read);
    args.set(1, kernelWsRows, AccessModeIds::read);
    args.set(2, wsIndices, AccessModeIds::read);
    args.set(3, ldK);
    args.set(4, f, AccessModeIds::read);
    args.set(5, C);
    args.set(6, eps);
    args.set(7, tau);
    args.set(8, maxInnerIteration);
    args.set(9, alpha, AccessModeIds::readwrite);
    args.set(10, deltaalpha, AccessModeIds::readwrite);
    args.set(11, resinfo, AccessModeIds::readwrite);

    KernelRange localRange(nWS);
    KernelRange globalRange(nWS);

    KernelNDRange range(1);
    range.global(globalRange, &status);
    DAAL_CHECK_STATUS_VAR(status);
    range.local(localRange, &status);
    DAAL_CHECK_STATUS_VAR(status);

    context.run(range, kernel, args, &status);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

template <typename algorithmFPType, typename ParameterType>
bool SVMTrainOneAPI<algorithmFPType, ParameterType, boser>::checkStopCondition(const algorithmFPType diff, const algorithmFPType diffPrev,
                                                                               const algorithmFPType eps, int & sameLocalDiff)
{
    sameLocalDiff = abs(diff - diffPrev) < eps * 1e-3 ? sameLocalDiff + 1 : 0;

    if (sameLocalDiff > 5)
    {
        return true;
    }
    return false;
}

template <typename algorithmFPType, typename ParameterType>
double SVMTrainOneAPI<algorithmFPType, ParameterType, boser>::calculateObjective(const services::Buffer<algorithmFPType> & y,
                                                                                 const services::Buffer<algorithmFPType> & alpha,
                                                                                 const services::Buffer<algorithmFPType> & grad,
                                                                                 const size_t nVectors)
{
    double obj     = 0.0f;
    auto yHost     = y.toHost(ReadWriteMode::readOnly).get();
    auto alphaHost = alpha.toHost(ReadWriteMode::readOnly).get();
    auto gradHost  = grad.toHost(ReadWriteMode::readOnly).get();
    for (size_t i = 0; i < nVectors; i++)
    {
        obj += alphaHost[i] - (gradHost[i] + yHost[i]) * alphaHost[i] * yHost[i] * 0.5;
    }
    return obj;
}

template <typename algorithmFPType, typename ParameterType>
services::Status SVMTrainOneAPI<algorithmFPType, ParameterType, boser>::compute(const NumericTablePtr & xTable, NumericTable & yTable,
                                                                                daal::algorithms::Model * r, const ParameterType * svmPar)
{
    services::Status status;

    auto & context       = services::Environment::getInstance()->getDefaultExecutionContext();
    const auto idType    = TypeIds::id<algorithmFPType>();
    const auto idTypeInt = TypeIds::id<int>();

    auto & deviceInfo = context.getInfoDevice();

    if (const char * env_p = std::getenv("SVM_VERBOSE"))
    {
        printf(">> VERBOSE MODE\n");
        verbose = true;
        printf(">> MAX WORK SIZE = %d\n", (int)deviceInfo.max_work_group_size);
    }

    const algorithmFPType C(svmPar->C);
    const algorithmFPType eps(svmPar->accuracyThreshold);
    const algorithmFPType tau(svmPar->tau);
    const size_t maxIterations(svmPar->maxIterations);
    const size_t cacheSize(svmPar->cacheSize);
    kernel_function::KernelIfacePtr kernel = svmPar->kernel->clone();
    // TODO
    const size_t innerMaxIterations(100);

    const size_t nVectors  = xTable->getNumberOfRows();
    const size_t nFeatures = xTable->getNumberOfColumns();
    // ai = 0
    auto alphaU = context.allocate(idType, nVectors, &status);
    context.fill(alphaU, 0.0, &status);
    DAAL_CHECK_STATUS_VAR(status);
    auto alphaBuff = alphaU.get<algorithmFPType>();

    // fi = -yi
    auto fU = context.allocate(idType, nVectors, &status);
    DAAL_CHECK_STATUS_VAR(status);
    auto gradBuff = fU.get<algorithmFPType>();

    BlockDescriptor<algorithmFPType> yBD;
    DAAL_CHECK_STATUS(status, yTable.getBlockOfRows(0, nVectors, ReadWriteMode::readOnly, yBD));
    auto yBuff = yBD.getBuffer();

    // TOD: Delete xblock. He needs only for kernel
    BlockDescriptor<algorithmFPType> xBD;
    DAAL_CHECK_STATUS(status, xTable->getBlockOfRows(0, nVectors, ReadWriteMode::readOnly, xBD));
    auto xBuff = xBD.getBuffer();

    DAAL_CHECK_STATUS(status, initGrad(yBuff, gradBuff, nVectors));

    TaskWorkingSet<algorithmFPType> workSet(nVectors, verbose);

    DAAL_CHECK_STATUS(status, workSet.init());

    const size_t nWS = workSet.getSize();
    const size_t q   = nWS / 2;

    auto deltaalphaU = context.allocate(idType, nWS, &status);
    DAAL_CHECK_STATUS_VAR(status);
    auto deltaalphaBuff = deltaalphaU.get<algorithmFPType>();

    auto resinfoU = context.allocate(idType, 2, &status);
    DAAL_CHECK_STATUS_VAR(status);
    auto resinfoBuff = resinfoU.get<algorithmFPType>();

    int localInnerIteration  = 0;
    int sameLocalDiff        = 0;
    int innerIteration       = -1;
    algorithmFPType diff     = algorithmFPType(0);
    algorithmFPType diffPrev = algorithmFPType(0);

    SVMCacheOneAPIIface<algorithmFPType> * cache = nullptr;

    if (cacheSize > nWS * nVectors * sizeof(algorithmFPType))
    {
        cache = SVMCacheOneAPI<noCache, algorithmFPType>::create(cacheSize, nWS, nVectors, xTable, kernel, verbose, status);
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

    for (size_t iter = 0; iter < 1 /*maxIterations*/; iter++)
    {
        if (iter != 0)
        {
            DAAL_CHECK_STATUS(status, workSet.saveQWSIndeces(q));
        }
        {
            const auto t_0 = high_resolution_clock::now();

            DAAL_CHECK_STATUS(status, workSet.selectWS(yBuff, alphaBuff, gradBuff, C));

            if (verbose)
            {
                const auto t_1           = high_resolution_clock::now();
                const float duration_sec = duration_cast<milliseconds>(t_1 - t_0).count();
                printf(">>>> SelectWS.compute time(ms) = %.1f\n", duration_sec);
                fflush(stdout);
            }
        }

        auto & wsIndices = workSet.getWSIndeces();
        {
            const auto t_0 = high_resolution_clock::now();

            DAAL_CHECK_STATUS(status, cache->compute(xBuff, wsIndices, nFeatures));

            if (verbose)
            {
                const auto t_1           = high_resolution_clock::now();
                const float duration_sec = duration_cast<milliseconds>(t_1 - t_0).count();
                printf(">>>> Kernel.compute time(ms) = %.1f\n", duration_sec);
                fflush(stdout);
            }
        }

        // TODO: Save half elements from kernel on 1+ iterations
        auto kernelWS = cache->getSetRowsBlock();

        {
            const auto t_0 = high_resolution_clock::now();

            DAAL_CHECK_STATUS(status, smoKernel(yBuff, kernelWS, wsIndices, nVectors, gradBuff, C, eps, tau, innerMaxIterations, alphaBuff,
                                                deltaalphaBuff, resinfoBuff, nWS));
            {
                auto resinfoHost = resinfoBuff.toHost(ReadWriteMode::readOnly, &status).get();
                innerIteration   = int(resinfoHost[0]);
                diff             = resinfoHost[1];
            }

            if (verbose)
            {
                const auto t_1           = high_resolution_clock::now();
                const float duration_sec = duration_cast<milliseconds>(t_1 - t_0).count();
                printf(">>>> smoKernel (ms) = %.3f\n", duration_sec);
                printf(">>>> innerIteration = %d diff = %.1f\n", innerIteration, diff);
                fflush(stdout);
            }
        }

        {
            const auto t_0 = high_resolution_clock::now();

            DAAL_CHECK_STATUS(status, updateGrad(kernelWS, deltaalphaBuff, gradBuff, nVectors, nWS));

            {
                auto resinfoHost = resinfoBuff.toHost(ReadWriteMode::readOnly).get();
                innerIteration   = int(resinfoHost[0]);
                diff             = resinfoHost[1];
            }

            if (verbose)
            {
                const auto t_1           = high_resolution_clock::now();
                const float duration_sec = duration_cast<milliseconds>(t_1 - t_0).count();
                printf(">>>> updateGrad (ms) = %.1f\n", duration_sec);
                fflush(stdout);
            }

            localInnerIteration += innerIteration;
        }
        if (verbose)
        {
            double obj = calculateObjective(yBuff, alphaBuff, gradBuff, nVectors);
            printf(">>>>>> calculateObjective diff = %.3lf\n", obj);
        }

        if (checkStopCondition(diff, diffPrev, eps, sameLocalDiff))
        {
            if (verbose)
            {
                printf(">>>> checkStopCondition diff = %.3f diffPrev = %.3f\n", diff, diffPrev);
            }
            break;
        }
        diffPrev = diff;
    }

    DAAL_CHECK_STATUS(status, xTable->releaseBlockOfRows(xBD));

    Result<algorithmFPType> result(alphaBuff, fBuff, yBuff, C, nVectors);

    DAAL_CHECK_STATUS(status, result.setResultsToModel(*xTable, *static_cast<Model *>(r)));

    DAAL_CHECK_STATUS(status, yTable.releaseBlockOfRows(yBD));

    delete cache;

    return status;
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
