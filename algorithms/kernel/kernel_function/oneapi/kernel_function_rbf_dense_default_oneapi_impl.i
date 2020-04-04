/* file: kernel_function_rbf_dense_default_oneapi_impl.i */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  RBF kernel functions implementation
//--
*/

#ifndef __KERNEL_FUNCTION_RBF_DENSE_DEFAULT_IMPL_ONEAPI_I__
#define __KERNEL_FUNCTION_RBF_DENSE_DEFAULT_IMPL_ONEAPI_I__

#include "algorithms/kernel_function/kernel_function_types_rbf.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "externals/service_math.h"
#include "externals/service_ittnotify.h"
#include "service/kernel/oneapi/blas_gpu.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace rbf
{
namespace internal
{
using namespace daal::oneapi::internal;
using namespace daal::oneapi::internal::math;

template <typename algorithmFPType>
static services::Status KernelImplRBF<defaultDense, algorithmFPType>::buildProgram(ClKernelFactoryIface & factory)
{
    services::String options = getKeyFPType<algorithmFPType>();

    services::String cachekey("__daal_algorithms_kernel_function_");
    cachekey.add(options);
    options.add(" -D LOCAL_SUM_SIZE=256 ");

    services::Status status;
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelKF, options.c_str(), &status);
    return status;
}

template <typename algorithmFPType>
static services::Status KernelImplRBF<defaultDense, algorithmFPType>::lazyAllocate(oneapi::internal::UniversalBuffer & x, const size_t n)
{
    services::Status status;
    ExecutionContextIface & ctx = services::Environment::getInstance()->getDefaultExecutionContext();
    const TypeIds::Id idType    = oneapi::internal::TypeIds::id<algorithmFPType>();

    if (x.empty() || x.get<algorithmFPType>().size() < n)
    {
        x = ctx.allocate(idType, n, &status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status KernelImplRBF<defaultDense, algorithmFPType>::computeRBF(const services::Buffer<algorithmFPType> & sqrA1,
                                                                          const services::Buffer<algorithmFPType> & sqrA2, const uint32_t ld,
                                                                          const algorithmFPType coeff, services::Buffer<algorithmFPType> & rbf,
                                                                          const size_t nVectors1, const size_t nVectors2)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(computeRBF);

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();

    services::Status status = buildProgram(factory);
    DAAL_CHECK_STATUS_VAR(status);

    auto kernel = factory.getKernel("computeRBF");

    const algorithmFPType threshold = math::expThreshold<algorithmFPType>();

    KernelArguments args(6);
    args.set(0, sqrA1, AccessModeIds::read);
    args.set(1, sqrA2, AccessModeIds::read);
    args.set(2, ld);
    args.set(3, threshold);
    args.set(4, coeff);
    args.set(5, rbf, AccessModeIds::readwrite);

    KernelRange range(nVectors1, nVectors2);

    context.run(range, kernel, args, &status);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

template <typename algorithmFPType>
services::Status KernelImplRBF<defaultDense, algorithmFPType>::computeInternalVectorVector(NumericTable & a1, NumericTable & a2, NumericTable * r,
                                                                                           const ParameterBase * par)
{
    return services::ErrorMethodNotImplemented;
}

template <typename algorithmFPType>
services::Status KernelImplRBF<defaultDense, algorithmFPType>::computeInternalMatrixVector(NumericTable & a1, NumericTable & a2, NumericTable & r,
                                                                                           const ParameterBase * par)
{
    return services::ErrorMethodNotImplemented;
}

template <typename algorithmFPType>
services::Status KernelImplRBF<defaultDense, algorithmFPType>::computeInternalMatrixMatrix(NumericTable & a1, NumericTable & a2, NumericTable & r,
                                                                                           const ParameterBase * par)
{
    //prepareData
    services::Status status;

    auto & context    = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & deviceInfo = context.getInfoDevice();

    const size_t nVectors1 = a1.getNumberOfRows();
    const size_t nVectors2 = a2.getNumberOfRows();

    const size_t nFeatures1 = a1.getNumberOfColumns();
    const size_t nFeatures2 = a2.getNumberOfColumns();
    DAAL_ASSERT(nFeatures1 == nFeatures2);

    //compute
    const Parameter * rbfPar    = static_cast<const Parameter *>(par);
    const algorithmFPType coeff = algorithmFPType(-0.5 / (rbfPar->sigma * rbfPar->sigma));

    const size_t startRows = 0;
    DAAL_CHECK_STATUS(status, a1.getBlockOfRows(startRows, nVectors1, ReadWriteMode::readOnly, a1BD));
    DAAL_CHECK_STATUS(status, a2.getBlockOfRows(startRows, nVectors2, ReadWriteMode::readOnly, a2BD));

    DAAL_CHECK_STATUS(status, r.getBlockOfRows(startRows, nVectors1, ReadWriteMode::readWrite, rBD));

    const services::Buffer<algorithmFPType> a1Buf = a1BD.getBuffer();
    const services::Buffer<algorithmFPType> a2Buf = a2BD.getBuffer();

    services::Buffer<algorithmFPType> rBuf = rBD.getBuffer();

    DAAL_CHECK_STATUS(status, lazyAllocate(_sqrA1U, nVectors1));
    DAAL_CHECK_STATUS(status, lazyAllocate(_sqrA2U nVectors2));

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(KernelRBF.SumOfSquared);

        Reducer::reduce(Reducer::BinaryOp::SUMS_OF_SQUARED, math::Layout::RowMajor, a1Buf, sqrA1U, nVectors1, nFeatures1, &status);
        Reducer::reduce(Reducer::BinaryOp::SUMS_OF_SQUARED, math::Layout::RowMajor, a2Buf, sqrA2U, nVectors2, nFeatures2, &status);
    }

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(KernelRBF.gemm);
        // TODO: Need block GEMM to avoid copying

        algorithmFPType alpha = algorithmFPType(-2.0);
        algorithmFPType beta  = algorithmFPType(0.0);

        DAAL_CHECK_STATUS(
            status, BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::Trans, nVectors1, nVectors2,
                                                    nFeatures1, alpha, a1Buf, nFeatures1, 0, a2Buf, nFeatures2, 0, beta, rBuf, nVectors2, 0));
    }

    sqrA1Buff = sqrA1U.gen<algorithmFPType>();
    sqrA2Buff = sqrA2U.gen<algorithmFPType>();

    DAAL_CHECK_STATUS(status, computeRBF(sqrA1Buff, sqrA2Buff, r.getNumberOfColumns(), coeff, rBuf, nVectors1, nVectors2));
    return status;
}

} // namespace internal
} // namespace rbf
} // namespace kernel_function
} // namespace algorithms
} // namespace daal
} // namespace daal

#endif
