/* file: svm_helper_oneapi.h */
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

#ifndef __SVM_HELPER_ONEAPI_H__
#define __SVM_HELPER_ONEAPI_H__

#include "service/kernel/data_management/service_numeric_table.h"
#include "service/kernel/oneapi/sorter.h"
#include "externals/service_ittnotify.h"

#include "algorithms/kernel/svm/oneapi/cl_kernels/svm_kernels.cl"

// TODO: DELETE
#include <cstdlib>
#include <chrono>
using namespace std::chrono;
//

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
template <typename T>
inline const T & min(const T & a, const T & b)
{
    return !(b < a) ? a : b;
}

template <typename T>
inline const T & max(const T & a, const T & b)
{
    return (a < b) ? b : a;
}

template <typename T>
inline const T abs(const T & a)
{
    return a > 0 ? a : -a;
}

inline size_t maxpow2(size_t n)
{
    if (!(n & (n - 1)))
    {
        return n;
    }

    size_t count = 0;
    while (n > 1)
    {
        n >>= 1;
        count++;
    }
    return 1 << count;
}

using namespace daal::services::internal;
using namespace daal::oneapi::internal;

template <typename algorithmFPType>
struct HelperSVM
{
    static services::Status buildProgram(ClKernelFactoryIface & factory)
    {
        services::String options = getKeyFPType<algorithmFPType>();

        services::String cachekey("__daal_algorithms_svm_");
        cachekey.add(options);
        options.add(" -D LOCAL_SUM_SIZE=256 ");

        services::Status status;
        factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelSVM, options.c_str(), &status);
        return status;
    }

    static services::Status initGrad(const services::Buffer<algorithmFPType> & y, services::Buffer<algorithmFPType> & grad, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(initGrad);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("initGradient");

        KernelArguments args(2);
        args.set(0, y, AccessModeIds::read);
        args.set(1, grad, AccessModeIds::write);

        KernelRange range(n);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

    static services::Status rangeIndices(UniversalBuffer & x, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(range);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("range");

        KernelArguments args(1);
        args.set(0, x, AccessModeIds::readwrite);

        KernelRange range(n);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

    static services::Status argSort(const UniversalBuffer & f, UniversalBuffer & values, UniversalBuffer & indecesSort, UniversalBuffer & indeces,
                                    const size_t n)
    {
        services::Status status;
        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

        context.copy(values, 0, f, 0, n, &status);
        DAAL_CHECK_STATUS(status, rangeIndices(indecesSort, n));

        DAAL_CHECK_STATUS(status, sort::RadixSort::sortIndeces(values, indecesSort, values, indeces, n));

        return status;
    }

    static services::Status gatherIndices(const services::Buffer<int> & mask, const services::Buffer<int> & x, const size_t n,
                                          services::Buffer<int> & res, size_t & nRes)
    {
        services::Status status;

        int * indicator_host      = mask.toHost(ReadWriteMode::readOnly).get();
        int * sortedFIndices_host = x.toHost(ReadWriteMode::readOnly).get();
        int * tmpIndices_host     = res.toHost(ReadWriteMode::writeOnly).get();
        nRes                      = 0;
        for (int i = 0; i < n; i++)
        {
            if (indicator_host[sortedFIndices_host[i]])
            {
                tmpIndices_host[nRes] = sortedFIndices_host[i];
                nRes++;
            }
        }
        return status;
    }

    static services::Status gatherValues(const services::Buffer<int> & mask, const services::Buffer<algorithmFPType> & x, const size_t n,
                                         services::Buffer<algorithmFPType> & res, size_t & nRes)
    {
        // TODO: replace on partition from GBT
        services::Status status;

        int * indicator_host                  = mask.toHost(ReadWriteMode::readOnly).get();
        algorithmFPType * sortedFIndices_host = x.toHost(ReadWriteMode::readOnly).get();
        algorithmFPType * tmpIndices_host     = res.toHost(ReadWriteMode::writeOnly).get();
        nRes                                  = 0;
        for (int i = 0; i < n; i++)
        {
            if (indicator_host[i])
            {
                tmpIndices_host[nRes] = sortedFIndices_host[i];
                nRes++;
            }
        }
        return status;
    }

    static services::Status copyBlockIndices(const services::Buffer<algorithmFPType> & x, const services::Buffer<int> & ind,
                                             services::Buffer<algorithmFPType> & newX, const uint32_t nWS, const uint32_t p)
    {
        services::Status status;

        oneapi::internal::ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
        oneapi::internal::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        buildProgram(factory);

        const char * const kernelName      = "copyBlockIndices";
        oneapi::internal::KernelPtr kernel = factory.getKernel(kernelName);

        oneapi::internal::KernelArguments args(4);
        args.set(0, x, oneapi::internal::AccessModeIds::read);
        args.set(1, ind, oneapi::internal::AccessModeIds::read);
        args.set(2, p);
        args.set(3, newX, oneapi::internal::AccessModeIds::write);

        oneapi::internal::KernelRange range(p, nWS);

        ctx.run(range, kernel, args, &status);

        return status;
    }

    static services::Status checkUpper(const services::Buffer<algorithmFPType> & y, const services::Buffer<algorithmFPType> & alpha,
                                       services::Buffer<int> & indicator, const algorithmFPType C, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(checkUpper);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("checkUpper");

        KernelArguments args(4);
        args.set(0, y, AccessModeIds::read);
        args.set(1, alpha, AccessModeIds::read);
        args.set(2, C);
        args.set(3, indicator, AccessModeIds::write);

        KernelRange range(n);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

    static services::Status checkLower(const services::Buffer<algorithmFPType> & y, const services::Buffer<algorithmFPType> & alpha,
                                       services::Buffer<int> & indicator, const algorithmFPType C, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(checkLower);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("checkLower");

        KernelArguments args(4);
        args.set(0, y, AccessModeIds::read);
        args.set(1, alpha, AccessModeIds::read);
        args.set(2, C);
        args.set(3, indicator, AccessModeIds::write);

        KernelRange range(n);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

    static services::Status checkFree(const services::Buffer<algorithmFPType> & alpha, services::Buffer<int> & mask, const algorithmFPType C,
                                      const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(checkFree);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("checkFree");

        KernelArguments args(3);
        args.set(0, alpha, AccessModeIds::read);
        args.set(1, C);
        args.set(2, mask, AccessModeIds::write);

        KernelRange range(n);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

    static services::Status checkNotZero(const services::Buffer<algorithmFPType> & alpha, services::Buffer<int> & mask, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(checkNotZero);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("checkNotZero");

        KernelArguments args(2);
        args.set(0, alpha, AccessModeIds::read);
        args.set(1, mask, AccessModeIds::write);

        KernelRange range(n);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

    static services::Status computeDualCoeffs(const services::Buffer<algorithmFPType> & y, services::Buffer<algorithmFPType> & alpha, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(computeDualCoeffs);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("computeDualCoeffs");

        KernelArguments args(2);
        args.set(0, y, AccessModeIds::read);
        args.set(1, alpha, AccessModeIds::readwrite);

        KernelRange range(n);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
