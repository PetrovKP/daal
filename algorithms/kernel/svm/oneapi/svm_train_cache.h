/* file: svm_train_cache.h */
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
//  SVM cache structure implementation
//--
*/

#ifndef __SVM_TRAIN_CACHE_H__
#define __SVM_TRAIN_CACHE_H__

#include "service/kernel/service_utils.h"
#include "externals/service_memory.h"
#include "service/kernel/data_management/service_micro_table.h"
#include "service/kernel/data_management/service_numeric_table.h"

using namespace daal::services::internal;

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
/**
 * Types of caches for kernel function values
 */
enum SVMCacheOneAPIType
{
    noCache,     /*!< No storage for caching kernel function values is provided */
    simpleCache, /*!< Storage for caching ALL kernel function values is provided */
    lruCache     /*!< Storage for caching PART of kernel function values is provided;
                         LRU algorithm is used to exclude values from cache */
};

/**
 * Common interface for cache that stores kernel function values
 */
template <typename algorithmFPType>
class SVMCacheOneAPIIface
{
public:
    virtual ~SVMCacheOneAPIIface() {}

    virtual services::Status compute(const services::Buffer<algorithmFPType>& xBuff, const services::Buffer<int>& wsIndices, const size_t p) = 0;

    services::Buffer<algorithmFPType>& getSetRowsBlock() const {
        return _cache.get<algorithmFPType>();
    }

protected:
    UniversalBuffer _cache;
    const size_t _lineSize;                        /*!< Number of elements in the cache line */
    const size_t _nWS;                            /*!< Number of elements in the cache line */
    const kernel_function::KernelIfacePtr _kernel; /*!< Kernel function */

};

template <SVMCacheOneAPIType cacheType, typename algorithmFPType>
class SVMCacheOneAPI
{};

/**
 * No cache: kernel function values are not cached
 */
template <typename algorithmFPType>
class SVMCacheOneAPI<noCache, algorithmFPType> : public SVMCacheOneAPIIface<algorithmFPType>
{
    using HelperSVM = HelperSVM<algorithmFPType>;
    using super = SVMCacheOneAPIIface<algorithmFPType>;
    using thisType = SVMCacheOneAPI<noCache, algorithmFPType>;
    using super::_cache;
    using super::_kernel;
    using super::_lineSize;
    using super::_nWS;

public:
    ~SVMCacheOneAPI() {}

    DAAL_NEW_DELETE();

    static SVMCacheOneAPI * create(const size_t cacheSize, const size_t nWS, const size_t lineSize, const NumericTablePtr & xTable,
                             const kernel_function::KernelIfacePtr & kernel, Status & s)
    {
        s.clear();
        thisType * res = new thisType(nWS, lineSize, xTable, kernel);
        if (!res)
        {
            s.add(ErrorMemoryAllocationFailed);
        }
        else
        {
            s = res->init(cacheSize, xTable);
            if (!s)
            {
                delete res;
                res = nullptr;
            }
        }
        return res;
    }

    Status compute(const services::Buffer<algorithmFPType>& xBuff, const services::Buffer<int>& wsIndices, const size_t p) override{
        Status status;

        auto cacheBuff = _cache.get<algorithmFPType>();
        DAAL_CHECK_STATUS(status, HelperSVM::copyBlockIndices(xBuff, wsIndices, cacheBuff, _nWS, p));

        DAAL_CHECK_STATUS(status, _kernel->computeNoThrow());
    }

protected:
    SVMCacheOneAPI(const size_t nWS, const size_t lineSize,
        const NumericTablePtr & xTable, const kernel_function::KernelIfacePtr & kernel)
        : super(nWS, lineSize, kernel)
    {}

    Status init(size_t cacheSize, const NumericTablePtr & xTable)
    {
        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

        Status s = super::init();
        if (!s) return s;

        // TODO Check size cache
        _cache = context.allocate(TypeIds::id<algorithmFPType>(), _lineSize*_nWS, &s);
        DAAL_CHECK_STATUS_VAR(s);

        auto cacheBuff = _cache.get<algorithmFPType>();
        cacheTable = SyclHomogenNumericTable<algorithmFPType>::create(cacheBuff, _nWS, _lineSize, &s);

        const size_t p = xTable->getNumberOfColumns();
        _xWS = context.allocate(TypeIds::id<algorithmFPType>(), p*_nWS, &s);
        DAAL_CHECK_STATUS_VAR(s);

        auto xWSBuff = _xWS.get<algorithmFPType>();
        xWSTable = SyclHomogenNumericTable<algorithmFPType>::create(xWSBuff, _nWS, p, &s);

        DAAL_CHECK_STATUS_VAR(s);
        _kernel->getParameter()->computationMode = kernel_function::matrixMatrix;
        _kernel->getInput()->set(kernel_function::X, xWSBuff);
        _kernel->getInput()->set(kernel_function::Y, xTable);

        kernel_function::ResultPtr shRes(new kernel_function::Result());
        shRes->set(kernel_function::values, cacheTable);
        _kernel->setResult(shRes);
        return s;
    }


protected:
    UniversalBuffer _xWS;
};

} // namespace internal

} // namespace training

} // namespace svm

} // namespace algorithms

} // namespace daal

#endif
