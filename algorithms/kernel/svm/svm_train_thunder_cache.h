/* file: svm_train_thunder_cache.h */
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

#ifndef __SVM_TRAIN_THUNDER_CACHE_H__
#define __SVM_TRAIN_THUNDER_CACHE_H__

#include "service/kernel/service_utils.h"
#include "externals/service_memory.h"
#include "service/kernel/data_management/service_micro_table.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "data_management/data/numeric_table_sycl_homogen.h"
#include "algorithms/kernel/svm/svm_train_cache.h"
#include "externals/service_service.h"
#include "data_management/data/soa_numeric_table.h"

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
using namespace daal::data_management;

/**
 * Common interface for cache that stores kernel function values
 */
template <typename algorithmFPType, CpuType cpu>
class SVMCacheIface<thunder, algorithmFPType, cpu> : public SVMCacheCommonIface<algorithmFPType, cpu>
{
public:
    virtual ~SVMCacheIface() {}

    virtual services::Status getRowsBlock(const uint32_t * const indices, const size_t n, NumericTablePtr & block) = 0;

    virtual size_t getDataRowIndex(size_t rowIndex) const override { return rowIndex; }

protected:
    SVMCacheIface(const size_t cacheSize, const size_t lineSize, const kernel_function::KernelIfacePtr & kernel)
        : _lineSize(lineSize), _cacheSize(cacheSize), _kernel(kernel)
    {}

    const size_t _lineSize;                        /*!< Number of elements in the cache line */
    const size_t _cacheSize;                       /*!< Number of cache lines */
    const kernel_function::KernelIfacePtr _kernel; /*!< Kernel function */
};

/**
 * No cache: kernel function values are not cached
 */
template <typename algorithmFPType, CpuType cpu>
class SVMCache<thunder, lruCache, algorithmFPType, cpu> : public SVMCacheIface<thunder, algorithmFPType, cpu>
{
    using super    = SVMCacheIface<thunder, algorithmFPType, cpu>;
    using thisType = SVMCache<thunder, lruCache, algorithmFPType, cpu>;
    using super::_kernel;
    using super::_lineSize;
    using super::_cacheSize;

public:
    ~SVMCache() {}

    DAAL_NEW_DELETE();

    static SVMCachePtr<thunder, algorithmFPType, cpu> create(const size_t cacheSize, const size_t nSize, const size_t lineSize,
                                                             const NumericTablePtr & xTable, const kernel_function::KernelIfacePtr & kernel,
                                                             services::Status & status)
    {
        services::SharedPtr<thisType> res = services::SharedPtr<thisType>(new thisType(cacheSize, lineSize, xTable, kernel));
        if (!res)
        {
            status.add(ErrorMemoryAllocationFailed);
        }
        else
        {
            status = res->init(nSize);
            if (!status)
            {
                res.reset();
            }
        }
        return SVMCachePtr<thunder, algorithmFPType, cpu>(res);
    }

    services::Status getRowsBlock(const uint32_t * const indices, const size_t n, NumericTablePtr & block) override
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(cache.getRowsBlock);

        services::Status status;
        auto kernelResultTable   = SOANumericTable::create(n, _lineSize, DictionaryIface::FeaturesEqual::equal, &status);
        size_t nIndicesForKernel = 0;

        for (int i = 0; i < n; ++i)
        {
            int64_t cacheIndex = _lruCache.get(indices[i]);
            if (cacheIndex != -1)
            {
                DAAL_ASSERT(cacheIndex < _cacheSize)
                auto cachei = services::reinterpretPointerCast<algorithmFPType, byte>(_cache->getArraySharedPtr(cacheIndex));
                kernelResultTable->template setArray<algorithmFPType>(cachei, i);
            }
            else
            {
                _lruCache.put(indices[i]);
                cacheIndex = _lruCache.getFreeIndex();
                DAAL_ASSERT(cacheIndex < _cacheSize)
                auto cachei = services::reinterpretPointerCast<algorithmFPType, byte>(_cache->getArraySharedPtr(cacheIndex));
                kernelResultTable->template setArray<algorithmFPType>(cachei, i);
                // if (i >= n - 16) printf("%d ", (int)cacheIndex);
                _kernelIndex[nIndicesForKernel]         = cacheIndex;
                _kernelOriginalIndex[nIndicesForKernel] = indices[i];
                ++nIndicesForKernel;
            }
        }
        // printf("\n");

        // printf("\n nCountForKernel %lu _nComputeIndices %lu _nSelected %lu\n", nCountForKernel, _nComputeIndices, _nSelected);

        // for (int i = 0; i < 16; i++) printf("%d ", (int)_kernelIndex[i]);
        // printf("\n");
        if (nIndicesForKernel != 0)
        {
            DAAL_CHECK_STATUS(status, computeKernel(nIndicesForKernel, _kernelOriginalIndex.get()));
        }
        // printf("> [reinit] - finish\n");
        block = kernelResultTable;
        return status;
    }

protected:
    SVMCache(const size_t cacheSize, const size_t lineSize, const NumericTablePtr & xTable, const kernel_function::KernelIfacePtr & kernel)
        : super(cacheSize, lineSize, kernel), _lruCache(cacheSize), _xTable(xTable)
    {}

    services::Status computeKernel(const size_t nWorkElements, const uint32_t * indices)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(cache.getRowsBlock.computeKernel);

        services::Status status;

        _kernelComputeTable = SOANumericTable::create(nWorkElements, _lineSize, DictionaryIface::FeaturesEqual::equal, &status);

        // printf(">> [computeKernel] _nSelected %lu nWorkElements %lu\n", _nSelected, nWorkElements);

        for (size_t i = 0; i < nWorkElements; ++i)
        {
            const size_t kernelInd = _kernelIndex[i];
            // if (i < 16 || i > nWorkElements - 16) printf("%lu ", kernelInd);
            auto cachei = services::reinterpretPointerCast<algorithmFPType, byte>(_cache->getArraySharedPtr(kernelInd));
            _kernelComputeTable->template setArray<algorithmFPType>(cachei, i);
        }
        // printf("\n");

        const size_t p = _xTable->getNumberOfColumns();
        DAAL_CHECK_STATUS_VAR(status);

        SubDataTaskBase<algorithmFPType, cpu> * task = nullptr;
        if (_xTable->getDataLayout() == NumericTableIface::csrArray)
        {
            task = SubDataTaskCSR<algorithmFPType, cpu>::create(_xTable, nWorkElements);
        }
        else
        {
            task = SubDataTaskDense<algorithmFPType, cpu>::create(p, nWorkElements);
        }

        DAAL_CHECK_MALLOC(task);
        _blockTask = SubDataTaskBasePtr<algorithmFPType, cpu>(task);
        DAAL_CHECK_STATUS(status, _blockTask->copyDataByIndices(indices, _xTable));

        DAAL_CHECK_STATUS_VAR(status);
        _kernel->getParameter()->computationMode = kernel_function::matrixMatrix;

        _kernel->getInput()->set(kernel_function::X, _xTable);
        _kernel->getInput()->set(kernel_function::Y, _blockTask->getTableData());

        // _kernel->getInput()->set(kernel_function::X, _blockTask->getTableData());
        // _kernel->getInput()->set(kernel_function::Y, _xTable);

        kernel_function::ResultPtr shRes(new kernel_function::Result());
        shRes->set(kernel_function::values, _kernelComputeTable);
        _kernel->setResult(shRes);
        // printf(">> [reinit - start] _kernel->computeNoThrow(\n");

        {
            DAAL_ITTNOTIFY_SCOPED_TASK(cache.getRowsBlock.computeKernel.compute);
            DAAL_CHECK_STATUS(status, _kernel->computeNoThrow());
        }
        // printf(">> [reinit - finish] _kernel->computeNoThrow(\n");

        // ReadRows<algorithmFPType, cpu> mtData(kernelComputeTable.get(), 0, 2);
        // const algorithmFPType * res = mtData.get();
        // for (int i = 0; i < 16; i++) printf("%lf ", res[i]);
        // printf("\n");

        return status;
    }

    services::Status init(const size_t nSize)
    {
        services::Status status;
        printf("cacheSize %lu _lineSize %lu nSize %lu\n", _cacheSize, _lineSize, nSize);
        _kernelIndex.reset(nSize);
        DAAL_CHECK_MALLOC(_kernelIndex.get());
        _kernelOriginalIndex.reset(nSize);
        DAAL_CHECK_MALLOC(_kernelOriginalIndex.get());

        auto dict = NumericTableDictionary::create(_cacheSize, DictionaryIface::FeaturesEqual::equal, &status);
        DAAL_CHECK_STATUS_VAR(status);
        DAAL_CHECK_STATUS(status, dict->template setAllFeatures<algorithmFPType>());
        _cache = SOANumericTable::create(dict, _lineSize, NumericTable::AllocationFlag::doAllocate, &status);
        DAAL_CHECK_STATUS_VAR(status);
        return status;
    }

protected:
    LRUCache<cpu, uint32_t> _lruCache;
    const NumericTablePtr & _xTable;
    SubDataTaskBasePtr<algorithmFPType, cpu> _blockTask;
    TArray<uint32_t, cpu> _kernelOriginalIndex;
    TArray<uint32_t, cpu> _kernelIndex;
    services::SharedPtr<SOANumericTable> _cache;
    services::SharedPtr<SOANumericTable> _kernelComputeTable;
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
