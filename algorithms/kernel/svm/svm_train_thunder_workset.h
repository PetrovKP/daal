/* file: svm_train_thunder_workset.h */
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
//  SVM workset structure implementation
//--
*/

#ifndef __SVM_TRAIN_THUNDER_WORKSET_H__
#define __SVM_TRAIN_THUNDER_WORKSET_H__

#include "service/kernel/service_utils.h"
#include "algorithms/kernel/service_sort.h"
#include "algorithms/kernel/service_heap.h"

#include <algorithm>

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
using namespace daal::services::internal;

template <typename algorithmFPType, CpuType cpu>
struct TaskWorkingSet
{
    using IndexType = uint32_t;

    TaskWorkingSet(const size_t nVectors, const size_t maxWS) : _nVectors(nVectors), _maxWS(maxWS) {}

    struct IdxValType
    {
        algorithmFPType key;
        IndexType val;
        static int compare(const void * a, const void * b)
        {
            if (static_cast<const IdxValType *>(a)->key < static_cast<const IdxValType *>(b)->key) return -1;
            return static_cast<const IdxValType *>(a)->key > static_cast<const IdxValType *>(b)->key;
        }
        bool operator<(const IdxValType & o) const { return key < o.key; }
        bool operator>(const IdxValType & o) const { return key > o.key; }
    };

    services::Status init()
    {
        services::Status status;
        _sortedFIndices.reset(_nVectors);
        DAAL_CHECK_MALLOC(_sortedFIndices.get());

        _indicator.reset(_nVectors);
        DAAL_CHECK_MALLOC(_indicator.get());
        services::internal::service_memset_seq<bool, cpu>(_indicator.get(), false, _nVectors);

        _nWS       = services::internal::min<cpu, algorithmFPType>(maxPowTwo(_nVectors), _maxWS);
        _nSelected = 0;

        _wsIndices.reset(_nWS);
        DAAL_CHECK_MALLOC(_wsIndices.get());

        return status;
    }

    size_t getSize() const { return _nWS; }

    services::Status copyLastToFirst()
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(select.copyLastToFirst);

        services::Status status;
        const size_t q = _nWS / 2;

        services::internal::daal_memcpy_s(_wsIndices.get(), q * sizeof(IndexType), _wsIndices.get() + _nWS - q, q * sizeof(IndexType));
        _nSelected = q;
        services::internal::service_memset_seq<bool, cpu>(_indicator.get(), false, _nVectors);
        for (size_t i = 0; i < q; ++i)
        {
            _indicator[_wsIndices[i]] = true;
        }

        return status;
    }

    services::Status select(const algorithmFPType * y, const algorithmFPType * alpha, const algorithmFPType * f, const algorithmFPType * cw)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(select);
        services::Status status;
        IdxValType * sortedFIndices = _sortedFIndices.get();
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(select.copy);

            for (size_t i = 0; i < _nVectors; ++i)
            {
                _sortedFIndices[i].key = f[i];
                _sortedFIndices[i].val = i;
            }

            // const size_t blockSize = 1024;
            // const size_t nBlocks   = _nVectors / blockSize + !!(_nVectors % blockSize);
            // daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            //     const size_t startRow = iBlock * blockSize;
            //     const size_t endRow   = (iBlock != nBlocks - 1) ? startRow + blockSize : _nVectors;
            //     for (size_t i = startRow; i < endRow; ++i)
            //     {
            //         sortedFIndices[i].key = f[i];
            //         sortedFIndices[i].val = i;
            //     }
            // });
        }

        // algorithms::internal::makeMaxHeap<cpu>(sortedFIndicesMax, sortedFIndicesMax + _nVectors,
        //                                        [](const IdxValType & a, const IdxValType & b) { return a > b; });

        // std::make_heap(sortedFIndicesMin, sortedFIndicesMin + _nVectors, [](const IdxValType & a, const IdxValType & b) { return a < b; });

        // algorithms::internal::makeMaxHeap<cpu>(sortedFIndicesMax, sortedFIndicesMax + _nVectors,
        //                                        [](const IdxValType & a, const IdxValType & b) { return a < b; });
        // algorithms::internal::makeMaxHeap<cpu>(sortedFIndicesMin, sortedFIndicesMin + _nVectors,
        //                                        [](const IdxValType & a, const IdxValType & b) { return a > b; });

        // int64_t pRight = 0;

        // const size_t rk = _nSelected + (_nWS - _nSelected) / 2;
        // {
        //     int64_t pLeft = 0;
        //     DAAL_ITTNOTIFY_SCOPED_TASK(select.select1);
        //     while (_nSelected < rk && pLeft < _nVectors)
        //     {
        //         if (pLeft < _nVectors)
        //         {
        //             // algorithms::internal::popMaxHeap<cpu>(sortedFIndicesMax, sortedFIndicesMax + 1,
        //             //                                       [](const IdxValType & a, const IdxValType & b) { return a < b; });

        //             IndexType i = sortedFIndicesMax->val;
        //             algorithms::internal::popMaxHeap<cpu>(sortedFIndicesMax, sortedFIndicesMax + _nVectors - pLeft,
        //                                                   [](const IdxValType & a, const IdxValType & b) { return a > b; });
        //             while (_indicator[i] || !HelperTrainSVM<algorithmFPType, cpu>::isUpper(y[i], alpha[i], cw[i]))
        //             {
        //                 pLeft++;
        //                 if (pLeft == _nVectors)
        //                 {
        //                     break;
        //                 }
        //                 i = sortedFIndicesMax->val;
        //                 algorithms::internal::popMaxHeap<cpu>(sortedFIndicesMax, sortedFIndicesMax + _nVectors - pLeft,
        //                                                       [](const IdxValType & a, const IdxValType & b) { return a > b; });

        //                 // algorithms::internal::popMaxHeap<cpu>(sortedFIndicesMax, sortedFIndicesMax + 1,
        //                 //                                       [](const IdxValType & a, const IdxValType & b) { return a < b; });
        //             }
        //             if (pLeft < _nVectors)
        //             {
        //                 _wsIndices[_nSelected] = i;
        //                 _indicator[i]          = true;
        //                 ++_nSelected;
        //             }
        //         }
        //     }
        // }

        // algorithms::internal::makeMaxHeap<cpu>(sortedFIndicesMax, sortedFIndicesMax + _nVectors,
        //                                        [](const IdxValType & a, const IdxValType & b) { return a < b; });
        // {
        //     int64_t pLeft = 0;
        //     DAAL_ITTNOTIFY_SCOPED_TASK(select.select2);
        //     while (_nSelected < _nWS && pLeft < _nVectors)
        //     {
        //         if (pLeft < _nVectors)
        //         {
        //             // algorithms::internal::popMaxHeap<cpu>(sortedFIndicesMax, sortedFIndicesMax + 1,
        //             //                                       [](const IdxValType & a, const IdxValType & b) { return a < b; });

        //             IndexType i = sortedFIndicesMax->val;
        //             algorithms::internal::popMaxHeap<cpu>(sortedFIndicesMax, sortedFIndicesMax + _nVectors - pLeft,
        //                                                   [](const IdxValType & a, const IdxValType & b) { return a < b; });
        //             // printf("%d %.1lf\n", (int)i, sortedFIndicesMax->key);
        //             // ++sortedFIndicesMax;
        //             // IndexType i = sortedFIndices[pLeft].val;
        //             while (_indicator[i] || !HelperTrainSVM<algorithmFPType, cpu>::isLower(y[i], alpha[i], cw[i]))
        //             {
        //                 pLeft++;
        //                 if (pLeft == _nVectors)
        //                 {
        //                     break;
        //                 }
        //                 i = sortedFIndicesMax->val;
        //                 algorithms::internal::popMaxHeap<cpu>(sortedFIndicesMax, sortedFIndicesMax + _nVectors - pLeft,
        //                                                       [](const IdxValType & a, const IdxValType & b) { return a < b; });

        //                 // algorithms::internal::popMaxHeap<cpu>(sortedFIndicesMax, sortedFIndicesMax + 1,
        //                 //                                       [](const IdxValType & a, const IdxValType & b) { return a < b; });
        //                 // printf("%d %.1lf\n", (int)i, sortedFIndicesMax->key);
        //                 // ++sortedFIndicesMax;
        //             }
        //             if (pLeft < _nVectors)
        //             {
        //                 _wsIndices[_nSelected] = i;
        //                 _indicator[i]          = true;
        //                 ++_nSelected;
        //             }
        //         }
        //     }
        // }

        // for (size_t i = 0; i < nWS; i++)
        // {
        //     printf(" \n", );
        // }

        {
            DAAL_ITTNOTIFY_SCOPED_TASK(select.qSortByKey);

            algorithms::internal::qSortByKey<IdxValType, cpu>(_nVectors, sortedFIndices);
        }

        {
            int64_t pLeft  = 0;
            int64_t pRight = _nVectors - 1;
            DAAL_ITTNOTIFY_SCOPED_TASK(select.select);
            while (_nSelected < _nWS && (pRight >= 0 || pLeft < _nVectors))
            {
                if (pLeft < _nVectors)
                {
                    IndexType i = sortedFIndices[pLeft].val;
                    while (_indicator[i] || !HelperTrainSVM<algorithmFPType, cpu>::isUpper(y[i], alpha[i], cw[i]))
                    {
                        pLeft++;
                        if (pLeft == _nVectors)
                        {
                            break;
                        }
                        i = sortedFIndices[pLeft].val;
                    }
                    if (pLeft < _nVectors)
                    {
                        _wsIndices[_nSelected] = i;
                        _indicator[i]          = true;
                        ++_nSelected;
                    }
                }

                if (pRight >= 0)
                {
                    IndexType i = sortedFIndices[pRight].val;
                    while (_indicator[i] || !HelperTrainSVM<algorithmFPType, cpu>::isLower(y[i], alpha[i], cw[i]))
                    {
                        pRight--;
                        if (pRight == -1)
                        {
                            break;
                        }
                        i = sortedFIndices[pRight].val;
                    }
                    if (pRight >= 0)
                    {
                        _wsIndices[_nSelected] = i;
                        _indicator[i]          = true;
                        ++_nSelected;
                    }
                }
            }
        }
        // For cases, when weights are zero
        int64_t pLeft = 0;
        while (_nSelected < _nWS)
        {
            if (!_indicator[pLeft])
            {
                _wsIndices[_nSelected] = pLeft;
                _indicator[pLeft]      = true;
                ++_nSelected;
            }
            ++pLeft;
        }
        DAAL_ASSERT(_nSelected == _nWS);
        _nSelected = 0;
        return status;
    }

    const IndexType * getIndices() const { return _wsIndices.get(); }

protected:
    size_t maxPowTwo(size_t n)
    {
        if (!(n & (n - 1)))
        {
            return n;
        }

        size_t count = 0;
        while (n > 1)
        {
            n >>= 1;
            ++count;
        }
        return 1 << count;
    }

private:
    size_t _nVectors;
    size_t _maxWS;
    size_t _nSelected;
    size_t _nWS;

    TArray<IdxValType, cpu> _sortedFIndices;
    TArray<bool, cpu> _indicator;
    TArray<IndexType, cpu> _wsIndices;
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
