/* file: svm_train_result_oneapi.h */
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
//  SVM save result structure implementation
//--
*/

#ifndef __SVM_TRAIN_RESULT_ONEAPI_H__
#define __SVM_TRAIN_RESULT_ONEAPI_H__

#include "service/kernel/service_utils.h"
#include "algorithms/kernel/svm/oneapi/svm_helper_oneapi.h"
#include "service/kernel/oneapi/sum_reducer.h"

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
template <typename algorithmFPType>
class SaveResultModel
{
    using Helper = HelperSVM<algorithmFPType>;

public:
    SaveResultModel(services::Buffer<algorithmFPType> & alphaBuff, const services::Buffer<algorithmFPType> & fBuff,
                    const services::Buffer<algorithmFPType> & yBuff, const algorithmFPType C, const size_t nVectors)
        : _yBuff(yBuff), _coeffBuff(alphaBuff), _fBuff(fBuff), _C(C), _nVectors(nVectors)
    {}

    services::Status init()
    {
        services::Status status;
        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        _tmpValues     = context.allocate(TypeIds::id<algorithmFPType>(), _nVectors, &status);
        DAAL_CHECK_STATUS_VAR(status);
        _mask = context.allocate(TypeIds::id<int>(), _nVectors, &status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_CHECK_STATUS(status, Helper::computeDualCoeffs(_yBuff, _coeffBuff, _nVectors));

        return status;
    }

    services::Status setResultsToModel(const NumericTablePtr & xTable, Model & model) const
    {
        services::Status status;
        model.setNFeatures(xTable->getNumberOfColumns());

        size_t nSV;
        DAAL_CHECK_STATUS(status, setSVCoefficients(nSV, model));
        DAAL_CHECK_STATUS(status, setSVIndices(nSV, model));

        DAAL_CHECK_STATUS(status, setSVDense(model, xTable, nSV));

        /* Calculate bias and write it into model */
        algorithmFPType bias;
        DAAL_CHECK_STATUS(status, calculateBias(_C, bias));
        model.setBias(double(bias));
        return status;
    }

protected:
    services::Status setSVCoefficients(size_t & nSV, Model & model) const
    {
        services::Status status;

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

        auto tmpValuesBuff = _tmpValues.get<algorithmFPType>();
        auto maskBuff      = _mask.get<int>();

        DAAL_CHECK_STATUS(status, Helper::checkNotZero(_coeffBuff, maskBuff, _nVectors));
        nSV = 0;
        DAAL_CHECK_STATUS(status, Helper::gatherValues(maskBuff, _coeffBuff, _nVectors, tmpValuesBuff, nSV));
        printf("nSV %lu\n", nSV);

        NumericTablePtr svCoeffTable = model.getClassificationCoefficients();
        DAAL_CHECK_STATUS(status, svCoeffTable->resize(nSV));

        BlockDescriptor<algorithmFPType> svCoeffBlock;
        DAAL_CHECK_STATUS(status, svCoeffTable->getBlockOfRows(0, nSV, ReadWriteMode::writeOnly, svCoeffBlock));
        auto svCoeffBuff = svCoeffBlock.getBuffer();
        context.copy(svCoeffBuff, 0, tmpValuesBuff, 0, nSV, &status);

        DAAL_CHECK_STATUS(status, svCoeffTable->releaseBlockOfRows(svCoeffBlock));
        return status;
    }

    services::Status setSVIndices(size_t nSV, Model & model) const
    {
        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

        NumericTablePtr svIndicesTable = model.getSupportIndices();
        services::Status status;
        DAAL_CHECK_STATUS(status, svIndicesTable->resize(nSV));

        BlockDescriptor<int> svIndicesBlock;
        DAAL_CHECK_STATUS(status, svIndicesTable->getBlockOfRows(0, nSV, ReadWriteMode::writeOnly, svIndicesBlock));

        int * svIndices = svIndicesBlock.getBlockSharedPtr().get();

        const algorithmFPType * alpha = _coeffBuff.toHost(ReadWriteMode::readOnly).get();

        const algorithmFPType zero(0.0);
        for (size_t i = 0, iSV = 0; i < _nVectors; i++)
        {
            if (alpha[i] != zero)
            {
                svIndices[iSV++] = int(i);
            }
        }

        DAAL_CHECK_STATUS(status, svIndicesTable->releaseBlockOfRows(svIndicesBlock));
        return status;
    }

    services::Status setSVDense(Model & model, const NumericTablePtr & xTable, size_t nSV) const
    {
        services::Status status;

        const size_t nFeatures = xTable->getNumberOfColumns();

        NumericTablePtr svTable = model.getSupportVectors();
        DAAL_CHECK_STATUS(status, svTable->resize(nSV));
        if (nSV == 0) return status;

        BlockDescriptor<algorithmFPType> svBlock;
        DAAL_CHECK_STATUS(status, svTable->getBlockOfRows(0, nSV, ReadWriteMode::writeOnly, svBlock));
        auto svBuff = svBlock.getBuffer();

        NumericTablePtr svIndicesTable = model.getSupportIndices();
        BlockDescriptor<int> svIndicesBlock;
        DAAL_CHECK_STATUS(status, svIndicesTable->getBlockOfRows(0, nSV, ReadWriteMode::readOnly, svIndicesBlock));
        auto svIndicesBuff = svIndicesBlock.getBuffer();

        BlockDescriptor<algorithmFPType> xBlock;
        DAAL_CHECK_STATUS(status, xTable->getBlockOfRows(0, _nVectors, ReadWriteMode::readOnly, xBlock));
        auto xBuff = xBlock.getBuffer();

        DAAL_CHECK_STATUS(status, Helper::copyBlockIndices(xBuff, svIndicesBuff, svBuff, nSV, nFeatures));

        DAAL_CHECK_STATUS(status, svTable->releaseBlockOfRows(svBlock));
        DAAL_CHECK_STATUS(status, svIndicesTable->releaseBlockOfRows(svIndicesBlock));

        return status;
    }

    services::Status calculateBias(const algorithmFPType C, algorithmFPType & bias) const
    {
        services::Status status;

        const algorithmFPType fpMax = MaxVal<algorithmFPType>::get();

        auto tmpValuesBuff = _tmpValues.get<algorithmFPType>();
        auto maskBuff      = _mask.get<int>();

        /* free SV: (0 < alpha < C)*/
        DAAL_CHECK_STATUS(status, Helper::checkFree(_coeffBuff, maskBuff, C, _nVectors));
        size_t nFree = 0;
        DAAL_CHECK_STATUS(status, Helper::gatherValues(maskBuff, _fBuff, _nVectors, tmpValuesBuff, nFree));

        if (nFree > 0)
        {
            auto sumResult = math::SumReducer::sum(math::Layout::RowMajor, tmpValuesBuff, 1, nFree, &status);
            DAAL_CHECK_STATUS_VAR(status);
            auto sumHost = sumResult.sum.get<algorithmFPType>().toHost(data_management::readOnly, &status);
            bias         = -*sumHost / algorithmFPType(nFree);
        }
        else
        {
            algorithmFPType ub = -fpMax;
            algorithmFPType lb = fpMax;
            {
                DAAL_CHECK_STATUS(status, Helper::checkUpper(_yBuff, _coeffBuff, maskBuff, C, _nVectors));
                size_t nUpper = 0;
                DAAL_CHECK_STATUS(status, Helper::gatherValues(maskBuff, _fBuff, _nVectors, tmpValuesBuff, nUpper));
                auto resultOp = math::Reducer::reduce(math::Reducer::BinaryOp::MIN, math::Layout::RowMajor, tmpValuesBuff, 1, nUpper, &status);
                DAAL_CHECK_STATUS_VAR(status);
                UniversalBuffer minU = resultOp.reduceRes;
                auto minHost         = minU.get<algorithmFPType>().toHost(data_management::readOnly, &status);
                ub                   = *minHost;
            }
            {
                DAAL_CHECK_STATUS(status, Helper::checkLower(_yBuff, _coeffBuff, maskBuff, C, _nVectors));
                size_t nLower = 0;
                DAAL_CHECK_STATUS(status, Helper::gatherValues(maskBuff, _fBuff, _nVectors, tmpValuesBuff, nLower));
                auto resultOp = math::Reducer::reduce(math::Reducer::BinaryOp::MAX, math::Layout::RowMajor, tmpValuesBuff, 1, nLower, &status);
                DAAL_CHECK_STATUS_VAR(status);
                UniversalBuffer maxU = resultOp.reduceRes;
                auto maxHost         = maxU.get<algorithmFPType>().toHost(data_management::readOnly, &status);
                lb                   = *maxHost;
            }

            bias = -0.5 * (ub + lb);
        }

        return status;
    }

private:
    const services::Buffer<algorithmFPType> _yBuff;
    const services::Buffer<algorithmFPType> _fBuff;
    services::Buffer<algorithmFPType> _coeffBuff;
    UniversalBuffer _tmpValues;
    UniversalBuffer _mask;
    const algorithmFPType _C;
    const size_t _nVectors;
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
