/* file: svm_train_thunder_impl.i */
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
//  SVM training algorithm implementation using Thunder method
//--
*/
/*
//  DESCRIPTION
//
//  Definition of the functions for training with SVM 2-class classifier.
//
//  REFERENCES
//
//  1. Zeyi Wen, Jiashuai Shi, Bingsheng He
//     ThunderSVM: A Fast SVM Library on GPUs and CPUs,
//     Journal of Machine Learning Research, 19, 1-5 (2018)
//  2. Rong-En Fan, Pai-Hsuen Chen, Chih-Jen Lin,
//     Working Set Selection Using Second Order Information
//     for Training Support Vector Machines,
//     Journal of Machine Learning Research 6 (2005), pp. 1889___1918
//  3. Bernard E. boser, Isabelle M. Guyon, Vladimir N. Vapnik,
//     A Training Algorithm for Optimal Margin Classifiers.
//  4. Thorsten Joachims, Making Large-Scale SVM Learning Practical,
//     Advances in Kernel Methods - Support Vector Learning
*/

#ifndef __SVM_TRAIN_THUNDER_I__
#define __SVM_TRAIN_THUNDER_I__

#include "externals/service_memory.h"
#include "service/kernel/data_management/service_micro_table.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "service/kernel/service_utils.h"
#include "service/kernel/service_data_utils.h"
#include "externals/service_ittnotify.h"
#include "externals/service_blas.h"
#include "externals/service_math.h"

#include "algorithms/kernel/svm/svm_train_common.h"
#include "algorithms/kernel/svm/svm_train_thunder_workset.h"
#include "algorithms/kernel/svm/svm_train_thunder_cache.h"
#include "algorithms/kernel/svm/svm_train_result.h"

#include "algorithms/kernel/svm/svm_train_common_impl.i"

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
template <typename algorithmFPType, typename ParameterType, CpuType cpu>
services::Status SVMTrainImpl<thunder, algorithmFPType, ParameterType, cpu>::compute(const NumericTablePtr & xTable, const NumericTablePtr & wTable,
                                                                                     NumericTable & yTable, daal::algorithms::Model * r,
                                                                                     const ParameterType * svmPar)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(COMPUTE);

    services::Status status;

    const algorithmFPType C(svmPar->C);
    const algorithmFPType eps(svmPar->accuracyThreshold);
    const algorithmFPType tau(svmPar->tau);
    const size_t maxIterations(svmPar->maxIterations);
    const size_t cacheSize(svmPar->cacheSize);
    kernel_function::KernelIfacePtr kernel = svmPar->kernel->clone();

    const size_t nVectors = xTable->getNumberOfRows();

    TArray<algorithmFPType, cpu> alphaTArray(nVectors);
    DAAL_CHECK_MALLOC(alphaTArray.get());
    algorithmFPType * const alpha = alphaTArray.get();

    TArray<algorithmFPType, cpu> gradTArray(nVectors);
    DAAL_CHECK_MALLOC(gradTArray.get());
    algorithmFPType * const grad = gradTArray.get();

    TArray<algorithmFPType, cpu> cwTArray(nVectors);
    DAAL_CHECK_MALLOC(cwTArray.get());
    algorithmFPType * const cw = cwTArray.get();

    TArray<algorithmFPType, cpu> yTArray(nVectors);
    DAAL_CHECK_MALLOC(yTArray.get());
    algorithmFPType * const y = yTArray.get();

    SafeStatus safeStat;

    size_t nNonZeroWeights = nVectors;
    {
        const size_t blockSize = 2048;
        const size_t nBlocks   = nVectors / blockSize + !!(nVectors % blockSize);

        DAAL_ITTNOTIFY_SCOPED_TASK(init.set);
        TArrayScalable<size_t, cpu> weightsCounter;
        if (wTable.get())
        {
            weightsCounter.reset(blockSize);
            DAAL_CHECK_MALLOC(weightsCounter.get());
        }

        daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            const size_t startRow     = iBlock * blockSize;
            const size_t nRowsInBlock = (iBlock != nBlocks - 1) ? blockSize : nVectors - iBlock * blockSize;

            ReadColumns<algorithmFPType, cpu> mtY(yTable, 0, startRow, nRowsInBlock);
            DAAL_CHECK_BLOCK_STATUS_THR(mtY);
            const algorithmFPType * const yIn = mtY.get();

            ReadColumns<algorithmFPType, cpu> mtW(wTable.get(), 0, startRow, nRowsInBlock);
            DAAL_CHECK_BLOCK_STATUS_THR(mtW);
            const algorithmFPType * weights = mtW.get();

            if (weights)
            {
                weightsCounter[iBlock] = 0;
            }
            for (size_t i = 0; i < nRowsInBlock; ++i)
            {
                y[i + startRow]     = yIn[i] == 0 ? algorithmFPType(-1) : yIn[i];
                grad[i + startRow]  = -y[i + startRow];
                alpha[i + startRow] = algorithmFPType(0);
                cw[i + startRow]    = weights ? weights[i] * C : C;
                if (weights)
                {
                    weightsCounter[iBlock] += static_cast<size_t>(weights[i] != algorithmFPType(0));
                }
            }
        });

        if (wTable.get())
        {
            nNonZeroWeights = 0;
            for (size_t iBlock = 0; iBlock < nBlocks; iBlock++)
            {
                nNonZeroWeights += weightsCounter[iBlock];
            }
        }
    }

    TaskWorkingSet<algorithmFPType, cpu> workSet(nNonZeroWeights, nVectors, maxBlockSize);
    DAAL_CHECK_STATUS(status, workSet.init());
    const size_t nWS = workSet.getSize();

    algorithmFPType diff     = algorithmFPType(0);
    algorithmFPType diffPrev = algorithmFPType(0);
    size_t sameLocalDiff     = 0;

    TArray<algorithmFPType, cpu> buffer(nWS * MemSmoId::latest + nWS * nWS);
    DAAL_CHECK_MALLOC(buffer.get());

    TArray<algorithmFPType, cpu> deltaAlpha(nWS);

    DAAL_CHECK_MALLOC(deltaAlpha.get());

    SVMCachePtr<thunder, algorithmFPType, cpu> cachePtr;

    TArray<char, cpu> I(nWS);
    DAAL_CHECK_MALLOC(I.get());

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nVectors * sizeof(algorithmFPType), nVectors);

    // const size_t defaultCacheSize = services::internal::max<cpu, algorithmFPType>(nWS, cacheSize / nVectors / sizeof(algorithmFPType));
    // const size_t defaultCacheSize = nVectors;
    size_t defaultCacheSize = services::internal::min<cpu, algorithmFPType>(nVectors, cacheSize / nVectors / sizeof(algorithmFPType));
    defaultCacheSize        = services::internal::max<cpu, algorithmFPType>(nWS, defaultCacheSize);
    cachePtr                = SVMCache<thunder, lruCache, algorithmFPType, cpu>::create(defaultCacheSize, nWS, nVectors, xTable, kernel, status);

    TArrayScalable<algorithmFPType, cpu> gradtmp(256 * nVectors);
    DAAL_CHECK_MALLOC(gradtmp.get());

    size_t iter = 0;
    for (; iter < maxIterations; ++iter)
    {
        if (iter != 0)
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(copyLastToFirst);

            DAAL_CHECK_STATUS(status, workSet.copyLastToFirst());

            DAAL_CHECK_STATUS(status, cachePtr->copyLastToFirst());
        }

        {
            DAAL_CHECK_STATUS(status, workSet.select(y, alpha, grad, cw));
        }
        const uint32_t * wsIndices = workSet.getIndices();
        NumericTablePtr kernelWS;
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(getRowsBlock);

            DAAL_CHECK_STATUS(status, cachePtr->getRowsBlock(wsIndices, nWS, kernelWS));
        }
        DAAL_CHECK_STATUS(
            status, SMOBlockSolver(y, grad, wsIndices, kernelWS, nVectors, nWS, cw, eps, tau, buffer.get(), I.get(), alpha, deltaAlpha.get(), diff));

        DAAL_CHECK_STATUS(status, updateGrad(kernelWS, deltaAlpha.get(), gradtmp.get(), grad, nVectors, nWS));

        if (checkStopCondition(diff, diffPrev, eps, sameLocalDiff) && iter >= nNoChanges) break;
        diffPrev = diff;
    }
    printf("nIter: %lu; diff: %.3lf\n", iter, diff);
    SaveResultTask<algorithmFPType, cpu> saveResult(nVectors, y, alpha, grad, cachePtr.get());
    DAAL_CHECK_STATUS(status, saveResult.compute(*xTable, *static_cast<Model *>(r), cw));

    return status;
}

template <typename algorithmFPType, typename ParameterType, CpuType cpu>
services::Status SVMTrainImpl<thunder, algorithmFPType, ParameterType, cpu>::SMOBlockSolver(
    const algorithmFPType * y, const algorithmFPType * grad, const uint32_t * wsIndices, const NumericTablePtr & kernelWS, const size_t nVectors,
    const size_t nWS, const algorithmFPType * cw, const double eps, const double tau, algorithmFPType * buffer, char * I, algorithmFPType * alpha,
    algorithmFPType * deltaAlpha, algorithmFPType & localDiff) const
{
    DAAL_ITTNOTIFY_SCOPED_TASK(SMOBlockSolver);
    services::Status status;

    const size_t innerMaxIterations(nWS * cInnerIterations);

    algorithmFPType * const alphaLocal    = buffer + nWS * MemSmoId::alphaBuffID;
    algorithmFPType * const yLocal        = buffer + nWS * MemSmoId::yBuffID;
    algorithmFPType * const gradLocal     = buffer + nWS * MemSmoId::gradBuffID;
    algorithmFPType * const kdLocal       = buffer + nWS * MemSmoId::kdBuffID;
    algorithmFPType * const oldAlphaLocal = buffer + nWS * MemSmoId::oldAlphaBuffID;
    algorithmFPType * const cwLocal       = buffer + nWS * MemSmoId::cwBuffID;
    algorithmFPType * const kernelLocal   = buffer + nWS * MemSmoId::latest;

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(SMOBlockSolver.init);
        SafeStatus safeStat;

        /* Gather data to local buffers */
        const size_t blockSize = services::internal::min<cpu, algorithmFPType>(nWS, 16);
        const size_t nBlocks   = nWS / blockSize;
        daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            const size_t startRow = iBlock * blockSize;

            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = startRow; i < startRow + blockSize; ++i)
            {
                const size_t wsIndex = wsIndices[i];

                ReadColumns<algorithmFPType, cpu> mtKernel(kernelWS.get(), i, 0, nVectors);
                DAAL_CHECK_BLOCK_STATUS_THR(mtKernel);
                const algorithmFPType * const kernelWSData = mtKernel.get();
                yLocal[i]                                  = y[wsIndex];
                gradLocal[i]                               = grad[wsIndex];
                oldAlphaLocal[i]                           = alpha[wsIndex];
                alphaLocal[i]                              = alpha[wsIndex];
                cwLocal[i]                                 = cw[wsIndex];

                kdLocal[i] = kernelWSData[wsIndex];
                // if (i < 16) printf("%.2lf\n", kdLocal[i]);
                char Ii = free;
                Ii |= HelperTrainSVM<algorithmFPType, cpu>::isUpper(yLocal[i], alphaLocal[i], cwLocal[i]) ? up : free;
                Ii |= HelperTrainSVM<algorithmFPType, cpu>::isLower(yLocal[i], alphaLocal[i], cwLocal[i]) ? low : free;
                I[i] = Ii;
                for (size_t j = 0; j < nWS; ++j)
                {
                    kernelLocal[i * nWS + j] = kernelWSData[wsIndices[j]];
                }
            }
        });
    }

    algorithmFPType delta    = algorithmFPType(0);
    algorithmFPType localEps = algorithmFPType(0);
    localDiff                = algorithmFPType(0);
    int Bi                   = -1;
    int Bj                   = -1;

    size_t iter = 0;
    for (; iter < innerMaxIterations; ++iter)
    {
        algorithmFPType GMin  = HelperTrainSVM<algorithmFPType, cpu>::WSSi(nWS, gradLocal, I, Bi);
        algorithmFPType GMax  = -MaxVal<algorithmFPType>::get();
        algorithmFPType GMax2 = -MaxVal<algorithmFPType>::get();

        const algorithmFPType zero(0.0);
        const algorithmFPType KBiBi            = kdLocal[Bi];
        const algorithmFPType * const KBiBlock = &kernelLocal[Bi * nWS];

        HelperTrainSVM<algorithmFPType, cpu>::WSSjLocal(0, nWS, KBiBlock, kdLocal, gradLocal, I, GMin, KBiBi, tau, Bj, GMax, GMax2, delta);

        localDiff = GMax2 - GMin;

        if (iter == 0)
        {
            localEps = services::internal::max<cpu, algorithmFPType>(eps, localDiff * algorithmFPType(1e-1));
        }
        if (localDiff < localEps)
        {
            break;
        }

        const algorithmFPType yBi  = yLocal[Bi];
        const algorithmFPType yBj  = yLocal[Bj];
        const algorithmFPType cwBi = cwLocal[Bi];
        const algorithmFPType cwBj = cwLocal[Bj];

        /* Update coefficients */
        const algorithmFPType alphaBiDelta = (yBi > 0.0f) ? cwBi - alphaLocal[Bi] : alphaLocal[Bi];
        const algorithmFPType alphaBjDelta =
            services::internal::min<cpu, algorithmFPType>((yBj > 0.0f) ? alphaLocal[Bj] : cwBj - alphaLocal[Bj], delta);
        delta = services::internal::min<cpu, algorithmFPType>(alphaBiDelta, alphaBjDelta);

        /* Update alpha */
        alphaLocal[Bi] += delta * yBi;
        alphaLocal[Bj] -= delta * yBj;

        /* Update up/low sets */
        char IBi = free;
        IBi |= HelperTrainSVM<algorithmFPType, cpu>::isUpper(yBi, alphaLocal[Bi], cwBi) ? up : free;
        IBi |= HelperTrainSVM<algorithmFPType, cpu>::isLower(yBi, alphaLocal[Bi], cwBi) ? low : free;
        I[Bi] = IBi;

        char IBj = free;
        IBj |= HelperTrainSVM<algorithmFPType, cpu>::isUpper(yBj, alphaLocal[Bj], cwBj) ? up : free;
        IBj |= HelperTrainSVM<algorithmFPType, cpu>::isLower(yBj, alphaLocal[Bj], cwBj) ? low : free;
        I[Bj] = IBj;

        const algorithmFPType * const KBjBlock = &kernelLocal[Bj * nWS];

        /* Update gradient */
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nWS; i++)
        {
            const algorithmFPType KiBi = KBiBlock[i];
            const algorithmFPType KiBj = KBjBlock[i];
            gradLocal[i] += delta * (KiBi - KiBj);
        }
    }

    // printf("innerIter: %lu delta: %lf\n", iter, delta);

    /* Compute diff and scatter to alpha vector */
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nWS; ++i)
    {
        deltaAlpha[i]       = (alphaLocal[i] - oldAlphaLocal[i]) * yLocal[i];
        alpha[wsIndices[i]] = alphaLocal[i];
    }
    return status;
}

template <typename algorithmFPType, typename ParameterType, CpuType cpu>
services::Status SVMTrainImpl<thunder, algorithmFPType, ParameterType, cpu>::updateGrad(const NumericTablePtr & kernelWS,
                                                                                        const algorithmFPType * deltaalpha, algorithmFPType * tmpgrad,
                                                                                        algorithmFPType * grad, const size_t nVectors,
                                                                                        const size_t nWS)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(updateGrad2);

    SafeStatus safeStat;
    const size_t blockSizeWS = services::internal::min<cpu, algorithmFPType>(nWS, 128);
    const size_t nBlocksWS   = nWS / blockSizeWS;

    const size_t blockSizeGrad = 256;
    const size_t nBlocksGrad   = nVectors / blockSizeGrad + !!(nVectors % blockSizeGrad);

    DAAL_INT incX(1);
    DAAL_INT incY(1);
    DAAL_ITTNOTIFY_SCOPED_TASK(updateGrad2.threader_for);

    daal::threader_for(nBlocksWS, nBlocksWS, [&](const size_t iBlock) {
        const size_t startRowWS           = iBlock * blockSizeWS;
        const algorithmFPType deltaalphai = deltaalpha[startRowWS];

        daal::threader_for(nBlocksGrad, nBlocksGrad, [&](const size_t iBlockGrad) {
            const size_t nRowsInBlockGrad = (iBlockGrad != nBlocksGrad - 1) ? blockSizeGrad : nVectors - iBlockGrad * blockSizeGrad;
            const size_t startRowGrad     = iBlockGrad * blockSizeGrad;
            algorithmFPType * gradi       = &tmpgrad[nVectors * iBlock + startRowGrad];

            ReadColumns<algorithmFPType, cpu> mtKernel(kernelWS.get(), startRowWS, startRowGrad, nRowsInBlockGrad);
            DAAL_CHECK_BLOCK_STATUS_THR(mtKernel);
            const algorithmFPType * kernelBlockI = mtKernel.get();
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < nRowsInBlockGrad; j++)
            {
                gradi[j] = deltaalphai * kernelBlockI[j];
            }

            for (size_t i = 1; i < blockSizeWS; i++)
            {
                ReadColumns<algorithmFPType, cpu> mtKernel(kernelWS.get(), startRowWS + i, startRowGrad, nRowsInBlockGrad);
                DAAL_CHECK_BLOCK_STATUS_THR(mtKernel);
                algorithmFPType * kernelBlockI = (algorithmFPType *)mtKernel.get();
                algorithmFPType deltaalphai    = deltaalpha[startRowWS + i];
                Blas<algorithmFPType, cpu>::xxaxpy((DAAL_INT *)&nRowsInBlockGrad, &deltaalphai, kernelBlockI, &incX, gradi, &incY);
            }
        });
    });

    DAAL_ITTNOTIFY_SCOPED_TASK(updateGrad2.reduce);
    algorithmFPType one = algorithmFPType(1);
    for (size_t i = 0; i < nBlocksWS; i++)
    {
        Blas<algorithmFPType, cpu>::xxaxpy((DAAL_INT *)&nVectors, &one, &tmpgrad[i * nVectors], &incX, grad, &incY);
    }

    return services::Status();
}

template <typename algorithmFPType, typename ParameterType, CpuType cpu>
bool SVMTrainImpl<thunder, algorithmFPType, ParameterType, cpu>::checkStopCondition(const algorithmFPType diff, const algorithmFPType diffPrev,
                                                                                    const algorithmFPType eps, size_t & sameLocalDiff)
{
    sameLocalDiff = internal::Math<algorithmFPType, cpu>::sFabs(diff - diffPrev) < eps * 1e-3 ? sameLocalDiff + 1 : 0;
    if (sameLocalDiff > nNoChanges || diff < eps)
    {
        return true;
    }
    return false;
}

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
