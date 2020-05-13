/* file: sgd_dense_minibatch_oneapi_impl.i */
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
//  Implementation of SGD dense minibatch Batch algorithm on GPU.
//--
*/

#include "algorithms/kernel/optimization_solver/sgd/oneapi/cl_kernel/sgd_dense_minibatch.cl"
#include "algorithms/kernel/optimization_solver/iterative_solver_kernel.h"
#include "data_management/data/numeric_table_sycl_homogen.h"
#include "externals/service_math.h"
#include "service/kernel/oneapi/reducer.h"

#include "externals/service_ittnotify.h"

DAAL_ITTNOTIFY_DOMAIN(optimization_solver.sgd.batch.oneapi);

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
namespace internal
{
using namespace daal::oneapi::internal;
using namespace daal::data_management;
using namespace daal::oneapi::internal::math;

template <typename algorithmFPType>
inline algorithmFPType max(const algorithmFPType a, const algorithmFPType b)
{
    return (a < b) ? b : a;
}

template <typename algorithmFPType>
services::Status SGDKernelOneAPI<algorithmFPType, miniBatch>::makeStep(const uint32_t argumentSize,
                                                                       const services::Buffer<algorithmFPType> & prevWorkValueBuff,
                                                                       const services::Buffer<algorithmFPType> & gradientBuff,
                                                                       services::Buffer<algorithmFPType> & workValueBuff,
                                                                       const algorithmFPType learningRate, const algorithmFPType consCoeff)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(makeStep);
    services::Status status;

    ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    buildProgram(factory);

    const char * const kernelName = "makeStep";
    KernelPtr kernel              = factory.getKernel(kernelName);

    KernelArguments args(5);
    args.set(0, gradientBuff, AccessModeIds::read);
    args.set(1, prevWorkValueBuff, AccessModeIds::read);
    args.set(2, workValueBuff, AccessModeIds::readwrite);
    args.set(3, learningRate);
    args.set(4, consCoeff);

    KernelRange range(argumentSize);
    ctx.run(range, kernel, args, &status);

    return status;
}

template <typename algorithmFPType>
services::Status SGDKernelOneAPI<algorithmFPType, miniBatch>::vectorNorm(const services::Buffer<algorithmFPType> & x, const uint32_t n,
                                                                         algorithmFPType & norm)
{
    services::Status status;
    auto resultOp = Reducer::reduce(Reducer::BinaryOp::SUM_OF_SQUARES, Layout::RowMajor, x, 1, n, &status);
    DAAL_CHECK_STATUS_VAR(status);
    UniversalBuffer sumOfSq = resultOp.reduceRes;
    auto sumOfSqHost        = sumOfSq.get<algorithmFPType>().toHost(data_management::readOnly, &status);
    norm                    = *sumOfSqHost;
    norm                    = sqrt(norm);
    return status;
}

template <typename algorithmFPType>
void SGDKernelOneAPI<algorithmFPType, miniBatch>::buildProgram(ClKernelFactoryIface & factory)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(buildProgram);
    services::String options = getKeyFPType<algorithmFPType>();

    services::String cachekey("__daal_algorithms_optimization_solver_sgd_");
    cachekey.add(options);
    options.add(" -D LOCAL_SUM_SIZE=256 ");

    factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelSGDMiniBatch, options.c_str());
}

template <typename algorithmFPType>
services::Status SGDKernelOneAPI<algorithmFPType, miniBatch>::compute(HostAppIface * pHost, NumericTable * inputArgument, NumericTablePtr minimum,
                                                                      NumericTable * nIterations, Parameter<miniBatch> * parameter,
                                                                      NumericTable * learningRateSequence, NumericTable * batchIndices,
                                                                      OptionalArgument * optionalArgument, OptionalArgument * optionalResult,
                                                                      engines::BatchBase & engine)
{
    services::Status status;

    ExecutionContextIface & ctx = services::Environment::getInstance()->getDefaultExecutionContext();

    const size_t argumentSize = inputArgument->getNumberOfRows();
    const size_t nIter        = parameter->nIterations;
    const size_t L            = parameter->innerNIterations;
    const size_t batchSize    = parameter->batchSize;

    BlockDescriptor<int> nIterationsBD;
    DAAL_CHECK_STATUS(status, nIterations->getBlockOfRows(0, 1, ReadWriteMode::writeOnly, nIterationsBD));

    int * nProceededIterations = nIterationsBD.getBlockPtr();
    // if nIter == 0, set result as start point, the number of executed iters to 0
    if (nIter == 0 || L == 0)
    {
        nProceededIterations[0] = 0;
        return status;
    }

    NumericTable * lastIterationInput = optionalArgument ? NumericTable::cast(optionalArgument->get(iterative_solver::lastIteration)).get() : nullptr;
    NumericTable * pastWorkValueInput = optionalArgument ? NumericTable::cast(optionalArgument->get(sgd::pastWorkValue)).get() : nullptr;

    NumericTable * lastIterationResult = optionalResult ? NumericTable::cast(optionalResult->get(iterative_solver::lastIteration)).get() : nullptr;
    NumericTable * pastWorkValueResult = optionalResult ? NumericTable::cast(optionalResult->get(sgd::pastWorkValue)).get() : nullptr;

    const double accuracyThreshold = parameter->accuracyThreshold;

    sum_of_functions::BatchPtr function = parameter->function;
    const size_t nTerms                 = function->sumOfFunctionsParameter->numberOfTerms;

    DAAL_ASSERT(minimum->getNumberOfRows() == argumentSize);

    BlockDescriptor<algorithmFPType> workValueBD;
    DAAL_CHECK_STATUS(status, minimum->getBlockOfRows(0, argumentSize, ReadWriteMode::readWrite, workValueBD));
    services::Buffer<algorithmFPType> workValueBuff = workValueBD.getBuffer();

    auto workValueSNT = SyclHomogenNumericTable<algorithmFPType>::create(workValueBuff, 1, argumentSize);

    NumericTablePtr previousArgument = function->sumOfFunctionsInput->get(sum_of_functions::argument);
    function->sumOfFunctionsInput->set(sum_of_functions::argument, workValueSNT);

    BlockDescriptor<algorithmFPType> learningRateBD;
    DAAL_CHECK_STATUS(status, learningRateSequence->getBlockOfRows(0, 1, ReadWriteMode::readOnly, learningRateBD));
    const algorithmFPType * const learningRateArray = learningRateBD.getBlockPtr();

    NumericTable * conservativeSequence = parameter->conservativeSequence.get();

    BlockDescriptor<algorithmFPType> consCoeffsBD;
    DAAL_CHECK_STATUS(status, conservativeSequence->getBlockOfRows(0, 1, ReadWriteMode::readOnly, consCoeffsBD));
    const algorithmFPType * const consCoeffsArray = consCoeffsBD.getBlockPtr();

    const size_t consCoeffsLength   = conservativeSequence->getNumberOfColumns();
    const size_t learningRateLength = learningRateSequence->getNumberOfColumns();

    const IndicesStatus indicesStatus = (batchIndices ? user : (batchSize < nTerms ? random : all));
    services::SharedPtr<HomogenNumericTableCPU<int, sse2> > ntBatchIndices;

    if (indicesStatus == user || indicesStatus == random)
    {
        // Replace by SyclNumericTable when will be RNG on GPU
        ntBatchIndices = HomogenNumericTableCPU<int, sse2>::create(batchSize, 1, &status);
    }

    NumericTablePtr previousBatchIndices            = function->sumOfFunctionsParameter->batchIndices;
    function->sumOfFunctionsParameter->batchIndices = ntBatchIndices;

    const TypeIds::Id idType                            = TypeIds::id<algorithmFPType>();
    UniversalBuffer prevWorkValueU                      = ctx.allocate(idType, argumentSize, &status);
    services::Buffer<algorithmFPType> prevWorkValueBuff = prevWorkValueU.get<algorithmFPType>();

    size_t startIteration = 0, nProceededIters = 0;
    if (lastIterationInput)
    {
        BlockDescriptor<int> lastIterationInputBD;
        DAAL_CHECK_STATUS(status, lastIterationInput->getBlockOfRows(0, 1, ReadWriteMode::readOnly, learningRateBD));

        const int * lastIterationInputArray = lastIterationInputBD.getBlockPtr();
        startIteration                      = lastIterationInputArray[0];
        DAAL_CHECK_STATUS(status, lastIterationInput->releaseBlockOfRows(lastIterationInputBD));
    }

    if (pastWorkValueInput)
    {
        BlockDescriptor<algorithmFPType> pastWorkValueInputBD;
        pastWorkValueInput->getBlockOfRows(0, argumentSize, ReadWriteMode::readOnly, pastWorkValueInputBD);

        const services::Buffer<algorithmFPType> pastWorkValueInputBuff = pastWorkValueInputBD.getBuffer();

        ctx.copy(prevWorkValueBuff, 0, pastWorkValueInputBuff, 0, argumentSize, &status);
        DAAL_CHECK_STATUS(status, pastWorkValueInput->releaseBlockOfRows(pastWorkValueInputBD));
    }
    else
    {
        ctx.fill(prevWorkValueU, 0.0, &status);
    }

    // init workValue
    BlockDescriptor<algorithmFPType> startValueBD;
    DAAL_CHECK_STATUS(status, inputArgument->getBlockOfRows(0, argumentSize, ReadWriteMode::readOnly, startValueBD));
    const services::Buffer<algorithmFPType> startValueBuff = startValueBD.getBuffer();
    ctx.copy(workValueBuff, 0, startValueBuff, 0, argumentSize, &status);

    DAAL_CHECK_STATUS(status, inputArgument->releaseBlockOfRows(startValueBD));

    BlockDescriptor<int> predefinedBatchIndicesBD;
    DAAL_CHECK_STATUS(status, batchIndices->getBlockOfRows(0, nIter, ReadWriteMode::readOnly, learningRateBD));
    iterative_solver::internal::RngTask<int, sse2> rngTask(predefinedBatchIndicesBD.getBlockPtr(), batchSize);
    rngTask.init(nTerms, engine);

    algorithmFPType learningRate = learningRateArray[0];
    algorithmFPType consCoeff    = consCoeffsArray[0];

    UniversalBuffer gradientU                      = ctx.allocate(idType, argumentSize, &status);
    services::Buffer<algorithmFPType> gradientBuff = gradientU.get<algorithmFPType>();

    auto gradientSNT = SyclHomogenNumericTable<algorithmFPType>::create(gradientBuff, 1, argumentSize);
    function->getResult()->set(objective_function::gradientIdx, gradientSNT);

    *nProceededIterations = static_cast<int>(nIter);

    services::internal::HostAppHelper host(pHost, 10);
    for (size_t epoch = startIteration; epoch < (startIteration + nIter); epoch++)
    {
        if (epoch % L == 0 || epoch == startIteration)
        {
            learningRate = learningRateArray[(epoch / L) % learningRateLength];
            consCoeff    = consCoeffsArray[(epoch / L) % consCoeffsLength];
            if (indicesStatus == user || indicesStatus == random)
            {
                DAAL_ITTNOTIFY_SCOPED_TASK(generateUniform);
                const int * pValues = nullptr;
                DAAL_CHECK_STATUS(status, rngTask.get(pValues));
                ntBatchIndices->setArray(const_cast<int *>(pValues), ntBatchIndices->getNumberOfRows());
            }
        }

        DAAL_CHECK_STATUS(status, function->computeNoThrow());

        if (host.isCancelled(status, 1))
        {
            *nProceededIterations = static_cast<int>(epoch - startIteration);
            break;
        }

        if (epoch % L == 0)
        {
            if (nIter > 1 && accuracyThreshold > 0)
            {
                algorithmFPType pointNorm = algorithmFPType(0), gradientNorm = algorithmFPType(0);
                vectorNorm(workValueBuff, argumentSize, pointNorm);
                vectorNorm(gradientBuff, argumentSize, gradientNorm);
                const double gradientThreshold = accuracyThreshold * max(algorithmFPType(1), pointNorm);

                if (gradientNorm < gradientThreshold)
                {
                    *nProceededIterations = static_cast<int>(epoch - startIteration);
                    break;
                }
            }

            ctx.copy(prevWorkValueBuff, 0, workValueBuff, 0, argumentSize, &status);
        }
        DAAL_CHECK_STATUS(status, makeStep(argumentSize, prevWorkValueBuff, gradientBuff, workValueBuff, learningRate, consCoeff));
        nProceededIters++;
    }

    if (lastIterationResult)
    {
        BlockDescriptor<int> lastIterationResultBD;
        DAAL_CHECK_STATUS(status, lastIterationResult->getBlockOfRows(0, 1, ReadWriteMode::writeOnly, lastIterationResultBD));
        int * lastIterationResultArray = lastIterationResultBD.getBlockPtr();
        lastIterationResultArray[0]    = startIteration + nProceededIters;
        DAAL_CHECK_STATUS(status, lastIterationResult->releaseBlockOfRows(lastIterationResultBD));
    }

    if (pastWorkValueResult)
    {
        BlockDescriptor<algorithmFPType> pastWorkValueResultBD;
        DAAL_CHECK_STATUS(status, pastWorkValueResult->getBlockOfRows(0, argumentSize, ReadWriteMode::writeOnly, pastWorkValueResultBD));

        services::Buffer<algorithmFPType> pastWorkValueResultBuffer = pastWorkValueResultBD.getBuffer();

        ctx.copy(pastWorkValueResultBuffer, 0, prevWorkValueBuff, 0, argumentSize, &status);
        DAAL_CHECK_STATUS(status, pastWorkValueResult->releaseBlockOfRows(pastWorkValueResultBD));
    }

    DAAL_CHECK_STATUS(status, batchIndices->releaseBlockOfRows(predefinedBatchIndicesBD));
    DAAL_CHECK_STATUS(status, learningRateSequence->releaseBlockOfRows(learningRateBD));
    DAAL_CHECK_STATUS(status, conservativeSequence->releaseBlockOfRows(consCoeffsBD));
    DAAL_CHECK_STATUS(status, minimum->releaseBlockOfRows(workValueBD));
    DAAL_CHECK_STATUS(status, nIterations->releaseBlockOfRows(nIterationsBD));

    function->sumOfFunctionsParameter->batchIndices = previousBatchIndices;
    function->sumOfFunctionsInput->set(sum_of_functions::argument, previousArgument);
    return status;
}

} // namespace internal
} // namespace sgd
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
