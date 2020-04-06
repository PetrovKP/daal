/* file: cross_entropy_loss_dense_default_batch_kernel.h */
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

//++
//  Declaration of template function that calculate cross_entropy_loss.
//--

#ifndef __CROSS_ENTROPY_LOSS_DENSE_DEFAULT_BATCH_KERNEL_H__
#define __CROSS_ENTROPY_LOSS_DENSE_DEFAULT_BATCH_KERNEL_H__

#include "services/env_detect.h"
#include "data_management/data/numeric_table.h"

#include "algorithms/kernel/svm/oneapi/svm_helper.h"

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
using namespace daal::internal;
using namespace daal::services;

template <typename algorithmFPType, typename ParameterType, Method method>
class SVMTrainOneAPI : public Kernel
{
public:
    services::Status compute(const NumericTablePtr & xTable, NumericTable & yTable, daal::algorithms::Model * r, const ParameterType * par);
<<<<<<< HEAD:algorithms/kernel/svm/oneapi/svm_train_kernel.h
=======
};

template <typename algorithmFPType, typename ParameterType>
class SVMTrainOneAPI<algorithmFPType, ParameterType, boser> : public Kernel
{
    using Helper = HelperSVM<algorithmFPType>;

public:
    services::Status compute(const NumericTablePtr & xTable, NumericTable & yTable, daal::algorithms::Model * r, const ParameterType * par);
>>>>>>> 53c7b11f... fix bugs:algorithms/kernel/svm/oneapi/svm_train_oneapi_kernel.h

protected:
<<<<<<< HEAD:algorithms/kernel/svm/oneapi/svm_train_kernel.h
    // LocalSMO();

<<<<<<< HEAD:algorithms/kernel/svm/oneapi/svm_train_kernel.h
    size_t GetWSSize(size_t nSamples);

=======
>>>>>>> 14431dac... kernel support was added:algorithms/kernel/svm/oneapi/svm_train_oneapi_kernel.h
=======
>>>>>>> 64f30ec0... smo local add & update F:algorithms/kernel/svm/oneapi/svm_train_oneapi_kernel.h
    services::Status initGrad(const services::Buffer<algorithmFPType> & y, services::Buffer<algorithmFPType> & f, const size_t n);
    services::Status updateGrad(const services::Buffer<algorithmFPType> & kernelWS, const services::Buffer<algorithmFPType> & deltaalpha,
                                services::Buffer<algorithmFPType> & grad, const size_t nVectors, const size_t nWS);
    services::Status smoKernel(const services::Buffer<algorithmFPType> & y, const services::Buffer<algorithmFPType> & kernelWsRows,
                               const services::Buffer<int> & wsIndices, const int ldK, const services::Buffer<algorithmFPType> & f,
                               const algorithmFPType C, const algorithmFPType eps, const algorithmFPType tau, const int maxInnerIteration,
                               services::Buffer<algorithmFPType> & alpha, services::Buffer<algorithmFPType> & deltaalpha,
                               services::Buffer<algorithmFPType> & resinfo, const size_t nWS);

    bool checkStopCondition(const algorithmFPType diff, const algorithmFPType diffPrev, const algorithmFPType eps, int & sameLocalDiff);

    double calculateObjective(const services::Buffer<algorithmFPType> & y, const services::Buffer<algorithmFPType> & alpha,
                              const services::Buffer<algorithmFPType> & grad, const size_t nVectors);

    bool verbose = false;
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif