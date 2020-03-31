/* file: cross_entropy_loss_dense_default_batch_kernel.h */
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

//++
//  Declaration of template function that calculate cross_entropy_loss.
//--

#ifndef __SVM_TRAIN_ONEAPI_H__
#define __SVM_TRAIN_ONEAPI_H__

#include "services/env_detect.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/svm/svm_train_types.h"
#include "algorithms/kernel/kernel.h"

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
using namespace daal::services;

template <typename algorithmFPType, typename ParameterType, Method method>
class SVMTrainOneAPI : public Kernel
{
public:
    services::Status compute(const NumericTablePtr & xTable, NumericTable & yTable, daal::algorithms::Model * r, const ParameterType * par);
};

template <typename algorithmFPType, typename ParameterType>
class SVMTrainOneAPI<algorithmFPType, ParameterType, boser> : public Kernel
{
    using Helper = HelperSVM<algorithmFPType>;
public:
    services::Status compute(const NumericTablePtr & xTable, NumericTable & yTable, daal::algorithms::Model * r, const ParameterType * par);

protected:
    // LocalSMO();

    services::Status initGrad(const services::Buffer<algorithmFPType> & y, services::Buffer<algorithmFPType> & f, const size_t n);

    // UpdateF();

    // // CalculateObjective();

    //   oneapi::internal::UniversalBuffer _uX;
    //   oneapi::internal::UniversalBuffer _uY;
    //   oneapi::internal::UniversalBuffer _fUniversal;
    //   oneapi::internal::UniversalBuffer _sigmoidUniversal;
    //   oneapi::internal::UniversalBuffer _subSigmoidYUniversal;
    bool verbose = false;
};

} // namespace internal

} // namespace training

} // namespace svm

} // namespace algorithms
} // namespace daal

#endif
