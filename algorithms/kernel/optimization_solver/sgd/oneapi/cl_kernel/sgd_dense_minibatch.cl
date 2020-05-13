/* file: sgd_dense_minibatch.cl */
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
//  Implementation of SGD dense minibatch OpenCL kernels.
//--
*/

#ifndef __SGD_DENSE_MINIBATCH_KERNELS_CL__
#define __SGD_DENSE_MINIBATCH_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE_DAAL(name, src) static const char *(name) = #src;

DECLARE_SOURCE_DAAL(
    clKernelSGDMiniBatch,

    __kernel void makeStep(const __global algorithmFPType * const gradient, const __global algorithmFPType * const prevWorkValue,
                           __global algorithmFPType * workValue, const algorithmFPType learningRate, const algorithmFPType consCoeff) {
        const uint j = get_global_id(0);

        workValue[j] = workValue[j] - learningRate * (gradient[j] + consCoeff * (workValue[j] - prevWorkValue[j]));
    }

);

#undef DECLARE_SOURCE_DAAL

#endif
