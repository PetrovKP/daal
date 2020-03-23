/* file: cross_entropy_loss_dense_default.cl */
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
//  Implementation of Cross-Entropy Loss OpenCL kernels.
//--
*/

#ifndef __SVM_TRAIN_KERNELS_CL__
#define __SVM_TRAIN_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE_DAAL(name, src) static const char *(name) = #src;

DECLARE_SOURCE_DAAL(
    clKernelSVMTrain,

    void initGradient(const __global algorithmFPType * const y, __global algorithmFPType * grad) {
        const int i = get_global_id(0);
        grad[i] = -y[i];
    }



);

#undef DECLARE_SOURCE_DAAL

#endif
