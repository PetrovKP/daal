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

    __kernel void initGradient(const __global algorithmFPType * const y, __global algorithmFPType * grad) {
        const int i = get_global_id(0);
        grad[i]     = -y[i];
    }

    __kernel void range(__global int * x) {
        const int i = get_global_id(0);
        x[i]        = i;
    }

    __kernel void checkUpper(const __global algorithmFPType * const y, const __global algorithmFPType * const alpha, const algorithmFPType C,
                             __global int * indicator) {
        const int i  = get_global_id(0);
        indicator[i] = y[i] > 0 && alpha[i] < C || y[i] < 0 && alpha[i] > 0;
    }

    __kernel void checkLower(const __global algorithmFPType * const y, const __global algorithmFPType * const alpha, const algorithmFPType C,
                             __global int * indicator) {
        const int i  = get_global_id(0);
        indicator[i] = y[i] > 0 && alpha[i] > 0 || y[i] < 0 && alpha[i] < C;
    }

    __kernel void copyBlockIndices(const __global algorithmFPType * const x, const __global int * const ind, const uint ldx, __global algorithmFPType * newX) {
        const uint index = get_global_id(1);
        const uint jCol  = get_global_id(0);

        const int iRow = ind[index];

        const __global algorithmFPType * const xi = &x[iRow * ldx];
        __global algorithmFPType * newXi          = &newX[index * ldx];

        newXi[jCol] = xi[jCol];
    }

);

#undef DECLARE_SOURCE_DAAL

#endif
