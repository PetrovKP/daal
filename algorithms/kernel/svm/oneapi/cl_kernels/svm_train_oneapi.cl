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

    inline bool inUpper(const algorithmFPType alpha, const algorithmFPType y, const algorithmFPType C) {
        // (0 < a && a < C) || (y == 1  && a == 0) || (y == -1 && a == C);
        return y > 0 && alpha < C || y < 0 && alpha > 0;
    }

    inline bool inLower(const algorithmFPType alpha, const algorithmFPType y, const algorithmFPType C) {
        // (0 < a && a < C) || (y == -1 && a == 0) || (y == 1 && a == C);
        return y > 0 && alpha > 0 || y < 0 && alpha < C;
    }

    __kernel void checkUpper(const __global algorithmFPType * const y, const __global algorithmFPType * const alpha, const algorithmFPType C,
                             __global int * indicator) {
        const int i  = get_global_id(0);
        indicator[i] = inUpper(y[i], alpha[i], C);
    }

    __kernel void checkLower(const __global algorithmFPType * const y, const __global algorithmFPType * const alpha, const algorithmFPType C,
                             __global int * indicator) {
        const int i  = get_global_id(0);
        indicator[i] = inLower(y[i], alpha[i], C);
    }

    __kernel void copyBlockIndices(const __global algorithmFPType * const x, const __global int * const ind, const uint ldx,
                                   __global algorithmFPType * newX) {
        const uint index = get_global_id(1);
        const uint jCol  = get_global_id(0);

        const int iRow = ind[index];

        const __global algorithmFPType * const xi = &x[iRow * ldx];
        __global algorithmFPType * newXi          = &newX[index * ldx];

        newXi[jCol] = xi[jCol];
    }

#define WS_SIZE 16

    __kernel void reduceMax(const __global algorithmFPType * values, __global int * indices) {
        const int group_size = get_local_size(0);
        const int local_id   = get_local_id(0);

        indices[local_id] = local_id;

        for (int stride = group_size / 2; stride > 0; stride >>= 1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);

            if (local_id < stride)
            {
                const algorithmFPType v  = values[local_id];
                const algorithmFPType vk = values[local_id + stride];
                if (vk >= v)
                {
                    indices[local_id] = indices[local_id + stride]
                }
            }
        }
    }

    __kernel void reduceMin(const __global algorithmFPType * values, __global int * indices) {
        const int group_size = get_local_size(0);
        const int local_id   = get_local_id(0);

        indices[local_id] = local_id;

        for (int stride = group_size / 2; stride > 0; stride >>= 1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);

            if (local_id < stride)
            {
                const algorithmFPType v  = values[local_id];
                const algorithmFPType vk = values[local_id + stride];
                if (vk <= v)
                {
                    indices[local_id] = indices[local_id + stride]
                }
            }
        }
    }

    algorithmFPType WSSi(const algorithmFPType gradi, const algorithmFPType alphai, const algorithmFPType yi, const algorithmFPType C, int & Bi) {
        const uint i = get_local_id(0);

        // TODO
        const algorithmFPType MIN_FLT = -1e20;

        Bi = -1;
        __local algorithmFPType objFunc[WS_SIZE];
        __local int indices[WS_SIZE];

        objFunc[i] = inUpper(alphai, yi, C) ? -yi * gradi : MIN_FLT;

        /* Find i index of the working set (Bi) */
        reduceMax(objFunc, indices);
        barrier(CLK_LOCAL_MEM_FENCE);
        Bi                         = indices[0];
        const algorithmFPType GMax = objFunc[Bi];

        return GMax;
    }

    algorithmFPType WSSj(const algorithmFPType gradi, const algorithmFPType alphai, const algorithmFPType yi, const algorithmFPType Kii,
                         const algorithmFPType KBiBi, const algorithmFPType KiBi, const algorithmFPType tau, const algorithmFPType GMax, int & Bj) {
        const uint i = get_local_id(0);

        Bj = -1;

        __local algorithmFPType objFunc[WS_SIZE];
        __local int indices[WS_SIZE];

        // TODO
        const algorithmFPType MAX_FLT = 1e20;

        const algorithmFPType zero = 0.0;
        const algorithmFPType two  = 2.0;

        const algorithmFPType ygrad = -yi * gradi;

        const algorithmFPType b = GMax - ygrad;
        const algorithmFPType a = max(Kii + KBiBi - two * KiBi, tau);

        const algorithmFPType dt = b / a;

        objFunc[i] = inLower(alphai, yi, C) && ygrad < GMax ? -b * dt : MAX_FLT;

        reduceMin(objFunc, indices);
        barrier(CLK_LOCAL_MEM_FENCE);
        Bj                         = indices[0];
        const algorithmFPType GMin = objFunc[Bj];

        return GMin;
    }

    __kernel void smoKernel(const __global algorithmFPType * const y, const __global algorithmFPType * const kernelWsRows,
                            const __global int * wsIndices, const uint ldx, const __global algorithmFPType * grad const algorithmFPType C,
                            const algorithmFPType tau, const int maxInnerIteration, __global algorithmFPType * alpha,
                            __global algorithmFPType * deltaalpha, __global algorithmFPType * resinfo) {
        const uint i = get_local_id(0);

        __local algorithmFPType kd[WS_SIZE];

        const int wsIndex = wsIndices[i];

        algorithmFPType gradi     = grad[wsIndex];
        algorithmFPType alphai    = alpha[wsIndex];
        algorithmFPType oldalphai = alphai;
        const algorithmFPType yi  = y[wsIndex];

        kd[i] = kernelWsRows[i * ldx + wsIndex];

        int iter = 0;
        for (; iter < maxInnerIteration; iter++)
        {
            int Bi, Bj;
            //  m(alpha) = max(-y[i]*grad[i]): i belongs to I_UP (alpha)
            const algorithmFPType ma = WSSi(gradi, alphai, yi, C, Bi);

            barrier(CLK_LOCAL_MEM_FENCE);

            const algorithmFPType Kii   = kd[i];
            const algorithmFPType KBiBi = kd[Bi];
            const algorithmFPType KiBi  = kernelWsRows[Bi * ldx + wsIndex];

            const algorithmFPType Ma = WSSj(gradi, alphai, yi, C, Kii, KBiBi, KiBi, tau, ma, Bj);

            barrier(CLK_LOCAL_MEM_FENCE);

            const algorithmFPType KiBj = kernelWsRows[Bj * ldx + wsIndex];

            // ma - Ma is used to check stopping condition
            const algorithmFPType curEps = ma - Ma;

            if (curEps < 10.0 * eps)
            {
                resinfo[1] = curEps;
                break;
            }
            // Update alpha

            deltaBi;
            deltaBj;
            if (i == Bi)
            {
                deltaBi = yi > 0 ? c - alphai : alphai;
            }
            if (i == Bj)
            {
                deltaBj                     = yi > 0 ? alphai : c - alphai;
                const algorithmFPType ygrad = -yi * gradi;
                const algorithmFPType b     = GMax - ygrad;
                const algorithmFPType a     = max(kd[i] + kd[Bi] - two * KiBj, tau);

                const algorithmFPType dt = b / a;
                deltaBj                  = min(deltaBj, dt);
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            const algorithmFPType delta = min(deltaBi, deltaBj);
            if (i == Bi)
            {
                alphai = alphai + yi * delta;
            }
            if (i == Bj)
            {
                alphai = alphai - yi * delta;
            }

            // Update gradient
            gradi = gradi + delta * (KiBi - KiBj);
        }
        alpha[wsIndex] = alphai;
        deltaalpha[i]  = (oldalphai - alphai) * yi;
        resinfo[0]     = iter;
    }

);

#undef DECLARE_SOURCE_DAAL

#endif
