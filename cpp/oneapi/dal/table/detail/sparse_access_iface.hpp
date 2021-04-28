/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#pragma once

#include "oneapi/dal/array.hpp"

namespace oneapi::dal::detail {
namespace v1 {

enum class csr_indexing { zero_based, one_based };

template <typename T>
struct sparse_block {
    array<T> data;
    array<std::int64_t> column_indices;
    array<std::int64_t> row_indices;
    csr_indexing indexing;

    sparse_block() : data(), column_indices(), row_indices(), indexing(csr_indexing::one_based) {}
};

#define PULL_SPARSE_BLOCK_SIGNATURE_HOST(T)                   \
    void pull_sparse_block(const default_host_policy& policy, \
                           sparse_block<T>& block,            \
                           const range& row_range)

#define DECLARE_PULL_SPARSE_BLOCK_HOST(T) virtual PULL_SPARSE_BLOCK_SIGNATURE_HOST(T) = 0;

#define DEFINE_TEMPLATE_PULL_SPARSE_BLOCK_HOST(Derived, T)                        \
    PULL_SPARSE_BLOCK_SIGNATURE_HOST(T) override {                                \
        static_cast<Derived*>(this)->pull_sparse_block(policy, block, row_range); \
    }

class pull_sparse_block_iface {
public:
    virtual ~pull_sparse_block_iface() = default;

    DECLARE_PULL_SPARSE_BLOCK_HOST(float)
    DECLARE_PULL_SPARSE_BLOCK_HOST(double)
    DECLARE_PULL_SPARSE_BLOCK_HOST(std::int32_t)
};

template <typename Derived>
class pull_sparse_block_template : public base, public pull_sparse_block_iface {
public:
    DEFINE_TEMPLATE_PULL_SPARSE_BLOCK_HOST(Derived, float)
    DEFINE_TEMPLATE_PULL_SPARSE_BLOCK_HOST(Derived, double)
    DEFINE_TEMPLATE_PULL_SPARSE_BLOCK_HOST(Derived, std::int32_t)
};

#undef PULL_SPARSE_BLOCK_SIGNATURE_HOST
#undef DECLARE_PULL_SPARSE_BLOCK_HOST
#undef DEFINE_TEMPLATE_PULL_SPARSE_BLOCK_HOST

template <typename Object>
inline std::shared_ptr<pull_sparse_block_iface> get_pull_sparse_block_iface(Object&& obj) {
    const auto pimpl = pimpl_accessor{}.get_pimpl(std::forward<Object>(obj));
    return std::shared_ptr<pull_sparse_block_iface>{ pimpl, pimpl->get_pull_sparse_block_iface() };
}

} // namespace v1

using v1::sparse_block;
using v1::csr_indexing;
using v1::pull_sparse_block_iface;
using v1::pull_sparse_block_template;
using v1::get_pull_sparse_block_iface;

} // namespace oneapi::dal::detail
