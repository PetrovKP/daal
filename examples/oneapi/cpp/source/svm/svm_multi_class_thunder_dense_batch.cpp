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

#include "oneapi/dal/algo/svm.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;

int main(int argc, char const *argv[]) {
    const auto train_data_file_name = get_data_path("svm_multi_class_train_dense.csv");
    const auto train_label_file_name = get_data_path("svm_multi_class_train_dense_label.csv");
    // const auto test_data_file_name = get_data_path("svm_multi_class_test_dense.csv");
    // const auto test_label_file_name = get_data_path("svm_multi_class_train_dense_label.csv");

    const auto x_train = dal::read<dal::table>(dal::csv::data_source{ train_data_file_name });
    const auto y_train = dal::read<dal::table>(dal::csv::data_source{ train_label_file_name });

    const auto kernel_desc = dal::linear_kernel::descriptor{}.set_scale(1.0).set_shift(0.0);
    const auto svm_desc = dal::svm::descriptor{ kernel_desc }.set_classes_count(5).set_c(1.0);
    const auto result_train = dal::train(svm_desc, x_train, y_train);

    // std::cout << "Bias:\n" << result_train.get_bias() << std::endl;
    // std::cout << "Support indices:\n" << result_train.get_support_indices() << std::endl;

    // const auto x_test = dal::read<dal::table>(dal::csv::data_source{ test_data_file_name });
    // const auto y_true = dal::read<dal::table>(dal::csv::data_source{ test_label_file_name });

    // const auto result_test = dal::infer(svm_desc, result_train.get_model(), x_test);

    // std::cout << "Decision function result:\n" << result_test.get_decision_function() << std::endl;
    // std::cout << "Labels result:\n" << result_test.get_labels() << std::endl;
    // std::cout << "Labels true:\n" << y_true << std::endl;

    return 0;
}