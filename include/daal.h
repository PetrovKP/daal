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

#ifndef __DAAL_H__
#define __DAAL_H__

#include "services/daal_defines.h"
#include "services/daal_memory.h"
#include "services/base.h"
#include "services/env_detect.h"
#include "services/library_version_info.h"
#include "data_management/compression/bzip2compression.h"
#include "data_management/compression/compression.h"
#include "data_management/compression/compression_stream.h"
#include "data_management/compression/lzocompression.h"
#include "data_management/compression/rlecompression.h"
#include "data_management/compression/zlibcompression.h"
#include "data_management/features/compatibility.h"
#include "data_management/data_source/csv_feature_manager.h"
#include "data_management/data_source/data_source.h"
#include "data_management/data_source/data_source_utils.h"
#include "data_management/data_source/file_data_source.h"
#include "data_management/data_source/string_data_source.h"
#include "data_management/data/aos_numeric_table.h"
#include "data_management/data/csr_numeric_table.h"
#include "data_management/data/data_archive.h"
#include "data_management/data/memory_block.h"
#include "services/collection.h"
#include "data_management/data/data_block.h"
#include "data_management/data/factory.h"
#include "data_management/data/data_serialize.h"
#include "services/daal_shared_ptr.h"
#include "data_management/data/data_collection.h"
#include "data_management/data/input_collection.h"
#include "data_management/data/data_dictionary.h"
#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/merged_numeric_table.h"
#include "data_management/data/row_merged_numeric_table.h"
#include "data_management/data/matrix.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/soa_numeric_table.h"
#include "data_management/data/symmetric_matrix.h"
#include "algorithms/classifier/classifier_training_types.h"
#include "algorithms/classifier/classifier_training_batch.h"
#include "algorithms/classifier/classifier_training_online.h"
#include "algorithms/classifier/classifier_predict_types.h"
#include "algorithms/classifier/classifier_predict.h"
#include "algorithms/classifier/classifier_model.h"
#include "algorithms/kernel_function/kernel_function_types.h"
#include "algorithms/kernel_function/kernel_function_types_linear.h"
#include "algorithms/kernel_function/kernel_function_types_rbf.h"
#include "algorithms/kernel_function/kernel_function.h"
#include "algorithms/kernel_function/kernel_function_linear.h"
#include "algorithms/kernel_function/kernel_function_rbf.h"
#include "algorithms/svm/svm_model.h"
#include "algorithms/svm/svm_model_builder.h"
#include "algorithms/svm/svm_train_types.h"
#include "algorithms/svm/svm_train.h"
#include "algorithms/svm/svm_predict_types.h"
#include "algorithms/svm/svm_predict.h"
#include "algorithms/svm/svm_quality_metric_set_batch.h"
#include "algorithms/svm/svm_quality_metric_set_types.h"
#include "algorithms/multi_class_classifier/multi_class_classifier_model.h"
#include "algorithms/multi_class_classifier/multi_class_classifier_model_builder.h"
#include "algorithms/multi_class_classifier/multi_class_classifier_train.h"
#include "algorithms/multi_class_classifier/multi_class_classifier_predict.h"
#include "algorithms/multi_class_classifier/multi_class_classifier_quality_metric_set_batch.h"
#include "algorithms/multi_class_classifier/multi_class_classifier_quality_metric_set_types.h"
#include "algorithms/engines/engine.h"
#include "algorithms/engines/mt19937/mt19937.h"
#include "algorithms/engines/mt19937/mt19937_types.h"
#include "algorithms/engines/mcg59/mcg59.h"
#include "algorithms/engines/mcg59/mcg59_types.h"
#include "algorithms/engines/engine_family.h"
#include "algorithms/engines/mt2203/mt2203.h"
#include "algorithms/engines/mt2203/mt2203_types.h"

#endif /* #ifndef __DAAL_H__ */
