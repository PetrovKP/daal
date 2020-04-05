/* file: svm_two_class_dense_batch.cpp */
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
!  Content:
!    C++ example of two-class support vector machine (SVM) classification
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVM_TWO_CLASS_DENSE_BATCH"></a>
 * \example svm_two_class_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
string trainDatasetFileName = "../data/batch/svm_two_class_train_dense.csv";

string testDatasetFileName = "../data/batch/svm_two_class_test_dense.csv";

const size_t nFeatures = 20;

/* Parameters for the SVM kernel function */
kernel_function::KernelIfacePtr kernel(new kernel_function::linear::Batch<>());

// kernel_function::KernelIfacePtr kernel(new kernel_function::rbf::Batch<>(gamma));

/* Model object for the SVM algorithm */
svm::training::ResultPtr trainingResult;
classifier::prediction::ResultPtr predictionResult;
NumericTablePtr testGroundTruth;

void trainModel();
void testModel();
void printResults();

int main(int argc, char * argv[])
{
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    trainModel();

    testModel();

    printResults();

    return 0;
}

void trainModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and labels */
    NumericTablePtr trainData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    NumericTablePtr trainGroundTruth(new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(trainData, trainGroundTruth));

    /* Retrieve the data from the input file */
    trainDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to train the SVM model */
    svm::training::Batch<> algorithm;

    algorithm.parameter.kernel    = kernel;
    algorithm.parameter.cacheSize = 40000000;

    algorithm.parameter.accuracyThreshold = 0.1;
    algorithm.parameter.tau               = 1.0e-6;
    algorithm.parameter.maxIterations     = 10000000;
    algorithm.parameter.doShrinking       = false;

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruth);

    /* Build the SVM model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();

    auto model                   = trainingResult->get(classifier::training::model);
    NumericTablePtr svCoeffTable = model->getClassificationCoefficients();
    NumericTablePtr svIndices    = model->getSupportIndices();
    NumericTablePtr sv           = model->getSupportVectors();
    const size_t nSV             = svCoeffTable->getNumberOfRows();

    const float bias(model->getBias());

    printf("nSV %lu\n", nSV);
    printf("bias %lf\n", bias);
    // printNumeric<float>(svCoeffTable, "", "svCoeffTable", 25);
    // printNumeric<int>(svCoeffTable, "", "svCoeffTable", 25);
    printNumeric<int>(svIndices, "", "svIndices", 25);
    // printNumeric<float>(sv, "", "sv", 25);
}

void testModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and labels */
    NumericTablePtr testData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    testGroundTruth = NumericTablePtr(new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(testData, testGroundTruth));

    /* Retrieve the data from input file */
    testDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to predict SVM values */
    svm::prediction::Batch<> algorithm;

    algorithm.parameter.kernel = kernel;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model, trainingResult->get(classifier::training::model));

    /* Predict SVM values */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults()
{
    printNumericTables<int, float>(testGroundTruth, predictionResult->get(classifier::prediction::prediction), "Ground truth\t",
                                   "Classification results", "SVM classification results (first 20 observations):", 20);
}
