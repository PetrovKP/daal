
#include "daal_sycl.h"
#include "service.h"
#include "service_sycl.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

string trainDatasetFileName = "/nfs/inn/proj/numerics1/Users/kpetrov/ats/svm/svm_repo_impl/data.csv";

const size_t nFeatures = 123;

void trainModel();

kernel_function::KernelIfacePtr kernel(new kernel_function::linear::Batch<>());

int main(int argc, char * argv[])
{
    checkArguments(argc, argv, 1, &trainDatasetFileName);

    for (const auto & deviceSelector : getListOfDevices())
    {
        const auto & nameDevice = deviceSelector.first;
        const auto & device     = deviceSelector.second;
        if (!device.is_gpu()) continue;
        cl::sycl::queue queue(device);
        std::cout << "Running on " << nameDevice << "\n\n";

        daal::services::SyclExecutionContext ctx(queue);
        services::Environment::getInstance()->setDefaultExecutionContext(ctx);

        trainModel();
        // trainModel(trainingResult);
    }

    return 0;
}

void trainModel()
{
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

    auto trainData        = SyclHomogenNumericTable<>::create(nFeatures, 0, NumericTable::doNotAllocate);
    auto trainGroundTruth = SyclHomogenNumericTable<>::create(1, 0, NumericTable::doNotAllocate);

    NumericTablePtr mergedData(new MergedNumericTable(trainGroundTruth, trainData));

    trainDataSource.loadDataBlock(mergedData.get());

    svm::training::Batch<> algorithm;

    algorithm.parameter.kernel        = kernel;
    algorithm.parameter.cacheSize     = 40000000;
    algorithm.parameter.C             = 1.0;
    algorithm.parameter.maxIterations = 10;

    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruth);

    /* Build the SVM model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    // trainingResult = algorithm.getResult();
}
