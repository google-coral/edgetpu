#include <algorithm>
#include <numeric>
#include <vector>

#include "absl/flags/parse.h"
#include "benchmark/benchmark.h"
#include "glog/logging.h"
#include "src/cpp/learn/backprop/sgd_updater.h"
#include "src/cpp/learn/backprop/softmax_regression_model.h"
#include "src/cpp/learn/backprop/test_utils.h"

namespace coral {
namespace learn {
namespace backprop {

using ::Eigen::MatrixXf;
using ::Eigen::VectorXf;

constexpr int kTotalNumTrainingSamples = 1024;
constexpr int kTotalNumValidationSamples = 256;
constexpr int kTotalNumSamples =
    kTotalNumTrainingSamples + kTotalNumValidationSamples;
constexpr int kNumTrainingEpochs = 500;
constexpr int kBatchSize = 100;

template <int NumClass, int FeatureDim>
static void BM_SoftmaxRegressionBackprop(benchmark::State& state) {
  // For latency benchmark purposes, the distribution of training data does not
  // matter. We just use random values.
  std::vector<VectorXf> means;
  std::vector<MatrixXf> cov_mats;
  means.reserve(NumClass);
  cov_mats.reserve(NumClass);
  for (int i = 0; i < NumClass; ++i) {
    means.push_back(VectorXf::Random(FeatureDim));
    cov_mats.push_back(MatrixXf::Random(FeatureDim, FeatureDim));
  }
  const std::vector<int> class_sizes(NumClass, kTotalNumSamples / NumClass);
  const auto training_data = generate_fake_data(class_sizes, means, cov_mats,
                                                kTotalNumTrainingSamples);
  SoftmaxRegressionModel model(FeatureDim, NumClass);
  model.Initialize();
  SgdUpdater sgd_updater(/*learning_rate=*/0.01);
  TrainConfig train_config = {kNumTrainingEpochs, kBatchSize,
                              /*print_every=*/-1};
  while (state.KeepRunning()) {
    model.Train(training_data, train_config, &sgd_updater);
  }
}
BENCHMARK_TEMPLATE(BM_SoftmaxRegressionBackprop, 4, 256);
BENCHMARK_TEMPLATE(BM_SoftmaxRegressionBackprop, 16, 256);
BENCHMARK_TEMPLATE(BM_SoftmaxRegressionBackprop, 4, 1024);
BENCHMARK_TEMPLATE(BM_SoftmaxRegressionBackprop, 16, 1024);

}  // namespace backprop
}  // namespace learn
}  // namespace coral

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
