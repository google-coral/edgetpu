#include "src/cpp/learn/backprop/softmax_regression_model.h"

#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "Eigen/src/Core/Matrix.h"
#include "glog/logging.h"
#include "src/cpp/error_reporter.h"
#include "src/cpp/learn/backprop/cross_entropy_loss.h"
#include "src/cpp/learn/backprop/fully_connected.h"
#include "src/cpp/learn/backprop/multi_variate_normal_distribution.h"
#include "src/cpp/learn/backprop/operator.h"
#include "src/cpp/learn/backprop/sgd_updater.h"
#include "src/cpp/learn/backprop/softmax.h"
#include "src/cpp/learn/backprop/tensor.h"
#include "src/cpp/learn/utils.h"

namespace coral {
namespace learn {
namespace backprop {

void SoftmaxRegressionModel::Initialize() {
  // Randomly set weights for mat_w_ from a gaussian distribution.
  mat_w_ = Tensor::Ones(feature_dim_, num_classes_);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dist(0, 1);
  auto random = [&gen, &dist](float x) -> float { return dist(gen); };
  mat_w_ = weight_scale_ * mat_w_.unaryExpr(random);

  // Set weights for vec_b_ to zero.
  vec_b_ = Tensor::Zero(1, num_classes_);

  // Create forward and backward layers for model.
  forward_ops_.emplace_back(std::unique_ptr<Operator>(new FullyConnected()));
  forward_ops_.emplace_back(std::unique_ptr<Operator>(new Softmax()));
  forward_ops_.emplace_back(std::unique_ptr<Operator>(new CrossEntropyLoss()));
  backward_ops_.emplace_back(
      std::unique_ptr<Operator>(new FullyConnectedGradient()));
  backward_ops_.emplace_back(std::unique_ptr<Operator>(new SoftmaxGradient()));
  backward_ops_.emplace_back(
      std::unique_ptr<Operator>(new CrossEntropyGradient()));

  caches_ = std::vector<Tensor>(3);
  logit_min_ = std::numeric_limits<float>::infinity();
  logit_max_ = -std::numeric_limits<float>::infinity();

  VLOG(1) << "DONE INITIALIZING";
}

float SoftmaxRegressionModel::GetLoss(const Tensor& mat_x,
                                      const Tensor& labels) {
  // cache[0] is Tensor logits
  caches_[0] = forward_ops_[0]->Compute({mat_x, mat_w_, vec_b_})[0];
  logit_min_ = std::min(logit_min_, caches_[0].minCoeff());
  logit_max_ = std::max(logit_max_, caches_[0].maxCoeff());
  // cache[1] is Tensor probs
  caches_[1] = forward_ops_[1]->Compute({caches_[0]})[0];
  // cache[2] is Tensor loss
  caches_[2] = forward_ops_[2]->Compute({labels, caches_[1]})[0];
  // add regularization term
  VLOG(1) << "Adding regularization term to loss";
  return caches_[2](0, 0) +
         0.5 * reg_ * (mat_w_.transpose() * mat_w_).array().sum();
}

std::vector<Tensor> SoftmaxRegressionModel::GetGrads(const Tensor& mat_x,
                                                     const Tensor& labels) {
  Tensor dprobs = backward_ops_[2]->Compute({labels, caches_[1]})[0];
  Tensor dlogits = backward_ops_[1]->Compute({caches_[0], dprobs})[0];
  std::vector<Tensor> xwb_grads =
      backward_ops_[0]->Compute({mat_x, mat_w_, vec_b_, dlogits});
  return {xwb_grads[1], xwb_grads[2]};
}

// Helper function to compute the argmax for each row of input Tensor.
// Eigen does not have a built-in rowwise operation for argmax.
std::vector<int> rowwise_argmax(const Tensor& input) {
  std::vector<int> output(input.rows(), -1);
  Tensor::Index argmax;
  for (int i = 0; i < input.rows(); i++) {
    input.row(i).maxCoeff(&argmax);
    output[i] = argmax;
  }
  return output;
}

// Helper function to get a random set of b indices for the batch where b is the
// batch_size.
// The indices are in the range (0, n) where n+1 is number of rows possible.
std::vector<int> GetBatchIndices(const Tensor& tensor, int batch_size) {
  std::random_device random_device;
  std::mt19937 mersenne_engine{random_device()};
  std::uniform_int_distribution<int> dist{0,
                                          static_cast<int>(tensor.rows() - 1)};
  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
  std::vector<int> vec(batch_size);
  std::generate(std::begin(vec), std::end(vec), gen);
  return vec;
}

std::vector<int> SoftmaxRegressionModel::RunInference(const Tensor& mat_x) {
  Tensor scores = forward_ops_[0]->Compute({mat_x, mat_w_, vec_b_})[0];
  return rowwise_argmax(scores);
}

float SoftmaxRegressionModel::GetAccuracy(const Tensor& mat_x,
                                          const std::vector<int>& labels) {
  CHECK_EQ(labels.size(), mat_x.rows());
  const auto result = RunInference(mat_x);
  int correct = 0;
  for (int r = 0; r < result.size(); ++r) {
    if (result[r] == labels[r]) ++correct;
  }
  return static_cast<float>(correct) / labels.size();
}

void SoftmaxRegressionModel::Train(const TrainingData& data,
                                   const TrainConfig& train_config,
                                   SgdUpdater* sgd_updater) {
  // For each iteration in num_iter, use a random batch of inputs of size
  // batch_size to calculate gradients and learn model weights.
  for (int i = 0; i < train_config.num_iter; i++) {
    const auto& batch_indices =
        GetBatchIndices(data.training_data, train_config.batch_size);
    Tensor train_batch, labels_batch;
    train_batch = data.training_data(batch_indices, Eigen::all);

    // Create one-hot label vectors
    labels_batch = Tensor::Zero(train_config.batch_size, num_classes_);
    for (int r = 0; r < train_config.batch_size; ++r) {
      labels_batch(r, data.training_labels[batch_indices[r]]) = 1.0f;
    }
    float loss = GetLoss(train_batch, labels_batch);

    // Update model weights mat_w_ and vec_b.
    std::vector<Tensor> grads = GetGrads(train_batch, labels_batch);
    sgd_updater->Update(grads, {&mat_w_, &vec_b_});

    if (train_config.print_every > 0 && i % train_config.print_every == 0) {
      LOG(INFO) << "Loss: " << loss;
      LOG(INFO) << "Train Acc: "
                << GetAccuracy(data.training_data, data.training_labels);
      LOG(INFO) << "Valid Acc: "
                << GetAccuracy(data.validation_data, data.validation_labels);
    }
  }
}

void SoftmaxRegressionModel::SaveAsTfliteModel(
    const std::string& embedding_extractor_model_path,
    const std::string& output_model_path) const {
  LOG(INFO) << "Logit min: " << logit_min_ << ", max: " << logit_max_;
  coral::EdgeTpuErrorReporter reporter;
  CHECK_EQ(learn::AppendFullyConnectedAndSoftmaxLayerToModel(
               embedding_extractor_model_path, output_model_path, mat_w_.data(),
               mat_w_.size(), vec_b_.data(), vec_b_.size(), logit_min_,
               logit_max_, &reporter),
           kEdgeTpuApiOk)
      << reporter.message();
}

}  // namespace backprop
}  // namespace learn
}  // namespace coral
