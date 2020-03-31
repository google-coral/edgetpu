/*
A demo for on-device backprop (transfer learning) of a classification model.

This demo runs a similar task as described in TF Poets tutorial, except that
learning happens on-device.
https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0

Here are the steps to prepare the experiment data set:
  1) mkdir -p /tmp/retrain/
  2) curl http://download.tensorflow.org/example_images/flower_photos.tgz \
       | tar xz -C /tmp/retrain
  3) mogrify -format bmp /tmp/retrain/flower_photos//*//*.jpg

For more information, see
https://coral.ai/docs/edgetpu/retrain-classification-ondevice-backprop/
*/

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>

#include <random>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "glog/logging.h"
#include "src/cpp/basic/basic_engine.h"
#include "src/cpp/classification/engine.h"
#include "src/cpp/learn/backprop/softmax_regression_model.h"
#include "src/cpp/test_utils.h"

ABSL_FLAG(std::string, embedding_extractor_path,
          "/tmp/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite",
          "Path to the embedding extractor tflite model.");
ABSL_FLAG(std::string, data_dir, "/tmp/retrain/flower_photos",
          "Path to the data set.");
ABSL_FLAG(std::string, output_model_path, "/tmp/trained_model_edgetpu.tflite",
          "Path to the output tflite model.");

namespace coral {
namespace learn {
namespace backprop {
namespace {

// Pair of class label and image path.
using LabelAndPath = std::pair<int, std::string>;

constexpr double kValidationDataRatio = 0.1;
constexpr double kTestDataRatio = 0.1;
constexpr int kNumTrainingIterations = 500;
constexpr int kBatchSize = 100;
constexpr int kPrintEvery = 100;

// Returns list of bmp file paths under the given folder.
std::vector<std::string> ListBmpFilesUnderDir(const std::string& parent_dir) {
  std::vector<std::string> file_list;
  file_list.reserve(4096);  // reserve enough elements for the flower dataset.
  DIR* dir = opendir(parent_dir.c_str());
  if (dir) {
    struct dirent* file = nullptr;
    while ((file = readdir(dir)) != nullptr) {
      if (strstr(file->d_name, ".bmp"))
        file_list.push_back(parent_dir + "/" + file->d_name);
    }
    closedir(dir);
  }
  LOG(INFO) << file_list.size() << " bmp files found in folder " << parent_dir;
  return file_list;
}

// Returns number of classes.
int ListFilesInSubdirs(const std::string& grandparent_dir,
                       std::vector<LabelAndPath>* label_and_paths) {
  CHECK(label_and_paths);
  DIR* dir = opendir(grandparent_dir.c_str());
  int label = 0;
  if (dir) {
    struct dirent* subdir = nullptr;
    while ((subdir = readdir(dir)) != nullptr) {
      if (subdir->d_type != DT_DIR) continue;
      if (!strcmp(subdir->d_name, ".")) continue;
      if (!strcmp(subdir->d_name, "..")) continue;
      // Read samples of a new class.
      LOG(INFO) << "Read samples from subfolder " << subdir->d_name
                << " for class label " << label;
      const auto files =
          ListBmpFilesUnderDir(grandparent_dir + "/" + subdir->d_name);
      for (const auto& p : files) {
        label_and_paths->push_back(std::make_pair(label, p));
      }
      ++label;
    }
    closedir(dir);
  }
  return label;
}

// Returns number of classes.
int SplitDataset(const std::string& data_dir,
                 std::vector<LabelAndPath>* training_label_and_paths,
                 std::vector<LabelAndPath>* validation_label_and_paths,
                 std::vector<LabelAndPath>* test_label_and_paths) {
  CHECK(training_label_and_paths);
  CHECK(validation_label_and_paths);
  CHECK(test_label_and_paths);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);
  std::vector<LabelAndPath> all_label_and_paths;
  const int num_classes = ListFilesInSubdirs(data_dir, &all_label_and_paths);
  for (const auto& p : all_label_and_paths) {
    const double r = dis(gen);
    if (r < kValidationDataRatio) {
      validation_label_and_paths->push_back(p);
    } else if (r < kValidationDataRatio + kTestDataRatio) {
      test_label_and_paths->push_back(p);
    } else {
      training_label_and_paths->push_back(p);
    }
  }
  LOG(INFO) << "Number of training samples: "
            << training_label_and_paths->size();
  LOG(INFO) << "Number of validation samples: "
            << validation_label_and_paths->size();
  LOG(INFO) << "Number of test samples: " << test_label_and_paths->size();
  return num_classes;
}

void ExtractEmbedding(BasicEngine* embedding_extractor,
                      const std::vector<LabelAndPath>& label_and_paths,
                      int num_classes, int feature_dim, Tensor* embeddings,
                      std::vector<int>* labels) {
  CHECK(embedding_extractor);
  CHECK(embeddings);
  CHECK(labels);

  embeddings->resize(label_and_paths.size(), feature_dim);
  labels->resize(label_and_paths.size(), -1);

  const auto& input_shape = embedding_extractor->get_input_tensor_shape();
  for (int r = 0; r < label_and_paths.size(); ++r) {
    const int label = label_and_paths[r].first;
    CHECK_LT(label, num_classes);
    const auto& image_path = label_and_paths[r].second;

    // Extract embedding vector.
    const std::vector<uint8_t> input_tensor = GetInputFromImage(
        image_path, {input_shape[1], input_shape[2], input_shape[3]});
    std::vector<std::vector<float>> results =
        embedding_extractor->RunInference(input_tensor);
    CHECK_EQ(results.size(), 1);
    const auto& embedding = results[0];
    CHECK_EQ(embedding.size(), feature_dim);
    embeddings->row(r) =
        Eigen::Map<const Eigen::VectorXf>(embedding.data(), embedding.size());

    // Create one-hot label vector.
    (*labels)[r] = label;
  }
}

// Returns classification accuracy.
float EvaluateTrainedModel(const std::string& model_path,
                           const std::vector<LabelAndPath>& label_and_paths) {
  // Load the model.
  coral::ClassificationEngine engine(model_path);
  std::vector<int> input_tensor_shape = engine.get_input_tensor_shape();
  int num_hits = 0;
  for (int r = 0; r < label_and_paths.size(); ++r) {
    // Read the image.
    const std::vector<uint8_t> input_tensor = coral::GetInputFromImage(
        label_and_paths[r].second,
        {input_tensor_shape[1], input_tensor_shape[2], input_tensor_shape[3]});
    // Classify without thresholding.
    auto results = engine.ClassifyWithInputTensor(
        input_tensor, /*threshold=*/-std::numeric_limits<float>::infinity(),
        /*top_k=*/1);
    CHECK_EQ(results.size(), 1);
    if (results[0].id == label_and_paths[r].first) ++num_hits;
  }
  return static_cast<float>(num_hits) / label_and_paths.size();
}

void TrainAndEvaluate(const std::string& embedding_extractor_path,
                      const std::string& data_dir,
                      const std::string& output_model_path) {
  BasicEngine embedding_extractor(embedding_extractor_path);
  const auto& output_sizes = embedding_extractor.get_all_output_tensors_sizes();
  CHECK_EQ(output_sizes.size(), 1);
  const int feature_dim = output_sizes[0];

  const auto& t0 = std::chrono::steady_clock::now();
  std::vector<LabelAndPath> training_label_and_paths,
      validation_label_and_paths, test_label_and_paths;
  const int num_classes =
      SplitDataset(data_dir, &training_label_and_paths,
                   &validation_label_and_paths, &test_label_and_paths);

  TrainingData training_data;
  ExtractEmbedding(&embedding_extractor, training_label_and_paths, num_classes,
                   feature_dim, &training_data.training_data,
                   &training_data.training_labels);
  ExtractEmbedding(&embedding_extractor, validation_label_and_paths,
                   num_classes, feature_dim, &training_data.validation_data,
                   &training_data.validation_labels);

  // Run training
  const auto& t1 = std::chrono::steady_clock::now();
  SoftmaxRegressionModel model(feature_dim, num_classes);
  model.Initialize();
  TrainConfig train_config = {kNumTrainingIterations, kBatchSize, kPrintEvery};
  SgdUpdater sgd_updater;
  model.Train(training_data, train_config, &sgd_updater);

  const auto& t2 = std::chrono::steady_clock::now();
  LOG(INFO)
      << "Time to get embedding vectors (ms): "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  LOG(INFO)
      << "Time to train last layer (ms): "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  // Append learned weights to input model and save as tflite format.
  model.SaveAsTfliteModel(embedding_extractor_path, output_model_path);

  // Evaluate the trained model.
  const float training_accuracy =
      EvaluateTrainedModel(output_model_path, training_label_and_paths);
  LOG(INFO) << "Accuracy on training data: " << training_accuracy;
  const float validation_accuracy =
      EvaluateTrainedModel(output_model_path, validation_label_and_paths);
  LOG(INFO) << "Accuracy on validation data: " << validation_accuracy;
  const float test_accuracy =
      EvaluateTrainedModel(output_model_path, test_label_and_paths);
  LOG(INFO) << "Accuracy on test data: " << test_accuracy;
}

}  // namespace
}  // namespace backprop
}  // namespace learn
}  // namespace coral

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  coral::learn::backprop::TrainAndEvaluate(
      absl::GetFlag(FLAGS_embedding_extractor_path),
      absl::GetFlag(FLAGS_data_dir), absl::GetFlag(FLAGS_output_model_path));
}
