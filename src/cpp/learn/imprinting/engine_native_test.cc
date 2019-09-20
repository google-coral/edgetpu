#include "src/cpp/learn/imprinting/engine_native.h"

#include <cmath>

#include "absl/flags/parse.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/cpp/learn/imprinting/imprinting_test_base.h"
#include "src/cpp/test_utils.h"

namespace coral {
namespace learn {
namespace imprinting {

class ImprintingEngineNativeTest : public ImprintingTestBase {
 protected:
  struct TrainingDatapoint {
    TrainingDatapoint(const std::vector<Image>& images_,
                      const int groundtruth_class_id_)
        : groundtruth_class_id(groundtruth_class_id_) {
      image_number = images_.size();
      CHECK_GT(image_number, 0);
      image_size = images_[0].size();
      images.resize(image_number * image_size);
      int offset = 0;
      for (int i = 0; i < image_number; ++i) {
        CHECK_EQ(images_[i].size(), image_size);
        std::copy(images_[i].begin(), images_[i].end(), images.data() + offset);
        offset += image_size;
      }
    }
    std::vector<uint8_t> images;
    int image_number;
    int image_size;
    const int groundtruth_class_id;
  };

  EdgeTpuApiStatus CreateImprintingEngineNative(const std::string& model_path,
                                                bool keep_classes) {
    ImprintingEngineNativeBuilder builder(model_path, keep_classes);
    return builder(&imprinting_engine_native_);
  }

  EdgeTpuApiStatus OnlineTrain(
      const std::vector<TrainingDatapoint>& training_datapoints,
      const std::string& output_file_path) {
    for (const auto& training_datapoint : training_datapoints) {
      auto status = imprinting_engine_native_->Train(
          training_datapoint.images.data(), training_datapoint.image_number,
          training_datapoint.image_size,
          training_datapoint.groundtruth_class_id);
      if (status == kEdgeTpuApiError) {
        return kEdgeTpuApiError;
      }
    }
    return imprinting_engine_native_->SaveModel(output_file_path);
  }

  std::unique_ptr<ImprintingEngineNative> imprinting_engine_native_;
};

TEST_P(ImprintingEngineNativeTest, TestInitializationCheck) {
  const std::string expected_error_message =
      "ImprintingEngineNative must be initialized! Please ensure the instance "
      "is created by ImprintingEngineNativeBuilder!";

  ImprintingEngineNative engine;
  const std::string& output_file_path =
      GenerateOutputModelPath("test_initialization");

  TrainingDatapoint training_datapoint({cat_train_0_}, 0);
  EXPECT_EQ(kEdgeTpuApiError,
            engine.Train(training_datapoint.images.data(),
                         training_datapoint.image_number,
                         training_datapoint.image_size,
                         training_datapoint.groundtruth_class_id));

  EXPECT_EQ(expected_error_message, engine.get_error_message());

  EXPECT_EQ(kEdgeTpuApiError, engine.SaveModel(output_file_path));
  EXPECT_EQ(expected_error_message, engine.get_error_message());

  std::map<int, float> metadata;
  EXPECT_EQ(kEdgeTpuApiError, engine.get_metadata(&metadata));
  EXPECT_EQ(expected_error_message, engine.get_error_message());
  EXPECT_EQ(kEdgeTpuApiError, engine.set_metadata(metadata));
  EXPECT_EQ(expected_error_message, engine.get_error_message());
}

TEST_P(ImprintingEngineNativeTest, TestInvalidModelPath) {
  const std::string model_path = GenerateInputModelPath("invalid_model_path");
  const std::string expected_error_message =
      "Failed to open file: " + model_path;
  ImprintingEngineNativeBuilder builder(model_path, /*keep_classes=*/false);
  EXPECT_EQ(kEdgeTpuApiError, builder(&imprinting_engine_native_));
  EXPECT_EQ(expected_error_message, builder.get_error_message());
}

TEST_P(ImprintingEngineNativeTest, TestMetadataForOnlineTrainingNoTraining) {
  ASSERT_EQ(kEdgeTpuApiOk,
            CreateImprintingEngineNative(
                GenerateInputModelPath("mobilenet_v1_1.0_224_l2norm_quant"),
                /*keep_classes=*/true));
  std::map<int, float> metadata;
  EXPECT_EQ(kEdgeTpuApiOk, imprinting_engine_native_->get_metadata(&metadata));
  EXPECT_EQ(metadata.size(), 0);

  const std::map<int, float> metadata_expected = {{0, 1.2}, {1, 2.4}, {2, 5.6}};
  const std::string output_file_path =
      GenerateOutputModelPath("metadata_added");
  EXPECT_EQ(kEdgeTpuApiOk,
            imprinting_engine_native_->set_metadata(metadata_expected));
  imprinting_engine_native_->SaveModel(output_file_path);

  EXPECT_EQ(kEdgeTpuApiOk, CreateImprintingEngineNative(output_file_path,
                                                        /*keep_classes=*/true));
  EXPECT_EQ(kEdgeTpuApiOk, imprinting_engine_native_->get_metadata(&metadata));
  CheckMetadata(metadata_expected, metadata);
}

TEST_P(ImprintingEngineNativeTest, TestMetadataForOnlineTrainingWithTraining) {
  ASSERT_EQ(kEdgeTpuApiOk,
            CreateImprintingEngineNative(
                GenerateInputModelPath("mobilenet_v1_1.0_224_l2norm_quant"),
                /*keep_classes=*/false));
  std::map<int, float> metadata;
  EXPECT_EQ(kEdgeTpuApiOk, imprinting_engine_native_->get_metadata(&metadata));
  EXPECT_EQ(metadata.size(), 0);

  const std::string output_file_path =
      GenerateOutputModelPath("metadata_trained");
  const std::vector<TrainingDatapoint> training_datapoints = {
      TrainingDatapoint({cat_train_0_}, 0),
      TrainingDatapoint({hotdog_train_0_, hotdog_train_0_}, 1)};
  EXPECT_EQ(kEdgeTpuApiOk, OnlineTrain(training_datapoints, output_file_path));

  ASSERT_EQ(kEdgeTpuApiOk, CreateImprintingEngineNative(
                               output_file_path, /*keep_classes=*/false));
  EXPECT_EQ(kEdgeTpuApiOk, imprinting_engine_native_->get_metadata(&metadata));
  EXPECT_EQ(metadata.size(), 0);

  const std::map<int, float> metadata_expected = {{0, 1.}, {1, 2.}};
  ASSERT_EQ(kEdgeTpuApiOk, CreateImprintingEngineNative(output_file_path,
                                                        /*keep_classes=*/true));
  EXPECT_EQ(kEdgeTpuApiOk, imprinting_engine_native_->get_metadata(&metadata));
  CheckMetadata(metadata_expected, metadata);
}

TEST_P(ImprintingEngineNativeTest, TestModelWithoutL2NormLayer) {
  ImprintingEngineNativeBuilder builder(
      TestDataPath(tpu_tflite_ ? "mobilenet_v1_1.0_224_quant_edgetpu.tflite"
                               : "mobilenet_v1_1.0_224_quant.tflite"),
      /*keep_classes=*/false);
  EXPECT_EQ(kEdgeTpuApiError, builder(&imprinting_engine_native_));
  EXPECT_EQ(
      "Unsupported model architecture. Input model must have an L2Norm layer.",
      builder.get_error_message());
}

TEST_P(ImprintingEngineNativeTest, TestModelWithoutTraining) {
  ASSERT_EQ(kEdgeTpuApiOk,
            CreateImprintingEngineNative(
                GenerateInputModelPath("mobilenet_v1_1.0_224_l2norm_quant"),
                /*keep_classes=*/false));
  const std::string output_file_path =
      GenerateOutputModelPath("model_without_training");
  // Remove output model generated by previous tests.
  std::remove(output_file_path.c_str());

  EXPECT_EQ(kEdgeTpuApiError,
            imprinting_engine_native_->SaveModel(output_file_path));
  EXPECT_EQ("Model without training won't be saved!",
            imprinting_engine_native_->get_error_message());
}

TEST_P(ImprintingEngineNativeTest, TestModelNotKeepWithoutTraining) {
  ASSERT_EQ(kEdgeTpuApiOk,
            CreateImprintingEngineNative(
                GenerateInputModelPath("mobilenet_v1_1.0_224_l2norm_quant"),
                /*keep_classes=*/false));
  const std::string output_file_path =
      GenerateOutputModelPath("model_notkeep_without_training");
  EXPECT_EQ(kEdgeTpuApiError,
            imprinting_engine_native_->SaveModel(output_file_path));
  EXPECT_EQ("Model without training won't be saved!",
            imprinting_engine_native_->get_error_message());
}

TEST_P(ImprintingEngineNativeTest, TestWithWrongPath) {
  ImprintingEngineNativeBuilder builder(
      tpu_tflite_ ? "wrong_path_edgetpu.tflite" : "wrong_path.tflite",
      /*keep_classes=*/false);
  EXPECT_EQ(kEdgeTpuApiError, builder(&imprinting_engine_native_));
  EXPECT_EQ(tpu_tflite_ ? "Failed to open file: wrong_path_edgetpu.tflite"
                        : "Failed to open file: wrong_path.tflite",
            builder.get_error_message());
}

TEST_P(ImprintingEngineNativeTest, TestOnlineTrainingIndexTooLarge) {
  EXPECT_EQ(kEdgeTpuApiOk,
            CreateImprintingEngineNative(
                GenerateInputModelPath("mobilenet_v1_1.0_224_l2norm_quant"),
                /*keep_classes=*/true));
  const std::vector<TrainingDatapoint> training_datapoints = {
      TrainingDatapoint({cat_train_0_}, 1002)};
  const std::string output_file_path =
      GenerateOutputModelPath("online_training_large_index");
  EXPECT_EQ(kEdgeTpuApiError,
            OnlineTrain(training_datapoints, output_file_path));
  EXPECT_EQ("The class index of a new category is too large!",
            imprinting_engine_native_->get_error_message());
}

TEST_P(ImprintingEngineNativeTest, TestOnlineTrainingChangeBaseModelClasses) {
  EXPECT_EQ(kEdgeTpuApiOk,
            CreateImprintingEngineNative(
                GenerateInputModelPath("mobilenet_v1_1.0_224_l2norm_quant"),
                /*keep_classes=*/true));
  const std::vector<TrainingDatapoint> training_datapoints = {
      TrainingDatapoint({cat_train_0_}, 100)};
  const std::string output_file_path =
      GenerateOutputModelPath("online_training_change_base_model_classes");
  EXPECT_EQ(kEdgeTpuApiError,
            OnlineTrain(training_datapoints, output_file_path));
  EXPECT_EQ(
      "Cannot change the base model classes not trained with imprinting "
      "method!",
      imprinting_engine_native_->get_error_message());
}

INSTANTIATE_TEST_CASE_P(ImprintingEngineNativeTest, ImprintingEngineNativeTest,
                        ::testing::Values(false, true));

}  // namespace imprinting
}  // namespace learn
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
