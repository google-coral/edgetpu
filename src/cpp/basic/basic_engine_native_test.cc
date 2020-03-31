#include "src/cpp/basic/basic_engine_native.h"

#include <algorithm>
#include <chrono>  // NOLINT
#include <random>
#include <thread>  // NOLINT

#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "src/cpp/fake_op.h"
#include "src/cpp/test_utils.h"

namespace coral {
namespace {

void ExpectNotInitialized(BasicEngineNative *engine) {
  EXPECT_EQ(
      "BasicEngineNative must be initialized! Please ensure the instance is "
      "created by BasicEngineNativeBuilder!",
      engine->get_error_message());
}

TEST(BasicEngineNativeBuilderTest, TestNullptrOutput) {
  BasicEngineNativeBuilder builder(
      TestDataPath("mobilenet_v1_1.0_224_quant.tflite"));
  EXPECT_EQ(kEdgeTpuApiError, builder(nullptr));
}

TEST(BasicEngineNativeBuilderTest, TestInvalidModelPath) {
  BasicEngineNativeBuilder builder("invalid_path.tflite");
  std::unique_ptr<BasicEngineNative> engine;
  EXPECT_EQ(kEdgeTpuApiError, builder(&engine));
  EXPECT_EQ("Could not open 'invalid_path.tflite'.",
            builder.get_error_message());
}

TEST(BasicEngineNativeBuilderTest, TestInvalidDevicePath) {
  BasicEngineNativeBuilder builder(
      TestDataPath("mobilenet_v1_1.0_224_quant.tflite"),
      "/invalid/device/path");
  std::unique_ptr<BasicEngineNative> engine;
  EXPECT_EQ(kEdgeTpuApiError, builder(&engine));
  EXPECT_EQ("Path /invalid/device/path does not map to an Edge TPU device.",
            builder.get_error_message());
}

TEST(BasicEngineNativeBuilderTest, MultipleEngines) {
  BasicEngineNativeBuilder builder(
      TestDataPath("mobilenet_v1_1.0_224_quant.tflite"));
  std::unique_ptr<BasicEngineNative> engine_1;
  EXPECT_EQ(kEdgeTpuApiOk, builder(&engine_1)) << builder.get_error_message();
  std::unique_ptr<BasicEngineNative> engine_2;
  EXPECT_EQ(kEdgeTpuApiOk, builder(&engine_2)) << builder.get_error_message();
}

TEST(BasicEngineNativeBuilderTest, OneTimeBuilder) {
  std::unique_ptr<tflite::FlatBufferModel> model(
      tflite::FlatBufferModel::BuildFromFile(
          TestDataPath("mobilenet_v1_1.0_224_quant.tflite").c_str()));
  BasicEngineNativeBuilder builder(std::move(model));
  std::unique_ptr<BasicEngineNative> engine_1;
  EXPECT_EQ(kEdgeTpuApiOk, builder(&engine_1)) << builder.get_error_message();
  std::unique_ptr<BasicEngineNative> engine_2;
  EXPECT_EQ(kEdgeTpuApiError, builder(&engine_2));
  EXPECT_EQ("model_ is nullptr!", builder.get_error_message());
}

TEST(BasicEngineNativeTest, TestWithoutInitialization) {
  BasicEngineNative engine;
  float const *result;
  size_t result_size;
  std::vector<uint8_t> input = GetRandomInput(3);
  EXPECT_EQ(kEdgeTpuApiError, engine.RunInference(input.data(), input.size(),
                                                  &result, &result_size));
  ExpectNotInitialized(&engine);

  // Test debug functions.
  int const *input_tensor_shape;
  int input_tensor_dim;
  EXPECT_EQ(kEdgeTpuApiError, engine.get_input_tensor_shape(&input_tensor_shape,
                                                            &input_tensor_dim));
  ExpectNotInitialized(&engine);

  size_t const *output_tensor_sizes;
  size_t output_tensor_num;
  EXPECT_EQ(kEdgeTpuApiError, engine.get_all_output_tensors_sizes(
                                  &output_tensor_sizes, &output_tensor_num));
  ExpectNotInitialized(&engine);
  EXPECT_EQ(kEdgeTpuApiError,
            engine.get_num_of_output_tensors(&output_tensor_num));
  ExpectNotInitialized(&engine);

  size_t tmp;
  EXPECT_EQ(kEdgeTpuApiError, engine.get_output_tensor_size(0, &tmp));
  ExpectNotInitialized(&engine);
  EXPECT_EQ(kEdgeTpuApiError, engine.total_output_array_size(&tmp));
  ExpectNotInitialized(&engine);

  // Check model's path.
  std::string path;
  EXPECT_EQ(kEdgeTpuApiError, engine.model_path(&path));
  ExpectNotInitialized(&engine);
}

TEST(BasicEngineNativeTest, TestNegativeTensorIndex) {
  std::unique_ptr<BasicEngineNative> engine;
  BasicEngineNativeBuilder builder(
      TestDataPath("mobilenet_v1_1.0_224_quant.tflite"));
  ASSERT_EQ(kEdgeTpuApiOk, builder(&engine)) << builder.get_error_message();
  size_t tensor_size;
  EXPECT_EQ(kEdgeTpuApiError, engine->get_output_tensor_size(-1, &tensor_size));
  EXPECT_EQ("tensor_index must >= 0!", engine->get_error_message());
}

TEST(BasicEngineNativeTest, TestIndexOutOfbound) {
  std::unique_ptr<BasicEngineNative> engine;
  BasicEngineNativeBuilder builder(
      TestDataPath("mobilenet_v1_1.0_224_quant.tflite"));
  ASSERT_EQ(kEdgeTpuApiOk, builder(&engine)) << builder.get_error_message();
  size_t tensor_size;
  EXPECT_EQ(kEdgeTpuApiError,
            engine->get_output_tensor_size(1001, &tensor_size));
  EXPECT_EQ("tensor_index doesn't exist!", engine->get_error_message());
}

TEST(BasicEngineNativeTest, TestDebugFunctions) {
  std::unique_ptr<BasicEngineNative> engine;
  BasicEngineNativeBuilder builder(
      TestDataPath("mobilenet_v1_1.0_224_quant.tflite"));
  ASSERT_EQ(kEdgeTpuApiOk, builder(&engine)) << builder.get_error_message();

  int const *input_tensor_shape;
  int input_tensor_dim;
  // Check input dimensions.
  EXPECT_EQ(kEdgeTpuApiOk, engine->get_input_tensor_shape(&input_tensor_shape,
                                                          &input_tensor_dim));

  EXPECT_EQ(4, input_tensor_dim);
  EXPECT_EQ(1, input_tensor_shape[0]);
  EXPECT_EQ(224, input_tensor_shape[1]);
  EXPECT_EQ(224, input_tensor_shape[2]);
  EXPECT_EQ(3, input_tensor_shape[3]);

  size_t required_tensor_size;
  EXPECT_EQ(kEdgeTpuApiOk, engine->get_input_array_size(&required_tensor_size));
  EXPECT_EQ(224 * 224 * 3, required_tensor_size);
  // Check output tensors.
  int result_size = 1001;
  size_t const *output_tensor_sizes;
  size_t output_tensor_num;
  EXPECT_EQ(kEdgeTpuApiOk, engine->get_all_output_tensors_sizes(
                               &output_tensor_sizes, &output_tensor_num));
  EXPECT_EQ(1, output_tensor_num);
  EXPECT_EQ(result_size, output_tensor_sizes[0]);
  EXPECT_EQ(kEdgeTpuApiOk,
            engine->get_num_of_output_tensors(&output_tensor_num));
  EXPECT_EQ(1, output_tensor_num);
  size_t tmp;
  EXPECT_EQ(kEdgeTpuApiOk, engine->get_output_tensor_size(0, &tmp));
  EXPECT_EQ(result_size, tmp);

  EXPECT_EQ(kEdgeTpuApiOk, engine->total_output_array_size(&tmp));
  EXPECT_EQ(result_size, tmp);

  // Check model's path.
  std::string path;
  EXPECT_EQ(kEdgeTpuApiOk, engine->model_path(&path));
  EXPECT_EQ(TestDataPath("mobilenet_v1_1.0_224_quant.tflite"), path);
}

TEST(BasicEngineNativeTest, TestDebugFunctionsOnSsdModel) {
  // Test SSD model.
  std::unique_ptr<BasicEngineNative> engine;
  BasicEngineNativeBuilder builder(
      TestDataPath("ssd_mobilenet_v1_coco_quant_postprocess.tflite"));
  ASSERT_EQ(kEdgeTpuApiOk, builder(&engine)) << builder.get_error_message();
  int const *input_tensor_shape;
  int input_tensor_dim;
  size_t const *output_tensor_sizes;
  size_t output_tensor_num;
  EXPECT_EQ(kEdgeTpuApiOk, engine->get_all_output_tensors_sizes(
                               &output_tensor_sizes, &output_tensor_num));
  // Check input dimensions.
  EXPECT_EQ(kEdgeTpuApiOk, engine->get_input_tensor_shape(&input_tensor_shape,
                                                          &input_tensor_dim));
  EXPECT_EQ(4, input_tensor_dim);
  EXPECT_EQ(1, input_tensor_shape[0]);
  EXPECT_EQ(300, input_tensor_shape[1]);
  EXPECT_EQ(300, input_tensor_shape[2]);
  EXPECT_EQ(3, input_tensor_shape[3]);
  size_t input_array_size;
  EXPECT_EQ(kEdgeTpuApiOk, engine->get_input_array_size(&input_array_size));
  EXPECT_EQ(300 * 300 * 3, input_array_size);
  // This SSD models is trained to recognize at most 20 bounding boxes.
  EXPECT_EQ(4, output_tensor_num);
  EXPECT_EQ(80, output_tensor_sizes[0]);
  EXPECT_EQ(20, output_tensor_sizes[1]);
  EXPECT_EQ(20, output_tensor_sizes[2]);
  EXPECT_EQ(1, output_tensor_sizes[3]);

  size_t output_array_size;
  EXPECT_EQ(kEdgeTpuApiOk, engine->total_output_array_size(&output_array_size));
  EXPECT_EQ(121, output_array_size);
}

TEST(BasicEngineNativeTest, TestRunInferenceFailure) {
  // Initialize with tflite file.
  std::unique_ptr<tflite::FlatBufferModel> model(
      tflite::FlatBufferModel::BuildFromFile(
          TestDataPath("invalid_models/model_invoking_error.tflite").c_str()));
  auto resolver = absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
  resolver->AddCustom(coral::kFakeOpDouble, coral::RegisterFakeOpDouble());
  std::unique_ptr<BasicEngineNative> engine;
  BasicEngineNativeBuilder builder(std::move(model), std::move(resolver));
  ASSERT_EQ(kEdgeTpuApiOk, builder(&engine)) << builder.get_error_message();
  std::string path;
  EXPECT_EQ(kEdgeTpuApiError, engine->model_path(&path));
  EXPECT_EQ("No model path!", engine->get_error_message());
  float const *result;
  size_t result_size;
  std::vector<uint8_t> input = GetRandomInput(3);
  EXPECT_EQ(kEdgeTpuApiError, engine->RunInference(input.data(), input.size(),
                                                   &result, &result_size));
  EXPECT_EQ("Node number 0 (fake-op-double) failed to invoke.\n",
            engine->get_error_message());
}

TEST(BasicEngineNativeTest, TestRunInferenceFailure_InputBufferTooSmall) {
  std::unique_ptr<BasicEngineNative> engine;
  BasicEngineNativeBuilder builder(
      TestDataPath("mobilenet_v1_1.0_224_quant.tflite"));
  ASSERT_EQ(kEdgeTpuApiOk, builder(&engine)) << builder.get_error_message();
  float const *result;
  size_t result_size;
  std::vector<uint8_t> input = GetRandomInput(224 * 224 * 3 - 1);
  EXPECT_EQ(kEdgeTpuApiError, engine->RunInference(input.data(), input.size(),
                                                   &result, &result_size));
  EXPECT_EQ(
      "Input buffer size 150527 smaller than model input tensor size 150528.",
      engine->get_error_message());
}

TEST(BasicEngineNativeTest, TestRunInferenceSuccess) {
  std::unique_ptr<BasicEngineNative> engine;
  BasicEngineNativeBuilder builder(
      TestDataPath("mobilenet_v1_1.0_224_quant_edgetpu.tflite"));
  ASSERT_EQ(kEdgeTpuApiOk, builder(&engine)) << builder.get_error_message();
  float const *result;
  size_t result_size;
  std::vector<uint8_t> input = GetRandomInput(224 * 224 * 3);
  EXPECT_EQ(kEdgeTpuApiOk, engine->RunInference(input.data(), input.size(),
                                                &result, &result_size));
  EXPECT_EQ("", engine->get_error_message());
}

TEST(BasicEngineNativeTest, TestRunInferenceSuccessFloatInputs) {
  std::unique_ptr<BasicEngineNative> engine;
  BasicEngineNativeBuilder builder(
      TestDataPath("mobilenet_v1_1.0_224_ptq_float_io_edgetpu.tflite"));
  ASSERT_EQ(kEdgeTpuApiOk, builder(&engine)) << builder.get_error_message();
  float const *result;
  size_t result_size;

  std::vector<float> float_inputs(224 * 224 * 3);
  // Generates random inputs within [-1,1].
  std::generate(float_inputs.begin(), float_inputs.end(), [] {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(-1, 1);
    return dist(mt);
  });
  EXPECT_EQ(kEdgeTpuApiOk,
            engine->RunInference(float_inputs.data(), float_inputs.size(),
                                 &result, &result_size));
  EXPECT_EQ("", engine->get_error_message());
}

TEST(BasicEngineNativeTest, TestRunInferenceSuccess_PaddedInputBuffer) {
  std::unique_ptr<BasicEngineNative> engine;
  BasicEngineNativeBuilder builder(
      TestDataPath("mobilenet_v1_1.0_224_quant_edgetpu.tflite"));
  ASSERT_EQ(kEdgeTpuApiOk, builder(&engine)) << builder.get_error_message();
  float const *result;
  size_t result_size;
  // Input buffer has one extra padding byte.
  std::vector<uint8_t> input = GetRandomInput(224 * 224 * 3 + 1);
  EXPECT_EQ(kEdgeTpuApiOk, engine->RunInference(input.data(), input.size(),
                                                &result, &result_size));
  EXPECT_EQ("", engine->get_error_message());
}

}  // namespace
}  // namespace coral

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
