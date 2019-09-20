#include "src/cpp/learn/utils.h"

#include <cmath>

#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "flatbuffers/flexbuffers.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "src/cpp/basic/basic_engine.h"
#include "src/cpp/error_reporter.h"
#include "src/cpp/test_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace coral {
namespace learn {
namespace {

tflite::QuantizationParametersT* GetKernelQuant(const tflite::ModelT* model_t,
                                                int op_index) {
  CHECK(model_t);
  CHECK_EQ(model_t->subgraphs.size(), 1);
  auto& subgraph = model_t->subgraphs[0];
  CHECK_GT(subgraph->operators.size(), op_index);
  auto& conv2d_op = subgraph->operators[op_index];
  auto& kernel_tensor = subgraph->tensors[conv2d_op->inputs[1]];
  return kernel_tensor->quantization.get();
}

// Generates dummy quantization parameters for conv2d operator.
// It assumes input tensor of conv2d operator has value within range [-1.0, 1.0]
std::vector<std::unique_ptr<tflite::QuantizationParametersT>>
GenerateQuantParamsForConv2d() {
  // quant_params[0] is for kernel tensor.
  // quant_params[1] is for bias tensor.
  // quant_params[2] is for output tensor.
  std::vector<std::unique_ptr<tflite::QuantizationParametersT>> quant_params(3);
  quant_params[0] =
      CreateQuantParam(/*min=*/{-1.0f}, /*max=*/{1.0f}, /*scale=*/{1.0f / 128},
                       /*zero_point=*/{128});
  quant_params[1] =
      CreateQuantParam(/*min=*/{}, /*max=*/{}, /*scale=*/{1.0f / (128 * 128)},
                       /*zero_point=*/{0});
  quant_params[2] =
      CreateQuantParam(/*min=*/{-1.0f}, /*max=*/{1.0f}, /*scale=*/{1.0f / 128},
                       /*zero_point=*/{128});
  return quant_params;
}

// Builds a test graph that consists of
//    input_tensor
//        |
//        v
//      Conv2d
//        |
//        v
//    output_tensor
std::unique_ptr<tflite::ModelT> BuildTestGraph(
    const std::vector<int>& input_shape, const std::vector<int>& kernel_shape,
    const std::vector<float>& kernel) {
  EdgeTpuErrorReporter reporter;
  auto model_t = absl::make_unique<tflite::ModelT>();
  model_t->description = "Hand-crafted tflite graph for testing";
  // Must specify, current version is 3.
  model_t->version = 3;

  // Create sentinel buffer.
  internal::AppendBuffer(/*buffer_size_bytes=*/0, model_t.get());

  // Create a subgraph with only input tensor.
  auto subgraph = absl::make_unique<tflite::SubGraphT>();
  const int input_buffer_index =
      internal::AppendBuffer(/*buffer_size_bytes=*/0, model_t.get());
  auto input_tensor_quant =
      CreateQuantParam(/*min=*/{-128.0f}, /*max=*/{128.0f}, /*scale=*/{1.0f},
                       /*zero_point=*/{128});
  const int input_tensor_index = internal::AppendTensor(
      input_shape, /*name=*/"TestGraph/input", input_buffer_index,
      tflite::TensorType_UINT8, std::move(input_tensor_quant), subgraph.get());
  subgraph->inputs.push_back(input_tensor_index);
  // Current graph output is input tensor itself.
  subgraph->outputs.push_back(input_tensor_index);
  model_t->subgraphs.push_back(std::move(subgraph));

  // Add Conv2d Operator.
  auto conv2d_kernel_quant =
      CreateQuantParam(/*min=*/{-128.0f}, /*max=*/{128.0f}, /*scale=*/{1.0f},
                       /*zero_point=*/{128});
  auto conv2d_bias_quant =
      CreateQuantParam(/*min=*/{}, /*max=*/{}, /*scale=*/{1.0f},
                       /*zero_point=*/{0});
  auto conv2d_output_quant =
      CreateQuantParam(/*min=*/{-128.0f}, /*max=*/{128.0f}, /*scale=*/{1.0f},
                       /*zero_point=*/{128});

  std::vector<internal::TensorConfig> tensor_configs;
  std::vector<int> bias_shape = {kernel_shape[0]};

  const std::vector<tflite::TensorT*> output_tensors =
      GetGraphOutputTensors(model_t.get());
  CHECK_EQ(output_tensors.size(), 1);
  const tflite::TensorT* current_output_tensor = output_tensors[0];
  std::vector<int> output_shape = internal::CalculateConv2dOutputShape(
      current_output_tensor->shape, kernel_shape);
  tensor_configs.push_back({"TestGraph/Conv2d/Kernel", tflite::TensorType_UINT8,
                            internal::TensorLocation::kParameter, kernel_shape,
                            conv2d_kernel_quant.release()});
  tensor_configs.push_back({"TestGraph/Conv2d/Bias", tflite::TensorType_INT32,
                            internal::TensorLocation::kParameter, bias_shape,
                            conv2d_bias_quant.release()});
  tensor_configs.push_back({"TestGraph/Conv2d/Output", tflite::TensorType_UINT8,
                            internal::TensorLocation::kOutput, output_shape,
                            conv2d_output_quant.release()});
  int conv2d_op_index;
  CHECK_EQ(AppendOperator(tensor_configs, tflite::BuiltinOperator_CONV_2D,
                          model_t.get(), &conv2d_op_index, &reporter),
           kEdgeTpuApiOk);

  // Set kernel value.
  auto* kernel_quant = GetKernelQuant(model_t.get(), conv2d_op_index);
  const auto quant_kernel = Quantize<uint8_t>(kernel, kernel_quant->scale[0],
                                              kernel_quant->zero_point[0]);
  internal::SetConv2dParams(quant_kernel, /*bias=*/{}, conv2d_op_index,
                            model_t.get());
  return model_t;
}

// Returns pointer to graph input tensor. It assumes that there is only one
// input to the graph.
tflite::TensorT* GetGraphInputTensor(const tflite::ModelT* model_t) {
  CHECK(model_t);
  CHECK_EQ(model_t->subgraphs.size(), 1);
  const auto& subgraph = model_t->subgraphs[0];
  // Get current graph input.
  const auto& graph_inputs = subgraph->inputs;
  CHECK_EQ(graph_inputs.size(), 1);
  const int graph_input_tensor_index = graph_inputs[0];
  VLOG(1) << "Graph input tensor index: " << graph_input_tensor_index;
  const auto& graph_input_tensor = subgraph->tensors[graph_input_tensor_index];
  return graph_input_tensor.get();
}

// Runs inference with ModelT as input type.
std::vector<float> RunInference(const tflite::ModelT* model_t,
                                const std::vector<float>& input_tensor) {
  // Finish building flat buffer.
  auto fbb = GetFlatBufferBuilder(model_t);
  auto flat_buffer_model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(fbb->GetBufferPointer()), fbb->GetSize());
  BasicEngine basic_engine(std::move(flat_buffer_model));

  const tflite::TensorT* graph_input_tensor = GetGraphInputTensor(model_t);
  const auto& q_param = graph_input_tensor->quantization;
  const auto q_input_tensor = Quantize<uint8_t>(input_tensor, q_param->scale[0],
                                                q_param->zero_point[0]);
  std::vector<std::vector<float>> results =
      basic_engine.RunInference(q_input_tensor);
  CHECK_EQ(results.size(), 1);
  std::vector<float> result = results[0];
  int result_size = result.size();
  VLOG(1) << "result_size: " << result_size;
  for (int i = 0; i < result_size; ++i) {
    VLOG(1) << "index: " << i << " value: " << result[i];
  }
  return result;
}

std::unique_ptr<tflite::ModelT> LoadModel(const std::string& model_path) {
  std::string input_model_content;
  EdgeTpuErrorReporter error_reporter;
  CHECK_EQ(ReadFile(model_path, &input_model_content, &error_reporter),
           kEdgeTpuApiOk)
      << error_reporter.message();
  const tflite::Model* model = tflite::GetModel(input_model_content.data());
  CHECK(model);
  return absl::WrapUnique<tflite::ModelT>(model->UnPack());
}

TEST(UtilsTest, BuildTestGraphAndRunInference) {
  const std::vector<float> kernel = {
      1.0f, 1.0f, 1.0f, 1.0f, 1.0f,  // kernel 1
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f,  // kernel 2
  };
  auto model_t = BuildTestGraph(/*input_shape=*/{1, 1, 1, 5},
                                /*kernel_shape=*/{2, 1, 1, 5}, kernel);
  EXPECT_EQ(model_t->subgraphs.size(), 1);
  EXPECT_EQ(model_t->subgraphs[0]->operators.size(), 1);

  const auto result = RunInference(
      model_t.get(), /*input_tensor=*/{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  EXPECT_EQ(2, result.size());
  EXPECT_NEAR(15.0f, result[0], 0.01);
  EXPECT_NEAR(55.0f, result[1], 0.01);
}

TEST(UtilsTest, AppendL2Norm) {
  const std::vector<float> kernel = {
      1.0f,  1.0f,  1.0f,  1.0f,  1.0f,   // kernel 1
      -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  // kernel 2
      3.0f,  3.0f,  3.0f,  3.0f,  3.0f,   // kernel 3
  };
  auto model_t = BuildTestGraph(/*input_shape=*/{1, 1, 1, 5},
                                /*kernel_shape=*/{3, 1, 1, 5}, kernel);

  EdgeTpuErrorReporter error_reporter;
  int op_index;
  ASSERT_EQ(kEdgeTpuApiOk,
            internal::AppendL2Norm(model_t.get(), &op_index, &error_reporter));
  EXPECT_EQ(1, op_index);
  EXPECT_EQ(2, model_t->subgraphs[0]->operators.size());

  const auto result = RunInference(
      model_t.get(), /*input_tensor=*/{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  EXPECT_EQ(3, result.size());
  EXPECT_NEAR(1 / std::sqrt(11.0f), result[0], 0.01);
  EXPECT_NEAR(-1 / std::sqrt(11.0f), result[1], 0.01);
  EXPECT_NEAR(3 / std::sqrt(11.0f), result[2], 0.01);
}

TEST(UtilsTest, AppendFullyConnectedLayer) {
  const std::vector<float> kernel = {
      1.0f,  1.0f,  1.0f,  1.0f,  1.0f,   // kernel 1
      -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  // kernel 2
  };
  auto model_t = BuildTestGraph(/*input_shape=*/{1, 1, 1, 5},
                                /*kernel_shape=*/{2, 1, 1, 5}, kernel);
  EdgeTpuErrorReporter error_reporter;
  int op_index;
  ASSERT_EQ(kEdgeTpuApiOk,
            internal::AppendL2Norm(model_t.get(), &op_index, &error_reporter));
  ASSERT_EQ(kEdgeTpuApiOk,
            internal::AppendFullyConnectedLayer(
                /*kernel_shape=*/{4, 1, 1, 2}, GenerateQuantParamsForConv2d(),
                model_t.get(), &op_index, &error_reporter));
  ASSERT_EQ(2, op_index);
  // Set weights for fully-connected layer.
  const std::vector<float>& fc_weights = {
      std::sqrt(2.0f) / 2, -std::sqrt(2.0f) / 2,  // kernel 1
      std::sqrt(2.0f) / 2, std::sqrt(2.0f) / 2,   // kernel 2
      std::sqrt(7.0f) / 4, 3.0f / 4,              // kernel 3
      std::sqrt(5.0f) / 3, 2.0f / 3,              // kernel 4
  };

  auto* fc_weights_quant = GetKernelQuant(model_t.get(), op_index);
  const auto quant_fc_weights = Quantize<uint8_t>(
      fc_weights, fc_weights_quant->scale[0], fc_weights_quant->zero_point[0]);
  internal::SetConv2dParams(quant_fc_weights, /*bias=*/{}, op_index,
                            model_t.get());
  EXPECT_EQ(model_t->subgraphs[0]->operators.size(), 3);

  const auto result = RunInference(
      model_t.get(), /*input_tensor=*/{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  // output tensor of L2Norm layer is [sqrt(2)/2, -sqrt(2)/2], with above
  // `fc_weights`, result is expected to be:
  // [1, 0, (sqrt(14)-sqrt(18))/8, (sqrt(10)-sqrt(8))/6]
  EXPECT_EQ(4, result.size());
  EXPECT_NEAR(1.0f, result[0], 0.01);
  EXPECT_NEAR(0.0f, result[1], 0.01);
  EXPECT_NEAR((std::sqrt(14.0f) - std::sqrt(18.0f)) / 8, result[2], 0.01);
  EXPECT_NEAR((std::sqrt(10.0f) - std::sqrt(8.0f)) / 6, result[3], 0.01);
}

TEST(UtilsTest, AppendReshape) {
  const std::vector<float> kernel = {
      1.0f,  1.0f,  1.0f,  1.0f,  1.0f,   // kernel 1
      -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  // kernel 2
  };
  auto model_t = BuildTestGraph(/*input_shape=*/{1, 1, 1, 5},
                                /*kernel_shape=*/{2, 1, 1, 5}, kernel);
  EdgeTpuErrorReporter error_reporter;
  int op_index;
  ASSERT_EQ(kEdgeTpuApiOk,
            internal::AppendL2Norm(model_t.get(), &op_index, &error_reporter));
  ASSERT_EQ(kEdgeTpuApiOk,
            internal::AppendFullyConnectedLayer(
                /*kernel_shape=*/{4, 1, 1, 2}, GenerateQuantParamsForConv2d(),
                model_t.get(), &op_index, &error_reporter));
  const auto fc_op_index = op_index;
  ASSERT_EQ(kEdgeTpuApiOk,
            internal::AppendReshape(model_t.get(), &op_index, &error_reporter));
  ASSERT_EQ(3, op_index);

  // Set weights for fully-connected layer.
  const std::vector<float>& fc_weights = {
      std::sqrt(2.0f) / 2, -std::sqrt(2.0f) / 2,  // kernel 1
      std::sqrt(2.0f) / 2, std::sqrt(2.0f) / 2,   // kernel 2
      std::sqrt(7.0f) / 4, 3.0f / 4,              // kernel 3
      std::sqrt(5.0f) / 3, 2.0f / 3,              // kernel 4
  };

  auto* fc_weights_quant = GetKernelQuant(model_t.get(), fc_op_index);
  const auto quant_fc_weights = Quantize<uint8_t>(
      fc_weights, fc_weights_quant->scale[0], fc_weights_quant->zero_point[0]);
  internal::SetConv2dParams(quant_fc_weights, /*bias=*/{}, fc_op_index,
                            model_t.get());

  ASSERT_EQ(model_t->subgraphs[0]->operators.size(), 4);
  // Check graph's output tensor's shape.
  const std::vector<tflite::TensorT*> output_tensors =
      GetGraphOutputTensors(model_t.get());
  ASSERT_EQ(output_tensors.size(), 1);
  const tflite::TensorT* current_output_tensor = output_tensors[0];
  ASSERT_EQ(current_output_tensor->shape.size(), 2);
  EXPECT_EQ(current_output_tensor->shape[0], 1);
  EXPECT_EQ(current_output_tensor->shape[1], 4);

  const auto result = RunInference(
      model_t.get(), /*input_tensor=*/{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  // output tensor of L2Norm layer is [sqrt(2)/2, -sqrt(2)/2], with above
  // `fc_weights`, result is expected to be:
  // [1, 0, (sqrt(14)-sqrt(18))/8, (sqrt(10)-sqrt(8))/6]
  ASSERT_EQ(4, result.size());
  EXPECT_NEAR(1.0f, result[0], 0.01);
  EXPECT_NEAR(0.0f, result[1], 0.01);
  EXPECT_NEAR((std::sqrt(14.0f) - std::sqrt(18.0f)) / 8, result[2], 0.01);
  EXPECT_NEAR((std::sqrt(10.0f) - std::sqrt(8.0f)) / 6, result[3], 0.01);
}

TEST(UtilsTest, AppendSoftmax) {
  const std::vector<float> kernel = {
      1.0f,  1.0f,  1.0f,  1.0f,  1.0f,   // kernel 1
      -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  // kernel 2
  };
  auto model_t = BuildTestGraph(/*input_shape=*/{1, 1, 1, 5},
                                /*kernel_shape=*/{2, 1, 1, 5}, kernel);
  EdgeTpuErrorReporter error_reporter;
  int op_index;
  ASSERT_EQ(kEdgeTpuApiOk,
            internal::AppendL2Norm(model_t.get(), &op_index, &error_reporter));
  ASSERT_EQ(kEdgeTpuApiOk,
            internal::AppendFullyConnectedLayer(
                /*kernel_shape=*/{4, 1, 1, 2}, GenerateQuantParamsForConv2d(),
                model_t.get(), &op_index, &error_reporter));
  const auto fc_op_index = op_index;
  ASSERT_EQ(kEdgeTpuApiOk,
            internal::AppendReshape(model_t.get(), &op_index, &error_reporter));
  ASSERT_EQ(kEdgeTpuApiOk,
            internal::AppendSoftmax(model_t.get(), &op_index, &error_reporter));
  ASSERT_EQ(4, op_index);

  // Set weights for fully-connected layer.
  const std::vector<float>& fc_weights = {
      std::sqrt(2.0f) / 2, -std::sqrt(2.0f) / 2,  // kernel 1
      std::sqrt(2.0f) / 2, std::sqrt(2.0f) / 2,   // kernel 2
      std::sqrt(7.0f) / 4, 3.0f / 4,              // kernel 3
      std::sqrt(5.0f) / 3, 2.0f / 3,              // kernel 4
  };

  auto* fc_weights_quant = GetKernelQuant(model_t.get(), fc_op_index);
  const auto quant_fc_weights = Quantize<uint8_t>(
      fc_weights, fc_weights_quant->scale[0], fc_weights_quant->zero_point[0]);
  internal::SetConv2dParams(quant_fc_weights, /*bias=*/{}, fc_op_index,
                            model_t.get());

  EXPECT_EQ(model_t->subgraphs[0]->operators.size(), 5);

  const auto result = RunInference(
      model_t.get(), /*input_tensor=*/{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  ASSERT_EQ(4, result.size());
  // Result after Fully-connect layer is:
  // [1, 0, (sqrt(14)-sqrt(18))/8, (sqrt(10)-sqrt(8))/6]
  std::vector<float> unnormalized_results = {
      std::exp(1.0f), std::exp(0.0f),
      std::exp((std::sqrt(14.0f) - std::sqrt(18.0f)) / 8),
      std::exp((std::sqrt(10.0f) - std::sqrt(8.0f)) / 6)};

  float sum = 0;
  for (const auto& r : unnormalized_results) {
    sum += r;
  }
  EXPECT_NEAR(unnormalized_results[0] / sum, result[0], 0.01);
  EXPECT_NEAR(unnormalized_results[1] / sum, result[1], 0.01);
  EXPECT_NEAR(unnormalized_results[2] / sum, result[2], 0.01);
  EXPECT_NEAR(unnormalized_results[3] / sum, result[3], 0.01);
}

// Test finding operators with a real model.
TEST(UtilsTest, FindOperators) {
  const std::string& model_path =
      TestDataPath("mobilenet_v1_1.0_224_quant.tflite");
  const auto model_t = LoadModel(model_path);
  const tflite::BuiltinOperator target_op = tflite::BuiltinOperator_CONV_2D;
  const auto& conv_op_indices = FindOperators(target_op, model_t.get());
  EXPECT_EQ(
      std::vector<int>({0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28}),
      conv_op_indices);
}

// Test finding a single operator with a real model.
TEST(UtilsTest, FindSingleOperator) {
  const std::string& model_path =
      TestDataPath("mobilenet_v1_1.0_224_quant.tflite");
  const auto model_t = LoadModel(model_path);
  EXPECT_EQ(30,
            FindSingleOperator(tflite::BuiltinOperator_SOFTMAX, model_t.get()));
  EXPECT_EQ(-1,
            FindSingleOperator(tflite::BuiltinOperator_LSTM, model_t.get()));
  EXPECT_EQ(-1, FindSingleOperator(tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                                   model_t.get()));
}

// Test finding operators given input tensor with a real model.
TEST(UtilsTest, FindOperatorsWithInput) {
  const std::string& model_path =
      TestDataPath("mobilenet_v1_1.0_224_quant.tflite");
  const auto model_t = LoadModel(model_path);
  const tflite::BuiltinOperator target_op = tflite::BuiltinOperator_CONV_2D;
  // Use MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6 as input tensor.
  const int input_tensor_index = 75;
  const int base_op_index = 0;
  const std::vector<int> conv_op_indices = FindOperatorsWithInput(
      target_op, input_tensor_index, model_t.get(), base_op_index);
  EXPECT_EQ(std::vector<int>({16}), conv_op_indices);
}

// Test finding a single operator given input tensor with a real model.
TEST(UtilsTest, FindSingleOperatorWithInput) {
  const std::string& model_path =
      TestDataPath("mobilenet_v1_1.0_224_quant.tflite");
  const auto model_t = LoadModel(model_path);
  // Use MobilenetV1/Logits/SpatialSqueeze as input tensor.
  const int input_tensor_index = 4;
  const int base_op_index = 0;
  const int softmax_op_index = FindSingleOperatorWithInput(
      tflite::BuiltinOperator_SOFTMAX, input_tensor_index, model_t.get(),
      base_op_index);
  EXPECT_EQ(30, softmax_op_index);

  const int nonexist_op_index = FindSingleOperatorWithInput(
      tflite::BuiltinOperator_LSTM, input_tensor_index, model_t.get(),
      base_op_index);
  EXPECT_EQ(-1, nonexist_op_index);
}

TEST(UtilsTest, AppendFullyConnectedAndSoftmaxLayerToModel) {
  const std::string& in_model_path = TestDataPath(
      "mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite");
  const std::string out_model_path = "/tmp/retrained_model_edgetpu.tflite";

  BasicEngine in_model_engine(in_model_path);
  const auto& input_tensor_shape = in_model_engine.get_input_tensor_shape();
  std::vector<uint8_t> random_input = GetRandomInput(
      input_tensor_shape[1] * input_tensor_shape[2] * input_tensor_shape[3]);
  const auto& in_model_results = in_model_engine.RunInference(random_input);
  ASSERT_EQ(1, in_model_results.size());
  const auto& in_model_result = in_model_results[0];

  const int embedding_vector_dim =
      in_model_engine.get_all_output_tensors_sizes()[0];
  float embedding_vector_sum = 0.0f;
  for (int i = 0; i < embedding_vector_dim; ++i) {
    embedding_vector_sum += in_model_result[i];
  }
  // Generates dummy weights, of dimension embedding_vector_dim x 3. Each kernel
  // has the following pattern (times a scalar to make max logits score = 1) :
  // Kernel 1: 1, 1, 1, ...
  // Kernel 2: 2, 2, 2, ...
  // kernel 3: 3, 3, 3, ...
  std::vector<float> weights(embedding_vector_dim * 3);
  const float scalar = 1 / (embedding_vector_sum * 3);
  std::fill(weights.begin(), weights.begin() + embedding_vector_dim, scalar);
  std::fill(weights.begin() + embedding_vector_dim,
            weights.begin() + embedding_vector_dim * 2, scalar * 2);
  std::fill(weights.begin() + embedding_vector_dim * 2,
            weights.begin() + embedding_vector_dim * 3, scalar * 3);
  std::vector<float> biases(3, 0.0f);

  std::vector<float> expected_fc_output = {embedding_vector_sum * scalar,
                                           embedding_vector_sum * scalar * 2,
                                           embedding_vector_sum * scalar * 3};
  const float out_tensor_min =
      *std::min_element(expected_fc_output.begin(), expected_fc_output.end());
  const float out_tensor_max =
      *std::max_element(expected_fc_output.begin(), expected_fc_output.end());

  EdgeTpuErrorReporter reporter;
  ASSERT_EQ(AppendFullyConnectedAndSoftmaxLayerToModel(
                in_model_path, out_model_path, weights.data(), weights.size(),
                biases.data(), biases.size(), out_tensor_min, out_tensor_max,
                &reporter),
            kEdgeTpuApiOk);

  BasicEngine out_model_engine(out_model_path);
  const auto& out_model_results = out_model_engine.RunInference(random_input);
  ASSERT_EQ(1, out_model_results.size());
  const auto& out_model_result = out_model_results[0];

  // Calculate expected value.
  std::vector<float> expected = expected_fc_output;
  float max_score = *std::max_element(expected.begin(), expected.end());
  // Subtract max_score to avoid overflow.
  for (auto& e : expected) {
    e -= max_score;
  }
  float exp_sum = 0.0;
  for (auto& e : expected) {
    e = std::exp(e);
    exp_sum += e;
  }
  for (auto& e : expected) {
    e /= exp_sum;
  }

  ASSERT_EQ(3, out_model_result.size());
  const float tol = 5e-3;
  for (int i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(expected[i], out_model_result[i], tol);
  }
}

TEST(UtilsTest,
     AppendFullyConnectedAndSoftmaxLayerToModelInvalidInputFilePath) {
  const std::string& in_model_path = TestDataPath("invalid_path.tflite");
  const std::string out_model_path = "/tmp/retrained_model_edgetpu.tflite";

  const int embedding_vector_dim = 1024;
  std::vector<float> weights(embedding_vector_dim * 3, 0.0f);
  std::vector<float> biases(3, 0.0f);
  const float out_tensor_min = -1.0f;
  const float out_tensor_max = 1.0f;

  EdgeTpuErrorReporter reporter;
  ASSERT_EQ(AppendFullyConnectedAndSoftmaxLayerToModel(
                in_model_path, out_model_path, weights.data(), weights.size(),
                biases.data(), biases.size(), out_tensor_min, out_tensor_max,
                &reporter),
            kEdgeTpuApiError);
  const std::string& expected_message =
      absl::Substitute("Failed to open file: $0", in_model_path);
  EXPECT_EQ(reporter.message(), expected_message);
}

}  // namespace
}  // namespace learn
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
