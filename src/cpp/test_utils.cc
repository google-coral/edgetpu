#include "src/cpp/test_utils.h"

#include <sys/types.h>

#include <chrono>  // NOLINT
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <thread>  // NOLINT

#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "src/cpp/basic/basic_engine.h"
#include "src/cpp/classification/engine.h"
#include "src/cpp/detection/engine.h"
#include "src/cpp/utils.h"
#include "tensorflow/lite/builtin_op_data.h"

ABSL_FLAG(std::string, test_data_dir, "test_data", "Test data directory");

namespace coral {

std::string GetTempPrefix() {
  const char* env_temp = getenv("TEMP");
  if (env_temp) {
    return env_temp;
  } else {
    return "/tmp";
  }
}

namespace {

using tflite::ops::builtin::BuiltinOpResolver;

template <typename SrcType, typename DstType>
DstType saturate_cast(SrcType val) {
  if (val > static_cast<SrcType>(std::numeric_limits<DstType>::max())) {
    return std::numeric_limits<DstType>::max();
  }
  if (val < static_cast<SrcType>(std::numeric_limits<DstType>::lowest())) {
    return std::numeric_limits<DstType>::lowest();
  }
  return static_cast<DstType>(val);
}

// Returns whether string ends with given suffix.
inline bool EndsWith(std::string const& value, std::string const& ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

// Returns total number of elements.
int ImageDimsToSize(const ImageDims& dims) {
  int size = 1;
  for (const auto& dim : dims) {
    size *= dim;
  }
  return size;
}

std::vector<uint8_t> DecodeBmp(const uint8_t* input, int row_size, int width,
                               int height, int channels, bool top_down) {
  std::vector<uint8_t> output(height * width * channels);
  for (int i = 0; i < height; i++) {
    int src_pos;
    int dst_pos;
    for (int j = 0; j < width; j++) {
      if (!top_down) {
        src_pos = ((height - 1 - i) * row_size) + j * channels;
      } else {
        src_pos = i * row_size + j * channels;
      }
      dst_pos = (i * width + j) * channels;
      switch (channels) {
        case 1:
          output[dst_pos] = input[src_pos];
          break;
        case 3:
          // BGR -> RGB
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          break;
        case 4:
          // BGRA -> RGBA
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          output[dst_pos + 3] = input[src_pos + 3];
          break;
        default:
          LOG(FATAL) << "Unexpected number of channels: " << channels;
          break;
      }
    }
  }
  return output;
}

// Reads BMP image. It will crahs upon failure.
std::vector<uint8_t> ReadBmp(const std::string& input_bmp_name,
                             ImageDims* image_dims) {
  std::string file_content;
  ReadFileOrDie(input_bmp_name, &file_content);
  CHECK(!file_content.empty()) << "Bmp image file is empty " << input_bmp_name;
  const uint8_t* img_bytes =
      reinterpret_cast<const uint8_t*>(file_content.data());

  // Data in BMP file header is stored in Little Endian format. The following
  // method should work on both Big and Little Endian machine.
  auto to_int32 = [](const unsigned char* p) -> int32_t {
    return p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
  };
  const int32_t header_size = to_int32(img_bytes + 10);
  const int32_t bpp = to_int32(img_bytes + 28);
  int* width = image_dims->data() + 1;
  int* height = image_dims->data();
  int* channels = image_dims->data() + 2;
  *width = to_int32(img_bytes + 18);
  *height = to_int32(img_bytes + 22);
  *channels = bpp / 8;
  // Currently supports RGB and grayscale image at this function.
  CHECK(*width > 0 && *height > 0 && (*channels == 3 || *channels == 1))
      << "Unsupported image format. width, height, channels: " << *width << ", "
      << *height << ", " << *channels << "\n";

  // there may be padding bytes when the width is not a multiple of 4 bytes
  // 8 * channels == bits per pixel
  const int row_size = (8 * (*channels) * (*width) + 31) / 32 * 4;
  // if height is negative, data layout is top down
  // otherwise, it's bottom up
  bool top_down = (*height < 0);
  // Decode image, allocating tensor once the image size is known
  const uint8_t* bmp_pixels = &img_bytes[header_size];
  return DecodeBmp(bmp_pixels, row_size, *width, abs(*height), *channels,
                   top_down);
}

// Resizes BMP image.
void ResizeImage(const ImageDims& in_dims, const uint8_t* in,
                 const ImageDims& out_dims, uint8_t* out) {
  const int image_height = in_dims[0];
  const int image_width = in_dims[1];
  const int image_channels = in_dims[2];
  const int wanted_height = out_dims[0];
  const int wanted_width = out_dims[1];
  const int wanted_channels = out_dims[2];
  const int number_of_pixels = image_height * image_width * image_channels;
  if (image_height == wanted_height && image_width == wanted_width &&
      image_channels == wanted_channels) {
    VLOG(1) << "No resizing needed for input image.";
    std::memcpy(out, in, number_of_pixels);
    return;
  }
  std::unique_ptr<tflite::Interpreter> interpreter(new tflite::Interpreter);
  int base_index = 0;
  // two inputs: input and new_sizes
  interpreter->AddTensors(2, &base_index);
  // one output
  interpreter->AddTensors(1, &base_index);
  // set input and output tensors
  interpreter->SetInputs({0, 1});
  interpreter->SetOutputs({2});
  // set parameters of tensors
  TfLiteQuantizationParams quant;
  interpreter->SetTensorParametersReadWrite(
      0, kTfLiteFloat32, "input",
      {1, image_height, image_width, image_channels}, quant);
  interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2},
                                            quant);
  interpreter->SetTensorParametersReadWrite(
      2, kTfLiteFloat32, "output",
      {1, wanted_height, wanted_width, wanted_channels}, quant);
  BuiltinOpResolver resolver;
  const TfLiteRegistration* resize_op =
      resolver.FindOp(tflite::BuiltinOperator_RESIZE_BILINEAR, 1);
  auto* params = reinterpret_cast<TfLiteResizeBilinearParams*>(
      malloc(sizeof(TfLiteResizeBilinearParams)));
  params->align_corners = false;
  params->half_pixel_centers = false;
  interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op,
                                     nullptr);
  interpreter->AllocateTensors();
  // fill input image
  // in[] are integers, cannot do memcpy() directly
  auto input = interpreter->typed_tensor<float>(0);
  for (int i = 0; i < number_of_pixels; i++) {
    input[i] = in[i];
  }
  // fill new_sizes
  interpreter->typed_tensor<int>(1)[0] = wanted_height;
  interpreter->typed_tensor<int>(1)[1] = wanted_width;
  interpreter->Invoke();
  auto output = interpreter->typed_tensor<float>(2);
  auto output_number_of_pixels =
      wanted_height * wanted_height * wanted_channels;
  for (int i = 0; i < output_number_of_pixels; i++) {
    out[i] = static_cast<uint8_t>(output[i]);
  }
}

// Converts RGB image to grayscale. Take the average.
std::vector<uint8_t> RgbToGrayscale(const std::vector<uint8_t>& in,
                                    const ImageDims& in_dims) {
  CHECK_GE(in_dims[2], 3);
  std::vector<uint8_t> result;
  int out_size = in_dims[0] * in_dims[1];
  result.resize(out_size);
  for (int in_idx = 0, out_idx = 0; in_idx < in.size();
       in_idx += in_dims[2], ++out_idx) {
    int r = in[in_idx];
    int g = in[in_idx + 1];
    int b = in[in_idx + 2];
    result[out_idx] = static_cast<uint8_t>((r + g + b) / 3);
  }
  return result;
}

}  // namespace

std::string TestDataPath(const std::string& name) {
  return absl::StrCat(absl::GetFlag(FLAGS_test_data_dir), "/", name);
}

std::vector<uint8_t> GetRandomInput(const int n) {
  unsigned int seed = 1;
  std::default_random_engine generator(seed);
  std::uniform_int_distribution<> distribution(0, UINT8_MAX);
  std::vector<uint8_t> result;
  result.resize(n);
  for (int i = 0; i < n; ++i) {
    result[i] = distribution(generator);
  }
  return result;
}

std::vector<uint8_t> GetRandomInput(std::vector<int> shape) {
  int n = 1;
  for (int i = 0; i < shape.size(); ++i) {
    n *= shape[i];
  }
  return GetRandomInput(n);
}

std::vector<uint8_t> GetInputFromImage(const std::string& image_path,
                                       const ImageDims& target_dims) {
  std::vector<uint8_t> result;
  if (!EndsWith(image_path, ".bmp")) {
    LOG(FATAL) << "Unsupported image type: " << image_path;
    return result;
  }
  result.resize(ImageDimsToSize(target_dims));
  ImageDims image_dims;
  std::vector<uint8_t> in = ReadBmp(image_path, &image_dims);
  CHECK(!in.empty()) << "Fail to read bmp image from file: " << image_path;

  if (target_dims[2] == 1 && (image_dims[2] == 3 || image_dims[2] == 4)) {
    in = RgbToGrayscale(in, image_dims);
    image_dims[2] = 1;
  }
  ResizeImage(image_dims, in.data(), target_dims, result.data());
  return result;
}

void TestWithRandomInput(const RandomInputTestParams& test_params) {
  TestWithRandomInput(test_params.model_path,
                      test_params.expected_num_output_tensors);
}

void TestWithRandomInput(const std::string& model_path,
                         int expected_num_output_tensors) {
  LOG(INFO) << "Testing model: " << model_path;
  BasicEngine engine(model_path);
  const auto& results =
      engine.RunInference(GetRandomInput(engine.get_input_tensor_shape()));
  ASSERT_EQ(expected_num_output_tensors, results.size());
  for (const auto& r : results) {
    EXPECT_FALSE(r.empty());
  }
}

std::string GenerateRandomFilePath(const std::string& prefix,
                                   const std::string& suffix) {
  return absl::StrCat(std::string(std::tmpnam(nullptr)), "_", prefix, suffix);
}

std::vector<std::vector<float>> TestWithImage(const std::string& model_path,
                                              const std::string& image_path) {
  // Load the model.
  LOG(INFO) << "Testing model: " << model_path;
  BasicEngine engine(model_path);
  std::vector<int> input_tensor_shape = engine.get_input_tensor_shape();
  // Read image.
  std::vector<uint8_t> input = GetInputFromImage(
      image_path,
      {input_tensor_shape[1], input_tensor_shape[2], input_tensor_shape[3]});
  CHECK(!input.empty()) << "Input image path: " << image_path;
  // Get result.
  return engine.RunInference(input);
}

bool TopKContains(const std::vector<ClassificationCandidate>& topk, int label) {
  for (const auto& entry : topk) {
    if (entry.id == label) return true;
  }
  LOG(ERROR) << "Top K results do not contain " << label;
  for (const auto& p : topk) {
    LOG(ERROR) << p.id << ", " << p.score;
  }
  return false;
}

void TestClassification(const ClassificationTestParams& test_params) {
  TestClassification(test_params.model_path, test_params.image_path,
                     test_params.effective_scale, test_params.effective_means,
                     test_params.rgb2bgr, test_params.score_threshold,
                     test_params.k, test_params.expected_topk_label);
}

void TestClassification(const std::string& model_path,
                        const std::string& image_path, float effective_scale,
                        const std::vector<float>& effective_means, bool rgb2bgr,
                        float score_threshold, int k, int expected_topk_label) {
  LOG(INFO) << "Testing parameters:";
  LOG(INFO) << "model_path: " << model_path;
  LOG(INFO) << "image_path: " << image_path;
  LOG(INFO) << "effective_scale: " << effective_scale;
  for (int i = 0; i < effective_means.size(); ++i)
    LOG(INFO) << "effective_means: " << effective_means[i];
  LOG(INFO) << "score_threshold: " << score_threshold;
  LOG(INFO) << "k: " << k;
  LOG(INFO) << "expected_topk_label: " << expected_topk_label;
  LOG(INFO) << "rgb2bgr: " << rgb2bgr;
  // Load the model.
  ClassificationEngine engine(model_path);
  std::vector<int> input_tensor_shape = engine.get_input_tensor_shape();
  // Read image.
  std::vector<uint8_t> input_tensor = GetInputFromImage(
      image_path,
      {input_tensor_shape[1], input_tensor_shape[2], input_tensor_shape[3]});

  const int num_channels = effective_means.size();
  if (rgb2bgr) {
    for (int i = 0; i < input_tensor.size(); i += num_channels) {
      input_tensor[i] = saturate_cast<float, uint8_t>(
          (input_tensor[i + 2] - effective_means[0]) / effective_scale);
      input_tensor[i + 1] = saturate_cast<float, uint8_t>(
          (input_tensor[i + 1] - effective_means[1]) / effective_scale);
      input_tensor[i + 2] = saturate_cast<float, uint8_t>(
          (input_tensor[i] - effective_means[2]) / effective_scale);
    }
  } else {
    for (int i = 0; i < input_tensor.size(); i += num_channels) {
      input_tensor[i] = saturate_cast<float, uint8_t>(
          (input_tensor[i] - effective_means[0]) / effective_scale);
      input_tensor[i + 1] = saturate_cast<float, uint8_t>(
          (input_tensor[i + 1] - effective_means[1]) / effective_scale);
      input_tensor[i + 2] = saturate_cast<float, uint8_t>(
          (input_tensor[i + 2] - effective_means[2]) / effective_scale);
    }
  }

  CHECK(!input_tensor.empty()) << "Input image path: " << image_path;
  const auto results =
      engine.ClassifyWithInputTensor(input_tensor, score_threshold, k);
  const bool top_k_contains_expected =
      TopKContains(results, expected_topk_label);
  if (!top_k_contains_expected) {
    LOG(ERROR) << "Top " << k << " results do not contain expected label "
               << expected_topk_label << " with threshold=" << score_threshold;
    const auto no_threshold_results = engine.ClassifyWithInputTensor(
        input_tensor, -std::numeric_limits<float>::infinity(), k);
    LOG(ERROR) << "Without score threshold, top " << k << " results are:";
    for (const auto& p : no_threshold_results) {
      LOG(ERROR) << p.id << ", " << p.score;
    }
  }
  EXPECT_TRUE(top_k_contains_expected);
}

// Tests a classification model with customized preprocessing.
void TestClassification(const std::string& model_path,
                        const std::string& image_path, float effective_scale,
                        const std::vector<float>& effective_means,
                        float score_threshold, int k, int expected_topk_label) {
  TestClassification(model_path, image_path, effective_scale, effective_means,
                     /*rgb2bgr=*/false, score_threshold, k,
                     expected_topk_label);
}

void TestClassification(const std::string& model_path,
                        const std::string& image_path, float score_threshold,
                        int k, int expected_topk_label) {
  const std::vector<float> effective_means{0, 0, 0};
  TestClassification(model_path, image_path, /*effective_scale=*/1.f,
                     effective_means, score_threshold, k, expected_topk_label);
}

void TestClassification(const std::string& model_path,
                        const std::string& image_path, float score_threshold,
                        int expected_top1_label) {
  TestClassification(model_path, image_path, score_threshold, /*k=*/1,
                     expected_top1_label);
}

void TestDetection(const std::string& model_path, const std::string& image_path,
                   const BoxCornerEncoding& expected_box, int expected_label,
                   float score_threshold, float iou_threshold) {
  DetectionEngine engine(model_path);
  std::vector<int> input_tensor_shape = engine.get_input_tensor_shape();
  // Read image.
  std::vector<uint8_t> input_tensor = GetInputFromImage(
      image_path,
      {input_tensor_shape[1], input_tensor_shape[2], input_tensor_shape[3]});

  auto candiates =
      engine.DetectWithInputTensor(input_tensor, score_threshold, /*top_k=*/1);
  ASSERT_EQ(candiates.size(), 1);
  DetectionCandidate result = candiates[0];
  EXPECT_EQ(result.label, expected_label);
  EXPECT_GT(result.score, score_threshold);
  EXPECT_GT(IntersectionOverUnion(result.corners, expected_box), iou_threshold);
}

void TestCatMsCocoDetection(const std::string& model_path,
                            float score_threshold, float iou_threshold) {
  TestDetection(model_path, TestDataPath("cat.bmp"),
                /*expected_box=*/{0.1, 0.1, 0.7, 1.0},
                /*expected_label=*/16, score_threshold, iou_threshold);
}

void BenchmarkModelsOnEdgeTpu(const std::vector<std::string>& model_paths,
                              benchmark::State& state) {
  const int number_models = model_paths.size();
  std::vector<std::unique_ptr<coral::BasicEngine>> engines;
  std::vector<std::vector<uint8_t>> inputs;
  for (int model_index = 0; model_index < number_models; ++model_index) {
    const auto& model_path = model_paths[model_index];
    std::unique_ptr<coral::BasicEngine> engine;
    if (model_index == 0) {
      engine = absl::make_unique<coral::BasicEngine>(model_path);
    } else {
      // Engines should run on the same EdgeTpu device.
      engine = absl::make_unique<coral::BasicEngine>(model_path,
                                                     engines[0]->device_path());
    }
    const auto& model_input = GetRandomInput(engine->get_input_tensor_shape());
    inputs.push_back(model_input);
    engines.push_back(std::move(engine));
  }
  while (state.KeepRunning()) {
    for (int i = 0; i < engines.size(); ++i) {
      engines[i]->RunInference(inputs[i]);
    }
  }
}

void BenchmarkModelOnEdgeTpu(const std::string& model_path,
                             benchmark::State& state) {
  BenchmarkModelsOnEdgeTpu({model_path}, state);
}

void RepeatabilityTest(const std::string& model_path, int runs) {
  BasicEngine engine(model_path);
  const auto& input_data = GetRandomInput(engine.get_input_tensor_shape());
  int error_count = 0;
  std::vector<std::vector<float>> reference_result =
      engine.RunInference(input_data);
  for (int r = 0; r < runs; ++r) {
    VLOG_EVERY_N(0, 100) << "inference running iter " << r << "...";
    const auto& result = engine.RunInference(input_data);
    const int num_outputs = result.size();
    CHECK_GT(num_outputs, 0);
    for (int i = 0; i < num_outputs; ++i) {
      for (int j = 0; j < result[i].size(); ++j) {
        if (result[i][j] != reference_result[i][j]) {
          VLOG(1) << "[ iteration = " << r << " ] output of tensor " << i
                  << " at position " << j << " differs from reference.\n"
                  << "( output = " << result[i][j]
                  << " reference = " << reference_result[i][j] << " )";
          ++error_count;
        }
      }
    }
  }
  EXPECT_EQ(0, error_count) << "total runs " << runs;
}

void InferenceStressTest(const std::string& model_path, int runs,
                         int sleep_sec) {
  BasicEngine engine(model_path);
  for (int i = 0; i < runs; ++i) {
    VLOG_EVERY_N(0, 100) << "inference running iter " << i << "...";
    const auto& input_data = GetRandomInput(engine.get_input_tensor_shape());
    const auto& result = engine.RunInference(input_data);
    CHECK(!result.empty());
    std::this_thread::sleep_for(std::chrono::seconds(sleep_sec));
  }
}

float ComputeIntersectionOverUnion(const std::vector<uint8_t>& mask1,
                                   const std::vector<uint8_t>& mask2) {
  int isec_area = 0;
  for (int i = 0; i < mask1.size(); ++i) {
    if (mask1[i] == mask2[i]) {
      ++isec_area;
    }
  }
  return static_cast<float>(isec_area) /
         (mask1.size() + mask2.size() - isec_area);
}

std::vector<float> ApplyArgmax(const std::vector<float>& raw_result,
                               int image_size) {
  CHECK_EQ(raw_result.size() % (image_size * image_size), 0);
  const int num_classes = raw_result.size() / (image_size * image_size);
  CHECK_GT(num_classes, 1);
  std::vector<float> argmax_result(image_size * image_size);
  for (int i = 0; i < image_size * image_size; ++i) {
    argmax_result[i] = std::distance(
        raw_result.begin() + i * num_classes,
        std::max_element(raw_result.begin() + i * num_classes,
                         raw_result.begin() + (i + 1) * num_classes));
  }
  return argmax_result;
}

void TestSegmentation(const std::string& model_name,
                      const std::string& image_name,
                      const std::string& groundtruth_name, int image_size,
                      float iou_threshold, bool model_has_argmax,
                      std::vector<uint8_t>* pred_segmentation) {
  const std::vector<std::vector<float>>& raw_outputs =
      TestWithImage(TestDataPath(model_name), TestDataPath(image_name));
  ASSERT_EQ(1, raw_outputs.size());
  std::vector<float> argmax_result;
  if (!model_has_argmax) {
    argmax_result = ApplyArgmax(raw_outputs[0], image_size);
  } else {
    argmax_result = raw_outputs[0];
  }
  ASSERT_EQ(image_size * image_size, argmax_result.size());
  pred_segmentation->resize(argmax_result.size());
  std::copy(argmax_result.begin(), argmax_result.end(),
            pred_segmentation->begin());
  // Read segmentation labels.
  std::vector<uint8_t> groundtruth_segmentation = GetInputFromImage(
      TestDataPath(groundtruth_name), {image_size, image_size, 1});
  ASSERT_EQ(image_size * image_size, groundtruth_segmentation.size());
  // Set contours which are labelled with 255 to 0 to be consistent with VOC2012
  // eval protocol.
  for (int i = 0; i < groundtruth_segmentation.size(); ++i) {
    if (groundtruth_segmentation[i] == 255) {
      groundtruth_segmentation[i] = 0;
    }
  }
  float seg_iou = ComputeIntersectionOverUnion(*pred_segmentation,
                                               groundtruth_segmentation);
  LOG(INFO) << seg_iou;
  EXPECT_GT(seg_iou, iou_threshold);
}

}  // namespace coral
