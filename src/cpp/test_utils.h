#ifndef EDGETPU_CPP_TEST_UTILS_H_
#define EDGETPU_CPP_TEST_UTILS_H_

#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "src/cpp/bbox_utils.h"

namespace coral {

// Defines dimension of an image, in height, width, depth order.
typedef std::array<int, 3> ImageDims;

enum CnnProcessorType { kEdgeTpu, kCpu };

enum CompilationType { kCoCompilation, kSingleCompilation };

// Retrieves test file path with file name.
std::string TestDataPath(const std::string& name);

// Generates a 1-d uint8 array with given size.
std::vector<uint8_t> GetRandomInput(int n);

// Generates a 1-d uint8 array with given input tensor shape.
std::vector<uint8_t> GetRandomInput(std::vector<int> shape);

// Gets input from images and resizes to `target_dims`. It will crash upon
// failure.
std::vector<uint8_t> GetInputFromImage(const std::string& image_path,
                                       const ImageDims& target_dims);

// Gets input from images as unsigned char arrays and resizes to `target_dims`. 
// It will crash upon failure.
std::vector<uint8_t> GetInputFromImageData(const uint8_t *data, const ImageDims &image_dims,
                                           const ImageDims &target_dims);

// Gets list of all models.
std::vector<std::string> GetAllModels();

// Tests model with random input. Ensures it's runnable.
void TestWithRandomInput(const std::string& model_path);

// Generate a temp tflite file name under /tmp/ folder.
std::string GenerateRandomFilePath(const std::string& prefix,
                                   const std::string& suffix);

// Tests model with a real image.
std::vector<std::vector<float>> TestWithImage(const std::string& model_path,
                                              const std::string& image_path);

// Returns top-k predictions as label-score pairs.
std::vector<std::pair<int, float>> GetTopK(const std::vector<float>& scores,
                                           float threshold, int top_k);

// Returns whether top k results contains a given label.
bool TopKContains(const std::vector<std::pair<int, float>>& topk, int label);

// Tests a classification model with customized preprocessing.
// Custom preprocessing is done by:
// (v - (mean - zero_point * scale * stddev)) / (stddev * scale)
// where zero_point and scale are the quantization parameters of the input
// tensor, and mean and stddev are the normalization parameters of the input
// tensor. Effective mean and scale should be
// (mean - zero_point * scale * stddev) and (stddev * scale) respectively.
// If rgb2bgr is true, the channels of input image will be shuffled from
// RGB to BGR.
void TestClassification(const std::string& model_path,
                        const std::string& image_path, float effective_scale,
                        const std::vector<float>& effective_means, bool rgb2bgr,
                        float score_threshold, int k, int expected_topk_label);

// Tests a classification model, default not to shuffle the RGB channels.
void TestClassification(const std::string& model_path,
                        const std::string& image_path, float effective_scale,
                        const std::vector<float>& effective_means,
                        float score_threshold, int k, int expected_topk_label);

// Tests a classification model.
void TestClassification(const std::string& model_path,
                        const std::string& image_path, float score_threshold,
                        int k, int expected_topk_label);

// Tests a classification model. Only checks the top1 result.
void TestClassification(const std::string& model_path,
                        const std::string& image_path, float score_threshold,
                        int expected_top1_label);

// Tests a SSD detection model. Only checks the first detection result.
void TestDetection(const std::string& model_path, const std::string& image_path,
                   const BoxCornerEncoding& expected_box, int expected_label,
                   float score_threshold, float iou_threshold);

// Tests a MSCOCO detection model with cat.bmp.
void TestCatMsCocoDetection(const std::string& model_path,
                            float score_threshold, float iou_threshold);

void BenchmarkModelOnEdgeTpu(const std::string& model_path,
                             benchmark::State& state);

// Benchmarks models on a sinlge EdgeTpu device.
void BenchmarkModelsOnEdgeTpu(const std::vector<std::string>& model_paths,
                              benchmark::State& state);

// This test will run inference with fixed randomly generated input for multiple
// times and ensure the inference result are constant.
//  - model_path: string, path to the FlatBuffer file.
//  - runs: number of iterations.
void RepeatabilityTest(const std::string& model_path, int runs);

// This test will run inference with given model for multiple times. Input are
// generated randomly and the result won't be checked.
//  - model_path: string, path of the FlatBuffer file.
//  - runs: number of iterations.
//  - sleep_sec: time interval between inferences. By default it's zero.
void InferenceStressTest(const std::string& model_path, int runs,
                         int sleep_sec = 0);

// Computes the iou of two masks.
float ComputeIntersectionOverUnion(const std::vector<uint8_t>& mask1,
                                   const std::vector<uint8_t>& mask2);

// Tests segmentation models that include ArgMax operator, returns the
// prediction results.
void TestSegmentationWithArgmax(const std::string& model_name,
                                const std::string& image_name,
                                const std::string& seg_name, int size,
                                float iou_threshold,
                                std::vector<uint8_t>* pred_segmentation);

}  // namespace coral

#endif  // EDGETPU_CPP_TEST_UTILS_H_
