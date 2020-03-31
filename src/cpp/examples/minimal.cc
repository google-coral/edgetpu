// Example to run a model using one Edge TPU.
// It depends only on tflite and edgetpu.h

#include <algorithm>
#include <chrono>  // NOLINT
#include <iostream>
#include <memory>
#include <ostream>
#include <string>

#include "edgetpu.h"
#include "src/cpp/examples/model_utils.h"
#include "src/cpp/test_utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

std::vector<uint8_t> decode_bmp(const uint8_t* input, int row_size, int width,
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
          std::cerr << "Unexpected number of channels: " << channels
                    << std::endl;
          std::abort();
          break;
      }
    }
  }
  return output;
}

std::vector<uint8_t> read_bmp(const std::string& input_bmp_name, int* width,
                              int* height, int* channels) {
  int begin, end;

  std::ifstream file(input_bmp_name, std::ios::in | std::ios::binary);
  if (!file) {
    std::cerr << "input file " << input_bmp_name << " not found\n";
    std::abort();
  }

  begin = file.tellg();
  file.seekg(0, std::ios::end);
  end = file.tellg();
  size_t len = end - begin;

  std::vector<uint8_t> img_bytes(len);
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char*>(img_bytes.data()), len);
  const int32_t header_size =
      *(reinterpret_cast<const int32_t*>(img_bytes.data() + 10));
  *width = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 18));
  *height = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 22));
  const int32_t bpp =
      *(reinterpret_cast<const int32_t*>(img_bytes.data() + 28));
  *channels = bpp / 8;

  // there may be padding bytes when the width is not a multiple of 4 bytes
  // 8 * channels == bits per pixel
  const int row_size = (8 * *channels * *width + 31) / 32 * 4;

  // if height is negative, data layout is top down
  // otherwise, it's bottom up
  bool top_down = (*height < 0);

  // Decode image, allocating tensor once the image size is known
  const uint8_t* bmp_pixels = &img_bytes[header_size];
  return decode_bmp(bmp_pixels, row_size, *width, abs(*height), *channels,
                    top_down);
}

int main(int argc, char* argv[]) {
  if (argc != 1 && argc != 3) {
    std::cout << " minimal <edgetpu model> <input resized image>" << std::endl;
    return 1;
  }

  // Modify the following accordingly to try different models and images.
  const std::string model_path =
      argc == 3 ? argv[1]
                : coral::GetTempPrefix() +
                      "/mobilenet_v1_1.0_224_quant_edgetpu.tflite";
  const std::string resized_image_path =
      argc == 3 ? argv[2] : coral::GetTempPrefix() + "/resized_cat.bmp";

  // Read model.
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  if (model == nullptr) {
    std::cerr << "Fail to build FlatBufferModel from file: " << model_path
              << std::endl;
    std::abort();
  }

  // Build interpreter.
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context =
      edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  std::unique_ptr<tflite::Interpreter> interpreter =
      coral::BuildEdgeTpuInterpreter(*model, edgetpu_context.get());

  // Read the resized image file.
  int width, height, channels;
  const std::vector<uint8_t>& input =
      read_bmp(resized_image_path, &width, &height, &channels);

  const auto& required_shape = coral::GetInputShape(*interpreter, 0);
  if (height != required_shape[0] || width != required_shape[1] ||
      channels != required_shape[2]) {
    std::cerr << "Input size mismatches: "
              << "width: " << width << " vs " << required_shape[0]
              << ", height: " << height << " vs " << required_shape[1]
              << ", channels: " << channels << " vs " << required_shape[2]
              << std::endl;
    std::abort();
  }
  // Print inference result.
  const auto& result = coral::RunInference(input, interpreter.get());
  auto it = std::max_element(result.begin(), result.end());
  std::cout << "[Image analysis] max value index: "
            << std::distance(result.begin(), it) << " value: " << *it
            << std::endl;
  return 0;
}
