// An example to detect image.
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "src/cpp/detection/engine.h"
#include "src/cpp/examples/label_utils.h"
#include "src/cpp/examples/model_utils.h"
#include "src/cpp/test_utils.h"

ABSL_FLAG(std::string, model_path,
          "./test_data/ssd_mobilenet_v1_fine_tuned_edgetpu.tflite",
          "Path to the tflite model.");

ABSL_FLAG(std::string, image_path, "./test_data/pets.bmp",
          "Path to the image to be classified.");

ABSL_FLAG(std::string, labels_path, "./test_data/pet_labels.txt",
          "Path to the imagenet labels.");

void ObjectDetection(const std::string& model_path, const std::string& image_path,
                   const std::string& labels_path) {
  // Load the model.
  coral::DetectionEngine engine(model_path);
  std::vector<int> input_tensor_shape = engine.get_input_tensor_shape();
  // Read the image.
  std::vector<uint8_t> input_tensor = coral::GetInputFromImage(
      image_path,
      {input_tensor_shape[1], input_tensor_shape[2], input_tensor_shape[3]});
  // Read the label file.
  auto labels = coral::ReadLabelFile(labels_path);

  auto results = engine.DetectWithInputTensor(input_tensor);
  for (const auto& result : results) {
    std::cout << "---------------------------" << std::endl;
    std::cout << "Candidate: " << labels[result.label] << std::endl;
    std::cout << "Score: " << result.score << std::endl;
    std::cout << "Corner: " << result.corners.DebugString() << std::endl;
  }
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  ObjectDetection(absl::GetFlag(FLAGS_model_path),
                absl::GetFlag(FLAGS_image_path),
                absl::GetFlag(FLAGS_labels_path));
}
