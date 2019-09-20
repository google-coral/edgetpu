#include "absl/flags/parse.h"
#include "benchmark/benchmark.h"
#include "glog/logging.h"
#include "src/cpp/test_utils.h"

namespace coral {

template <CnnProcessorType CnnProcessor, int ysize, int xsize>
static void BM_PoseNet_MobileNetV1_075_WithDecoder(benchmark::State& state) {
  const std::string model_prefix = "posenet_mobilenet_v1_075_" +
                                   std::to_string(ysize) + "_" +
                                   std::to_string(xsize);
  const std::string model_name =
      CnnProcessor == kEdgeTpu ? model_prefix + "_quant_decoder_edgetpu.tflite"
                               : model_prefix + "_quant_decoder.tflite";
  coral::BenchmarkModelOnEdgeTpu(TestDataPath("posenet/" + model_name), state);
}
BENCHMARK_TEMPLATE(BM_PoseNet_MobileNetV1_075_WithDecoder, kEdgeTpu, 353, 481);
BENCHMARK_TEMPLATE(BM_PoseNet_MobileNetV1_075_WithDecoder, kCpu, 353, 481);
BENCHMARK_TEMPLATE(BM_PoseNet_MobileNetV1_075_WithDecoder, kEdgeTpu, 481, 641);
BENCHMARK_TEMPLATE(BM_PoseNet_MobileNetV1_075_WithDecoder, kCpu, 481, 641);
BENCHMARK_TEMPLATE(BM_PoseNet_MobileNetV1_075_WithDecoder, kEdgeTpu, 721, 1281);
BENCHMARK_TEMPLATE(BM_PoseNet_MobileNetV1_075_WithDecoder, kCpu, 721, 1281);

}  // namespace coral

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
