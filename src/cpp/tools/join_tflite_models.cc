// Tool to join two tflite models. The models may contain custom operator,
// which can not be imported / exported properly by tflite/toco yet.
//
// Two models can be joined together only if the output tensors of
// input_graph_base are the input tensors of input_graph_head. All other tensors
// should have identical names.

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "src/cpp/tools/tflite_graph_util.h"

ABSL_FLAG(std::string, input_graph_base, "",
          "Path to the base input graph. Must be in tflite format.");

ABSL_FLAG(std::string, input_graph_head, "",
          "Path to the head input graph. Must be in tflite format.");

ABSL_FLAG(std::string, output_graph, "",
          "Path to the output graph. Output graph will be in tflite format.");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  coral::tools::ConcatTfliteModels(absl::GetFlag(FLAGS_input_graph_base),
                                   absl::GetFlag(FLAGS_input_graph_head),
                                   absl::GetFlag(FLAGS_output_graph));
}
