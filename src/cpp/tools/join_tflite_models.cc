// Tool to join two tflite models. The models may contain custom operator,
// which can not be imported / exported properly by tflite/toco yet.
//
// Two models can be joined together only if all the input tensors of
// input_graph_head are present as output tensors of input_graph_base.
// Any additional output tensors of input_graph_base will become dead ends,
// unless specified with --bypass_tensors, in which case they will be routed
// to the end as output_tensors of the final graph.
// Note also that there should be no name collisions between other tensors of
// the two input graphs.

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_split.h"
#include "src/cpp/tools/tflite_graph_util.h"

ABSL_FLAG(std::string, input_graph_base, "",
          "Path to the base input graph. Must be in tflite format.");

ABSL_FLAG(std::string, input_graph_head, "",
          "Path to the head input graph. Must be in tflite format.");

ABSL_FLAG(std::string, output_graph, "",
          "Path to the output graph. Output graph will be in tflite format.");

ABSL_FLAG(std::string, bypass_output_tensors, "",
          "A list of output tensor names from base input graph, which "
          "should also become output tensors in the merged output graph.");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  coral::tools::ConcatTfliteModels(
      absl::GetFlag(FLAGS_input_graph_base),
      absl::GetFlag(FLAGS_input_graph_head), absl::GetFlag(FLAGS_output_graph),
      absl::StrSplit(absl::GetFlag(FLAGS_bypass_output_tensors), ',',
                     absl::SkipEmpty()));
}
