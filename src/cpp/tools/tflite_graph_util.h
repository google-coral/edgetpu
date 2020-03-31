#ifndef EDGETPU_CPP_TOOLS_TFLITE_GRAPH_UTIL_H_
#define EDGETPU_CPP_TOOLS_TFLITE_GRAPH_UTIL_H_

// Utility library for tflite graph tooling related functions.

#include <string>
#include <vector>

#include "src/cpp/utils.h"

namespace coral {
namespace tools {

// Concatenates two tflite models into one, assuming each input model has
// only one subgraph.
// Optional Args:
//   bypass_output_tensors: A list of output tensor names from model0, which
//                          should also become output tensors in the merged
//                          graph (i.e. skip model1). By default any output
//                          tensors of model0 which are not input tensors for
//                          model1 become dead ends.
void ConcatTfliteModels(
    const std::string& model0_path, const std::string& model1_path,
    const std::string& output_path,
    const std::vector<std::string>& bypass_output_tensors = {});

}  // namespace tools
}  // namespace coral

#endif  // EDGETPU_CPP_TOOLS_TFLITE_GRAPH_UTIL_H_
