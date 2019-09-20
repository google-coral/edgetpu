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
void ConcatTfliteModels(const std::string& model0_path,
                        const std::string& model1_path,
                        const std::string& output_path);

}  // namespace tools
}  // namespace coral

#endif  // EDGETPU_CPP_TOOLS_TFLITE_GRAPH_UTIL_H_
