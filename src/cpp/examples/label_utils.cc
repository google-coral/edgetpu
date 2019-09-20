#include "src/cpp/examples/label_utils.h"

#include <memory>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"

namespace coral {

std::unordered_map<int, std::string> ReadLabelFile(
    const std::string& file_path) {
  std::unordered_map<int, std::string> labels;
  std::ifstream file(file_path.c_str());
  if (file.is_open()) {
    std::string line;
    while (getline(file, line)) {
      absl::RemoveExtraAsciiWhitespace(&line);
      std::vector<std::string> fields =
          absl::StrSplit(line, absl::MaxSplits(' ', 1));
      if (fields.size() == 2) {
        int label_id;
        if (!absl::SimpleAtoi(fields[0], &label_id)) {
          std::cerr << "The label id must be an integer" << std::endl;
          std::abort();
        }
        const std::string& label_name = fields[1];
        labels[label_id] = label_name;
      }
    }
  } else {
    std::cerr << "Cannot open file: " << file_path << std::endl;
    std::abort();
  }
  return labels;
}

}  // namespace coral
