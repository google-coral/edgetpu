#include "src/cpp/classification/engine.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <tuple>

#include "glog/logging.h"

namespace coral {

// Defines a comparator which allows us to rank ClassificationCandidate based on
// their score and id.
struct ClassificationCandidateComparator {
  bool operator()(const ClassificationCandidate& lhs,
                  const ClassificationCandidate& rhs) const {
    return std::tie(lhs.score, lhs.id) > std::tie(rhs.score, rhs.id);
  }
};

void ClassificationEngine::Validate() {
  std::vector<size_t> output_tensor_sizes = get_all_output_tensors_sizes();
  CHECK_EQ(output_tensor_sizes.size(), 1)
      << "Format error: classification model should have one output tensor "
         "only!";
}

std::vector<ClassificationCandidate>
ClassificationEngine::ClassifyWithInputTensor(const std::vector<uint8_t>& input,
                                              float threshold, int top_k) {
  std::vector<float> scores = RunInference(input)[0];
  std::priority_queue<ClassificationCandidate,
                      std::vector<ClassificationCandidate>,
                      ClassificationCandidateComparator>
      q;
  for (int i = 0; i < scores.size(); ++i) {
    if (scores[i] < threshold) continue;
    q.push(ClassificationCandidate(i, scores[i]));
    if (q.size() > top_k) q.pop();
  }

  std::vector<ClassificationCandidate> ret;
  while (!q.empty()) {
    ret.push_back(q.top());
    q.pop();
  }
  std::reverse(ret.begin(), ret.end());
  return ret;
}

}  // namespace coral
