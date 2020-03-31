#include "src/cpp/detection/engine.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <tuple>

#include "glog/logging.h"

namespace coral {

void DetectionEngine::Validate() {
  std::vector<size_t> output_tensor_sizes = get_all_output_tensors_sizes();
  CHECK_EQ(output_tensor_sizes.size(), 4)
      << "Format error: detection model should have 4 output tensors!";
  // The tensors are <bounding boxes, label ids, scores, number of predictions>.
  CHECK_EQ(output_tensor_sizes[0], output_tensor_sizes[1] * 4);
  CHECK_EQ(output_tensor_sizes[0], output_tensor_sizes[2] * 4);
  CHECK_EQ(output_tensor_sizes[3], 1);
}

std::vector<DetectionCandidate> DetectionEngine::DetectWithInputTensor(
    const std::vector<uint8_t>& input, float threshold, int top_k) {
  std::vector<std::vector<float>> output = RunInference(input);
  int n = lround(output[3][0]);

  std::priority_queue<DetectionCandidate, std::vector<DetectionCandidate>,
                      DetectionCandidateComparator>
      q;

  for (int i = 0; i < n; ++i) {
    int id = lround(output[1][i]);
    float score = output[2][i];
    if (score < threshold) continue;
    float y1 = std::max(static_cast<float>(0.0), output[0][4 * i]);
    float x1 = std::max(static_cast<float>(0.0), output[0][4 * i + 1]);
    float y2 = std::min(static_cast<float>(1.0), output[0][4 * i + 2]);
    float x2 = std::min(static_cast<float>(1.0), output[0][4 * i + 3]);
    q.push(
        DetectionCandidate({BoxCornerEncoding({x1, y1, x2, y2}), id, score}));
    if (q.size() > top_k) q.pop();
  }

  std::vector<DetectionCandidate> ret;
  while (!q.empty()) {
    ret.push_back(q.top());
    q.pop();
  }
  std::reverse(ret.begin(), ret.end());
  return ret;
}

}  // namespace coral
