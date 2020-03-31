#ifndef EDGETPU_CPP_POSENET_POSENET_DECODER_H_
#define EDGETPU_CPP_POSENET_POSENET_DECODER_H_

#include <ostream>
#include <queue>
#include <vector>

namespace coral {

// An adjacency list representing the directed edges connecting keypoints.
struct AdjacencyList {
  AdjacencyList() = default;
  AdjacencyList(const AdjacencyList&) = default;
  AdjacencyList& operator=(const AdjacencyList&) = default;

  explicit AdjacencyList(const int n_nodes)
      : child_ids(n_nodes), edge_ids(n_nodes) {}

  // child_ids[i] is a vector holding the node ids of all children of the i-th
  // node and edge_ids[i] is a vector holding the edge ids of all edges stemming
  // from the i-th node. If the k-th edge in the graph starts at the i-th node
  // and ends at the j-th node, then child_ids[i] and edge_ids will contain j
  // and k, respectively, at corresponding positions.
  std::vector<std::vector<int>> child_ids;
  std::vector<std::vector<int>> edge_ids;
};

namespace posenet_decoder_op {

static constexpr int kNumKeypoints = 17;

// These 16 edges allow traversing of the pose graph along the mid_offsets (see
// paper for details).
static constexpr int kNumEdges = 16;

struct Point {
  float y;  // all coordinate pairs always use y first.
  float x;
};

struct PoseKeypoints {
  Point keypoint[posenet_decoder_op::kNumKeypoints];
};

struct PoseKeypointScores {
  float keypoint[posenet_decoder_op::kNumKeypoints];
};

// Decodes poses from the score map, the short and mid offsets.
// "Block space" refers to the output y and z size of the network.
// For example if the network that takes a (353,481) (y,x) input image will have
// an output field of 23, 31. Thus the sizes of the input vectors to this
// functions will be
//   scores: 23x31xkNumKeypoints
//   short_offsets 23x31x2*kNumKeypoints (x and y per keypoint)
//   mid_offsets 23x31x2*2*kNumEdges (x and y for each fwd and bwd edge)
// Thus height and width need to be set to 23 and 31 respectively.
// nms_radius must also be given in these units.
// The output coordinates will be in pixel coordinates.
//
// For details see https://arxiv.org/abs/1803.08225
// PersonLab: Person Pose Estimation and Instance Segmentation with a
// Bottom-Up, Part-Based, Geometric Embedding Model
// George Papandreou, Tyler Zhu, Liang-Chieh Chen, Spyros Gidaris,
// Jonathan Tompson, Kevin Murphy

int DecodeAllPoses(
    const float* scores,                    // As logits, not post sigmoid
    const float* short_offsets,             // in block space (not pixels)
    const float* mid_offsets,               // in block space (not pixels)
    int height,                             // in block space (not pixels)
    int width,                              // in block space (not pixels)
    int max_detections,                     // maximum number of poses to detect
    float score_threshold,                  // between 0 and 1
    int mid_short_offset_refinement_steps,  // roughly 1-10
    float nms_radius,  // exclusion radius for keypoint of the same kind
                       // between different poses
    int stride,        // Stride of network - used to rescale keypoints
                       // into pixelspace. Typically stride=16
    PoseKeypoints* pose_keypoints,  // pointer to preallocated buffer of size
                                    // [max_detections*sizeof(PoseKeypoints)]
    PoseKeypointScores*
        pose_keypoint_scores,  // pointer to preallocated buffer
                               // of size
                               // [max_detections*sizeof(PoseKeypointScores)]
    float* pose_scores         // pointer to preallocated buffer of size
                               // [max_detections*sizeof(float)]
);

}  // namespace posenet_decoder_op

// Defines a 2-D keypoint with (x, y) float coordinates and its type id.
struct KeypointWithScore {
  KeypointWithScore(const posenet_decoder_op::Point& _point, const int _id,
                    const float _score)
      : point(_point), id(_id), score(_score) {}
  posenet_decoder_op::Point point;
  int id;
  float score;
  // NOLINTNEXTLINE: clang-diagnostic-unused-function
  friend std::ostream& operator<<(std::ostream& ost,
                                  const KeypointWithScore& keypoint) {
    return ost << keypoint.point.y << ", " << keypoint.point.x << ", "
               << keypoint.id << ", " << keypoint.score;
  }
};

// Defines a comparator which allows us to rank keypoints based on their score.
struct KeypointWithScoreComparator {
  bool operator()(const KeypointWithScore& lhs,
                  const KeypointWithScore& rhs) const {
    return lhs.score < rhs.score;
  }
};

using DecreasingScoreKeypointPriorityQueue =
    std::priority_queue<KeypointWithScore, std::vector<KeypointWithScore>,
                        KeypointWithScoreComparator>;

void DecreasingArgSort(const float* scores, const size_t len,
                       std::vector<int>* indices);

void DecreasingArgSort(const std::vector<float>& scores,
                       std::vector<int>* indices);

float ComputeSquaredDistance(const posenet_decoder_op::Point& a,
                             const posenet_decoder_op::Point& b);

float Sigmoid(const float x);

float Logodds(const float x);

void BuildLinearInterpolation(const float x, const int n, int* x_floor,
                              int* x_ceil, float* x_lerp);

void BuildBilinearInterpolation(const float y, const float x, const int height,
                                const int width, const int num_channels,
                                int* top_left, int* top_right, int* bottom_left,
                                int* bottom_right, float* y_lerp,
                                float* x_lerp);

void SampleTensorAtMultipleChannels(const float* tensor, const int height,
                                    const int width, const int num_channels,
                                    const float y, const float x,
                                    const int* result_channels,
                                    const size_t n_result_channels,
                                    float* result);

float SampleTensorAtSingleChannel(const float* tensor, const int height,
                                  const int width, const int num_channels,
                                  const posenet_decoder_op::Point& point,
                                  const int c);

posenet_decoder_op::Point FindDisplacedPosition(
    const float* short_offsets, const float* mid_offsets, const int height,
    const int width, const int num_keypoints, const int num_edges,
    const posenet_decoder_op::Point& source, const int edge_id,
    const int target_id, const int mid_short_offset_refinement_steps);

AdjacencyList BuildAdjacencyList();

void BacktrackDecodePose(
    const float* scores, const float* short_offsets, const float* mid_offsets,
    const int height, const int width, const int num_keypoints,
    const int num_edges, const KeypointWithScore& root,
    const AdjacencyList& adjacency_list,
    const int mid_short_offset_refinement_steps,
    posenet_decoder_op::PoseKeypoints* pose_keypoints,
    posenet_decoder_op::PoseKeypointScores* keypoint_scores);

void BuildKeypointWithScoreQueue(const float* scores,
                                 const float* short_offsets, const int height,
                                 const int width, const int num_keypoints,
                                 const float score_threshold,
                                 const int local_maximum_radius,
                                 DecreasingScoreKeypointPriorityQueue* queue);

bool PassKeypointNMS(const posenet_decoder_op::PoseKeypoints* poses,
                     const size_t n_poses, const KeypointWithScore& keypoint,
                     const float squared_nms_radius);

void FindOverlappingKeypoints(const posenet_decoder_op::PoseKeypoints& pose1,
                              const posenet_decoder_op::PoseKeypoints& pose2,
                              const float squared_radius,
                              std::vector<bool>* mask);

void PerformSoftKeypointNMS(
    const std::vector<int>& decreasing_indices,
    const posenet_decoder_op::PoseKeypoints* all_keypoint_coords,
    const posenet_decoder_op::PoseKeypointScores* all_keypoint_scores,
    const int num_keypoints, const float squared_nms_radius, const int topk,
    std::vector<float>* all_instance_scores);

}  // namespace coral

#endif  // EDGETPU_CPP_POSENET_POSENET_DECODER_H_
