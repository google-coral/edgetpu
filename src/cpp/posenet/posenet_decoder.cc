#include "src/cpp/posenet/posenet_decoder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

namespace coral {

using posenet_decoder_op::kNumKeypoints;
using posenet_decoder_op::Point;
using posenet_decoder_op::PoseKeypoints;
using posenet_decoder_op::PoseKeypointScores;

enum KeypointType {
  kNose,
  kLeftEye,
  kRightEye,
  kLeftEar,
  kRightEar,
  kLeftShoulder,
  kRightShoulder,
  kLeftElbow,
  kRightElbow,
  kLeftWrist,
  kRightWrist,
  kLeftHip,
  kRightHip,
  kLeftKnee,
  kRightKnee,
  kLeftAnkle,
  kRightAnkle
};

const std::array<std::pair<KeypointType, KeypointType>, 32> kEdgeList = {{

    // Forward edges
    {kNose, kLeftEye},
    {kLeftEye, kLeftEar},
    {kNose, kRightEye},
    {kRightEye, kRightEar},
    {kNose, kLeftShoulder},
    {kLeftShoulder, kLeftElbow},
    {kLeftElbow, kLeftWrist},
    {kLeftShoulder, kLeftHip},
    {kLeftHip, kLeftKnee},
    {kLeftKnee, kLeftAnkle},
    {kNose, kRightShoulder},
    {kRightShoulder, kRightElbow},
    {kRightElbow, kRightWrist},
    {kRightShoulder, kRightHip},
    {kRightHip, kRightKnee},
    {kRightKnee, kRightAnkle},

    // Backward edges
    {kLeftEye, kNose},
    {kLeftEar, kLeftEye},
    {kRightEye, kNose},
    {kRightEar, kRightEye},
    {kLeftShoulder, kNose},
    {kLeftElbow, kLeftShoulder},
    {kLeftWrist, kLeftElbow},
    {kLeftHip, kLeftShoulder},
    {kLeftKnee, kLeftHip},
    {kLeftAnkle, kLeftKnee},
    {kRightShoulder, kNose},
    {kRightElbow, kRightShoulder},
    {kRightWrist, kRightElbow},
    {kRightHip, kRightShoulder},
    {kRightKnee, kRightHip},
    {kRightAnkle, kRightKnee}}};

template <typename T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
  return v < lo ? lo : hi < v ? hi : v;
}

// Finds the indices of the scores if we sort them in decreasing order.
void DecreasingArgSort(const float* scores, const size_t len,
                       std::vector<int>* indices) {
  indices->resize(len);
  std::iota(indices->begin(), indices->end(), 0);
  std::sort(
      indices->begin(), indices->end(),
      [&scores](const int i, const int j) { return scores[i] > scores[j]; });
}

void DecreasingArgSort(const std::vector<float>& scores,
                       std::vector<int>* indices) {
  DecreasingArgSort(scores.data(), scores.size(), indices);
}
// Computes the squared distance between a pair of 2-D points.
float ComputeSquaredDistance(const Point& a, const Point& b) {
  const float dy = b.y - a.y;
  const float dx = b.x - a.x;
  return dy * dy + dx * dx;
}

// Computes the sigmoid of the input. The output is in (0, 1).
float Sigmoid(const float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Inverse of the sigmoid, computes log odds from a probability.
float Logodds(const float x) { return -std::log(1.0f / (x + 1E-6) - 1.0f); }

// Helper function for 1-D linear interpolation. It computes the floor and the
// ceiling of the input coordinate, as well as the weighting factor between the
// two interpolation endpoints, such that:
// y = (1 - x_lerp) * vec[x_floor] + x_lerp * vec[x_ceil]
void BuildLinearInterpolation(const float x, const int n, int* x_floor,
                              int* x_ceil, float* x_lerp) {
  const float x_proj = clamp(x, 0.0f, n - 1.0f);
  *x_floor = static_cast<int>(floorf(x_proj));
  *x_ceil = static_cast<int>(ceilf(x_proj));
  *x_lerp = x - (*x_floor);
}

// Helper function for 2-D bilinear interpolation. It computes the four corners
// of the 2x2 cell that contain the input coordinates (x, y), as well as the
// weighting factor between the four interpolation endpoints, such that:
// y =
//   (1 - y_lerp) * ((1 - x_lerp) * vec[top_left] + x_lerp * vec[top_right]) +
//   y_lerp * ((1 - x_lerp) * tensor[bottom_left] + x_lerp * vec[bottom_right])
void BuildBilinearInterpolation(const float y, const float x, const int height,
                                const int width, const int num_channels,
                                int* top_left, int* top_right, int* bottom_left,
                                int* bottom_right, float* y_lerp,
                                float* x_lerp) {
  int y_floor;
  int y_ceil;
  BuildLinearInterpolation(y, height, &y_floor, &y_ceil, y_lerp);
  int x_floor;
  int x_ceil;
  BuildLinearInterpolation(x, width, &x_floor, &x_ceil, x_lerp);
  *top_left = (y_floor * width + x_floor) * num_channels;
  *top_right = (y_floor * width + x_ceil) * num_channels;
  *bottom_left = (y_ceil * width + x_floor) * num_channels;
  *bottom_right = (y_ceil * width + x_ceil) * num_channels;
}

// Sample the input tensor values at position (x, y) and at multiple channels.
// The input tensor has shape [height, width, num_channels]. We bilinearly
// sample its value at tensor(y, x, c), for c in the channels specified. This
// is faster than calling the single channel interpolation function multiple
// times because the computation of the positions needs to be done only once.
void SampleTensorAtMultipleChannels(const float* tensor, const int height,
                                    const int width, const int num_channels,
                                    const float y, const float x,
                                    const int* result_channels,
                                    const size_t n_result_channels,
                                    float* result) {
  int top_left;
  int top_right;
  int bottom_left;
  int bottom_right;
  float y_lerp;
  float x_lerp;
  BuildBilinearInterpolation(y, x, height, width, num_channels, &top_left,
                             &top_right, &bottom_left, &bottom_right, &y_lerp,
                             &x_lerp);
  for (int i = 0; i < n_result_channels; ++i) {
    const int c = result_channels[i];
    result[i] = (1 - y_lerp) * ((1 - x_lerp) * tensor[top_left + c] +
                                x_lerp * tensor[top_right + c]) +
                y_lerp * ((1 - x_lerp) * tensor[bottom_left + c] +
                          x_lerp * tensor[bottom_right + c]);
  }
}

// Sample the input tensor values at position (x, y) and at a single channel.
// The input tensor has shape [height, width, num_channels]. We bilinearly
// sample its value at tensor(y, x, channel).
float SampleTensorAtSingleChannel(const float* tensor, const int height,
                                  const int width, const int num_channels,
                                  const Point& point, const int c) {
  float result;
  SampleTensorAtMultipleChannels(tensor, height, width, num_channels, point.y,
                                 point.x, &c, 1, &result);
  return result;
}

// Follows the mid-range offsets, and then refines the position by the short-
// range offsets for a fixed number of steps.
Point FindDisplacedPosition(const float* short_offsets,
                            const float* mid_offsets, const int height,
                            const int width, const int num_keypoints,
                            const int num_edges, const Point& source,
                            const int edge_id, const int target_id,
                            const int mid_short_offset_refinement_steps) {
  float y = source.y;
  float x = source.x;
  float offsets[2];
  // Follow the mid-range offsets.
  int channels[] = {edge_id, num_edges + edge_id};
  const int n_channels = 2;
  // Total size of mid_offsets is height x width x 2*2*num_edges
  SampleTensorAtMultipleChannels(mid_offsets, height, width, 2 * 2 * num_edges,
                                 y, x, channels, n_channels, &offsets[0]);
  y = clamp(y + offsets[0], 0.0f, height - 1.0f);
  x = clamp(x + offsets[1], 0.0f, width - 1.0f);
  // Refine by the short-range offsets.
  channels[0] = target_id;
  channels[1] = num_keypoints + target_id;
  for (int i = 0; i < mid_short_offset_refinement_steps; ++i) {
    SampleTensorAtMultipleChannels(short_offsets, height, width,
                                   2 * num_keypoints, y, x, channels,
                                   n_channels, &offsets[0]);
    y = clamp(y + offsets[0], 0.0f, height - 1.0f);
    x = clamp(x + offsets[1], 0.0f, width - 1.0f);
  }
  return Point{y, x};
}

// Build an adjacency list of the pose graph.
AdjacencyList BuildAdjacencyList() {
  AdjacencyList adjacency_list(posenet_decoder_op::kNumKeypoints);
  for (int k = 0; k < kEdgeList.size(); ++k) {
    const int parent_id = kEdgeList[k].first;
    const int child_id = kEdgeList[k].second;
    adjacency_list.child_ids[parent_id].push_back(child_id);
    adjacency_list.edge_ids[parent_id].push_back(k);
  }
  return adjacency_list;
}

void BacktrackDecodePose(const float* scores, const float* short_offsets,
                         const float* mid_offsets, const int height,
                         const int width, const int num_keypoints,
                         const int num_edges, const KeypointWithScore& root,
                         const AdjacencyList& adjacency_list,
                         const int mid_short_offset_refinement_steps,
                         PoseKeypoints* pose_keypoints,
                         PoseKeypointScores* keypoint_scores) {
  const float root_score = SampleTensorAtSingleChannel(
      scores, height, width, num_keypoints, root.point, root.id);

  // Used in order to put candidate keypoints in a priority queue w.r.t. their
  // score. Keypoints with higher score have higher priority and will be
  // decoded/processed first.
  DecreasingScoreKeypointPriorityQueue decode_queue;
  decode_queue.push(KeypointWithScore(root.point, root.id, root_score));

  // Keeps track of the keypoints whose position has already been decoded.
  std::vector<bool> keypoint_decoded(num_keypoints, false);

  while (!decode_queue.empty()) {
    // The top element in the queue is the next keypoint to be processed.
    const KeypointWithScore current_keypoint = decode_queue.top();
    decode_queue.pop();

    if (keypoint_decoded[current_keypoint.id]) continue;

    pose_keypoints->keypoint[current_keypoint.id] = current_keypoint.point;
    keypoint_scores->keypoint[current_keypoint.id] = current_keypoint.score;

    keypoint_decoded[current_keypoint.id] = true;

    // Add the children of the current keypoint that have not been decoded yet
    // to the priority queue.
    const int num_children =
        adjacency_list.child_ids[current_keypoint.id].size();
    for (int j = 0; j < num_children; ++j) {
      const int child_id = adjacency_list.child_ids[current_keypoint.id][j];
      int edge_id = adjacency_list.edge_ids[current_keypoint.id][j];
      if (keypoint_decoded[child_id]) continue;

      // The mid-offsets block is organized as 4 blocks of kNumEdges:
      // [fwd Y offsets][fwd X offsets][bwd Y offsets][bwd X offsets]
      // OTOH edge_id is [0,kNumEdges) for forward edges and
      // [kNumEdges, 2*kNumEdges) for backward edges.
      // Thus if the edge is a backward edge (>kNumEdges) then we need
      // to start 16 indices later to be correctly aligned with the mid-offsets.
      if (edge_id > posenet_decoder_op::kNumEdges) {
        edge_id += posenet_decoder_op::kNumEdges;
      }

      const Point child_point = FindDisplacedPosition(
          short_offsets, mid_offsets, height, width, num_keypoints, num_edges,
          current_keypoint.point, edge_id, child_id,
          mid_short_offset_refinement_steps);

      const float child_score = SampleTensorAtSingleChannel(
          scores, height, width, num_keypoints, child_point, child_id);

      decode_queue.emplace(child_point, child_id, child_score);
    }
  }
}

void BuildKeypointWithScoreQueue(const float* scores,
                                 const float* short_offsets, const int height,
                                 const int width, const int num_keypoints,
                                 const float score_threshold,
                                 const int local_maximum_radius,
                                 DecreasingScoreKeypointPriorityQueue* queue) {
  int score_index = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int offset_index = 2 * score_index;
      for (int j = 0; j < num_keypoints; ++j) {
        const float score = scores[score_index];
        if (score >= score_threshold) {
          // Only consider keypoints whose score is maximum in a local window.
          bool local_maximum = true;
          const int y_start = std::max(y - local_maximum_radius, 0);
          const int y_end = std::min(y + local_maximum_radius + 1, height);
          for (int y_current = y_start; y_current < y_end; ++y_current) {
            const int x_start = std::max(x - local_maximum_radius, 0);
            const int x_end = std::min(x + local_maximum_radius + 1, width);
            for (int x_current = x_start; x_current < x_end; ++x_current) {
              if (scores[y_current * width * num_keypoints +
                         x_current * num_keypoints + j] > score) {
                local_maximum = false;
                break;
              }
            }
            if (!local_maximum) break;
          }
          if (local_maximum) {
            const float dy = short_offsets[offset_index];
            const float dx = short_offsets[offset_index + num_keypoints];
            const float y_refined = clamp(y + dy, 0.0f, height - 1.0f);
            const float x_refined = clamp(x + dx, 0.0f, width - 1.0f);
            queue->emplace(Point{y_refined, x_refined}, j, score);
          }
        }

        ++score_index;
        ++offset_index;
      }
    }
  }
}

bool PassKeypointNMS(const PoseKeypoints* poses, const size_t n_poses,
                     const KeypointWithScore& keypoint,
                     const float squared_nms_radius) {
  for (size_t index = 0; index < n_poses; ++index) {
    if (ComputeSquaredDistance(keypoint.point,
                               poses[index].keypoint[keypoint.id]) <=
        squared_nms_radius) {
      return false;
    }
  }
  return true;
}

void FindOverlappingKeypoints(const PoseKeypoints& pose1,
                              const PoseKeypoints& pose2,
                              const float squared_radius,
                              std::vector<bool>* mask) {
  const int num_keypoints = mask->size();
  for (int k = 0; k < num_keypoints; ++k) {
    if (ComputeSquaredDistance(pose1.keypoint[k], pose2.keypoint[k]) <=
        squared_radius) {
      (*mask)[k] = true;
    }
  }
}

void PerformSoftKeypointNMS(const std::vector<int>& decreasing_indices,
                            const PoseKeypoints* all_keypoint_coords,
                            const PoseKeypointScores* all_keypoint_scores,
                            const int num_keypoints,
                            const float squared_nms_radius, const int topk,
                            std::vector<float>* all_instance_scores) {
  const int num_instances = decreasing_indices.size();
  all_instance_scores->resize(num_instances);
  // Indicates the occlusion status of the keypoints of the active instance.
  std::vector<bool> keypoint_occluded(num_keypoints);
  // Indices of the keypoints of the active instance in decreasing score value.
  std::vector<int> indices(num_keypoints);
  for (int i = 0; i < num_instances; ++i) {
    const int current_index = decreasing_indices[i];
    // Find the keypoints of the current instance which are overlapping with
    // the corresponding keypoints of the higher-scoring instances and
    // zero-out their contribution to the score of the current instance.
    std::fill(keypoint_occluded.begin(), keypoint_occluded.end(), false);
    for (int j = 0; j < i; ++j) {
      const int previous_index = decreasing_indices[j];
      FindOverlappingKeypoints(all_keypoint_coords[current_index],
                               all_keypoint_coords[previous_index],
                               squared_nms_radius, &keypoint_occluded);
    }
    // We compute the argsort keypoint indices based on the original keypoint
    // scores, but we do not let them contribute to the instance score if they
    // have been non-maximum suppressed.
    DecreasingArgSort(&all_keypoint_scores[current_index].keypoint[0],
                      num_keypoints, &indices);
    float total_score = 0.0f;
    for (int k = 0; k < topk; ++k) {
      if (!keypoint_occluded[indices[k]]) {
        total_score += all_keypoint_scores[current_index].keypoint[indices[k]];
      }
    }
    (*all_instance_scores)[current_index] = total_score / topk;
  }
}

namespace posenet_decoder_op {

int DecodeAllPoses(const float* scores, const float* short_offsets,
                   const float* mid_offsets, const int height, const int width,
                   const int max_detections, const float score_threshold,
                   const int mid_short_offset_refinement_steps,
                   const float nms_radius, const int stride,
                   PoseKeypoints* pose_keypoints,
                   PoseKeypointScores* pose_keypoint_scores,
                   float* pose_scores) {
  static const int kLocalMaximumRadius = 1;

  // score_threshold threshold as a logit, before sigmoid
  const float min_score_logit = Logodds(score_threshold);

  DecreasingScoreKeypointPriorityQueue queue;
  BuildKeypointWithScoreQueue(scores, short_offsets, height, width,
                              kNumKeypoints, min_score_logit,
                              kLocalMaximumRadius, &queue);
  AdjacencyList adjacency_list = BuildAdjacencyList();

  const int topk = kNumKeypoints;
  std::vector<int> indices(kNumKeypoints);

  int pose_counter = 0;

  // Generate at most max_detections object instances per image in decreasing
  // root part score order.
  std::vector<float> all_instance_scores;

  std::vector<PoseKeypoints> scratch_poses(max_detections);
  std::vector<PoseKeypointScores> scratch_keypoint_scores(max_detections);

  while (pose_counter < max_detections && !queue.empty()) {
    // The top element in the queue is the next root candidate.
    const KeypointWithScore root = queue.top();
    queue.pop();

    // Reject a root candidate if it is within a disk of `nms_radius` pixels
    // from the corresponding part of a previously detected instance.
    if (!PassKeypointNMS(scratch_poses.data(), pose_counter, root,
                         nms_radius * nms_radius)) {
      continue;
    }

    auto next_pose = &scratch_poses[pose_counter];
    auto next_scores = &scratch_keypoint_scores[pose_counter];
    for (int k = 0; k < kNumKeypoints; ++k) {
      next_pose->keypoint[k].x = -1.0f;
      next_pose->keypoint[k].y = -1.0f;
      next_scores->keypoint[k] = -1E5;
    }
    BacktrackDecodePose(scores, short_offsets, mid_offsets, height, width,
                        kNumKeypoints, kNumEdges, root, adjacency_list,
                        mid_short_offset_refinement_steps, next_pose,
                        next_scores);

    // Convert keypoint-level scores from log-odds to probabilities and compute
    // an initial instance-level score as the average of the scores of the top-k
    // scoring keypoints.
    for (int k = 0; k < kNumKeypoints; ++k) {
      next_scores->keypoint[k] = Sigmoid(next_scores->keypoint[k]);
    }
    DecreasingArgSort(&next_scores->keypoint[0], kNumKeypoints, &indices);
    float instance_score = 0.0f;
    for (int j = 0; j < topk; ++j) {
      instance_score += next_scores->keypoint[indices[j]];
    }
    instance_score /= topk;

    if (instance_score >= score_threshold) {
      pose_counter++;
      all_instance_scores.push_back(instance_score);
    }
  }

  // Sort the detections in decreasing order of their instance-level scores.
  std::vector<int> decreasing_indices;
  DecreasingArgSort(all_instance_scores, &decreasing_indices);

  // Keypoint-level soft non-maximum suppression and instance-level rescoring as
  // the average of the top-k keypoints in terms of their keypoint-level scores.
  PerformSoftKeypointNMS(decreasing_indices, scratch_poses.data(),
                         scratch_keypoint_scores.data(), kNumKeypoints,
                         nms_radius * nms_radius, topk, &all_instance_scores);

  // Sort the detections in decreasing order of their final instance-level
  // scores. Usually the order does not change but this is not guaranteed.
  DecreasingArgSort(all_instance_scores, &decreasing_indices);

  pose_counter = 0;
  for (size_t index : decreasing_indices) {
    if (all_instance_scores[index] < score_threshold) {
      break;
    }
    // Rescale keypoint coordinates into pixel space (much more useful for
    // user).
    for (int k = 0; k < kNumKeypoints; ++k) {
      pose_keypoints[pose_counter].keypoint[k].y =
          scratch_poses[index].keypoint[k].y * stride;
      pose_keypoints[pose_counter].keypoint[k].x =
          scratch_poses[index].keypoint[k].x * stride;
    }

    memcpy(&pose_keypoint_scores[pose_counter], &scratch_keypoint_scores[index],
           sizeof(PoseKeypointScores));
    pose_scores[pose_counter] = all_instance_scores[index];
    pose_counter++;
  }

  return pose_counter;
}

}  // namespace posenet_decoder_op
}  // namespace coral
