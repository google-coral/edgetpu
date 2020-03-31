// Tests correctness of the decoder.
#include "src/cpp/posenet/posenet_decoder.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "absl/flags/parse.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using coral::posenet_decoder_op::Point;
using coral::posenet_decoder_op::PoseKeypoints;
using coral::posenet_decoder_op::PoseKeypointScores;
using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::Pointwise;

namespace coral {

TEST(DecreasingArgSortTest, UnsortedVector) {
  std::vector<float> scores = {0.6, 0.897, 0.01, 0.345, 0.28473};
  std::vector<int> indices(scores.size());

  DecreasingArgSort(scores, &indices);
  EXPECT_THAT(indices, ElementsAre(1, 0, 3, 4, 2));
}

TEST(DecreasingArgSortTest, AllSameVector) {
  std::vector<float> scores = {0.5, 0.5, 0.5, 0.5, 0.5};
  std::vector<int> indices(scores.size());

  DecreasingArgSort(scores, &indices);
  EXPECT_THAT(indices, ElementsAre(0, 1, 2, 3, 4));
}

TEST(DecreasingArgSortTest, NegativeVector) {
  std::vector<float> scores = {0.6, -0.897, 0.01, 0.345, 0.28473};
  std::vector<int> indices(scores.size());

  DecreasingArgSort(scores, &indices);
  EXPECT_THAT(indices, ElementsAre(0, 3, 4, 2, 1));
}

TEST(ComputeSquaredDistanceTest, XYPoints) {
  Point a = {0.5, 0.5};
  Point b = {1, 1};

  EXPECT_EQ(ComputeSquaredDistance(a, b), 0.5f);
}

TEST(SigmoidTest, Zero) { EXPECT_EQ(Sigmoid(0), 0.5f); }

TEST(SigmoidTest, Twenty) { EXPECT_EQ(Sigmoid(20), 1.0f); }

TEST(SigmoidTest, NegativeFive) { EXPECT_FLOAT_EQ(Sigmoid(-5), 0.006692851f); }

TEST(LogoddsTest, NegativeNumber) { EXPECT_TRUE(std::isnan(Logodds(-1))); }

TEST(LogoddsTest, Zero) { EXPECT_FLOAT_EQ(Logodds(0), -13.81551); }

TEST(LogoddsTest, FiveTenths) { EXPECT_FLOAT_EQ(Logodds(0.5), 0.000004); }

TEST(BuildLinearInterpolationTest, BuildYIsValid) {
  const int height = 3;
  const float y = 0.25f * height;
  float y_lerp;
  int y_floor;
  int y_ceil;

  BuildLinearInterpolation(y, height, &y_floor, &y_ceil, &y_lerp);

  EXPECT_FLOAT_EQ(y_lerp, 0.75f);
  EXPECT_EQ(y_floor, 0);
  EXPECT_EQ(y_ceil, 1);
}

TEST(BuildBilinearInterpolationTest, BuildXYIsValid) {
  const int height = 3;
  const int width = 4;
  const int num_channels = 3;
  const float y = 0.25f * height;
  const float x = 0.33f * width;
  int top_left;
  int top_right;
  int bottom_left;
  int bottom_right;
  float y_lerp;
  float x_lerp;

  BuildBilinearInterpolation(y, x, height, width, num_channels, &top_left,
                             &top_right, &bottom_left, &bottom_right, &y_lerp,
                             &x_lerp);

  EXPECT_EQ(top_left, 3);
  EXPECT_EQ(top_right, 6);
  EXPECT_EQ(bottom_left, 15);
  EXPECT_EQ(bottom_right, 18);
  EXPECT_FLOAT_EQ(y_lerp, 0.75f);
  EXPECT_FLOAT_EQ(x_lerp, 0.32f);
}

TEST(SampleTensorAtMultipleChannelsTest, SampleIsCorrect) {
  // Create an input tensor with shape [height, width, num_channels] and
  // values specified as a linear function of the position.
  const int height = 3;
  const int width = 4;
  const int num_channels = 3;
  const int size = height * width * num_channels;
  float tensor[size];
  int index = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < num_channels; ++c) {
        tensor[index++] = y + x + c + 0.2f;
      }
    }
  }
  // Sample the tensor at a mid-point position and multiple channels and verify
  // we get the expected results.
  const float y = 0.25f * height;
  const float x = 0.33f * width;
  int channels[] = {2, 0, 0, 1};
  const size_t n_result_channels = 4;
  float result[n_result_channels];

  SampleTensorAtMultipleChannels(tensor, height, width, num_channels, y, x,
                                 channels, n_result_channels, result);

  for (int i = 0; i < n_result_channels; i++) {
    EXPECT_FLOAT_EQ(result[i], y + x + channels[i] + 0.2f);
  }
}

TEST(SampleTensorAtSingleChannelTest, SampleIsCorrect) {
  // Create an input tensor with shape [height, width, num_channels] and
  // values specified as a linear function of the position.
  const int height = 3;
  const int width = 4;
  const int num_channels = 3;
  const int size = height * width * num_channels;
  float tensor[size];
  int index = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < num_channels; ++c) {
        tensor[index++] = y + x + c + 0.1f;
      }
    }
  }
  // Sample the tensor at a mid-point position and multiple channels and verify
  // we get the expected results.
  Point point{0.25f * height, 0.33f * width};
  const int c = num_channels / 2;

  const float result = SampleTensorAtSingleChannel(tensor, height, width,
                                                   num_channels, point, c);

  EXPECT_FLOAT_EQ(result, point.y + point.x + c + 0.1f);
}

TEST(FindDisplacedPositionTest, PositionIsCorrectAllZeros) {
  // The short_offsets tensors has size [height, width, num_keypoints * 2].
  // The mid_offsets tensors has size [height, width, 2 * 2 * num_edges].
  const int height = 10;
  const int width = 8;
  const int num_keypoints = 3;
  const int num_edges = 2 * (num_keypoints - 1);  // Forward-backward chain.
  // Create a short_offsets tensor with all 0s
  float short_offsets[height * width * num_keypoints * 2] = {};
  // Create a mid_offsets tensor with all 0s
  float mid_offsets[height * width * 2 * 2 * num_edges] = {};
  Point source{4.1, 3.5};
  const int edge_id = 1;
  const int target_id = 2;

  for (int i = 0; i < 4; ++i) {
    Point point_result = FindDisplacedPosition(
        short_offsets, mid_offsets, height, width, num_keypoints, num_edges,
        source, edge_id, target_id,
        /* mid_short_offset_refinement_steps=*/i);

    // Expect that the values are the same
    EXPECT_FLOAT_EQ(point_result.y, source.y);
    EXPECT_FLOAT_EQ(point_result.x, source.x);
  }
}

TEST(FindDisplacedPositionTest, PositionIsCorrectAllOnes) {
  // The short_offsets tensors has size [height, width, num_keypoints * 2].
  // The mid_offsets tensors has size [height, width, 2 * 2 * num_edges].
  const int height = 10;
  const int width = 8;
  const int num_keypoints = 3;
  const int num_edges = 2 * (num_keypoints - 1);  // Forward-backward chain.
  // Create a short_offsets tensor with all 1s
  float short_offsets[height * width * num_keypoints * 2];
  std::fill(std::begin(short_offsets), std::end(short_offsets), 1.0f);
  // Create a mid_offsets tensor with all -1s
  float mid_offsets[height * width * 2 * 2 * num_edges];
  std::fill(std::begin(mid_offsets), std::end(mid_offsets), -1.0f);
  Point source{4.1, 3.5};
  const int edge_id = 1;
  const int target_id = 2;

  for (int i = 0; i < 4; ++i) {
    Point point_result = FindDisplacedPosition(
        short_offsets, mid_offsets, height, width, num_keypoints, num_edges,
        source, edge_id, target_id,
        /* mid_short_offset_refinement_steps=*/i);

    // We move once by the (-1, -1) mid-offsets array to (y1 - 1, x1 - 1), and
    // then i-times by the (1, 1) short-offsets array, to
    // (y1 + i - 1, x1 + i - 1).
    EXPECT_FLOAT_EQ(point_result.y, source.y + i - 1);
    EXPECT_FLOAT_EQ(point_result.x, source.x + i - 1);
  }
}

TEST(FindDisplacedPositionTest, PositionIsCorrect) {
  // The short_offsets tensors has size [height, width, num_keypoints * 2].
  // The mid_offsets tensors has size [height, width, 2 * 2 num_edges].
  const int height = 10;
  const int width = 8;
  const int num_keypoints = 3;
  const int num_edges = 2 * (num_keypoints - 1);  // Forward-backward chain.
  // Create a short_offsets tensor with increasing values scaled from [0, 1]
  const int short_offsets_size = height * width * num_keypoints * 2;
  const int short_offsets_max_range =
      height - 1 + width - 1 + num_keypoints * 2 - 1;
  float short_offsets[short_offsets_size];
  int short_offsets_index = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < num_keypoints * 2; ++c) {
        short_offsets[short_offsets_index++] =
            (y + x + c + 0.1f) / (short_offsets_max_range + 0.1f);
      }
    }
  }
  // Create a mid_offsets tensor with increasing values scaled from [-1.0, 1.0]
  const int mid_offsets_size = height * width * 2 * 2 * num_edges;
  const int mid_offsets_max_range =
      height - 1 + width - 1 + 2 * 2 * num_edges - 1;
  float mid_offsets[mid_offsets_size];
  int mid_offsets_index = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 2 * 2 * num_edges; ++c) {
        mid_offsets[mid_offsets_index++] =
            2 * (y + x + c + 0.1f) / (mid_offsets_max_range)-0.5f;
      }
    }
  }
  Point source{4.1, 3.5};
  const int edge_id = 1;
  const int target_id = 2;
  // Make array of expected keypoints based on parameter i that
  // will be passed to FindDisplacedPosition (below)
  Point expected_points[] = {
      Point{4.161290, 3.819355},
      Point{4.639046, 4.439290},
      Point{5.168825, 5.111249},
      Point{5.755558, 5.840163},
  };

  for (int i = 0; i < 4; i++) {
    Point point_result = FindDisplacedPosition(
        short_offsets, mid_offsets, height, width, num_keypoints, num_edges,
        source, edge_id, target_id,
        /* mid_short_offset_refinement_steps=*/i);
    EXPECT_FLOAT_EQ(point_result.y, expected_points[i].y);
    EXPECT_FLOAT_EQ(point_result.x, expected_points[i].x);
  }
}

TEST(BuildAdjacencyListTest, BuildIsValid) {
  AdjacencyList expected_adjacency_list;
  expected_adjacency_list.child_ids = {
      {1, 2, 5, 6}, {3, 0},   {4, 0},   {1},  {2}, {7, 11, 0},
      {8, 12, 0},   {9, 5},   {10, 6},  {7},  {8}, {13, 5},
      {14, 6},      {15, 11}, {16, 12}, {13}, {14}};
  expected_adjacency_list.edge_ids = {
      {0, 2, 4, 10}, {1, 16}, {3, 18},  {17}, {19}, {5, 7, 20},
      {11, 13, 26},  {6, 21}, {12, 27}, {22}, {28}, {8, 23},
      {14, 29},      {9, 24}, {15, 30}, {25}, {31}};

  AdjacencyList adjacency_list = BuildAdjacencyList();

  EXPECT_EQ(adjacency_list.child_ids, expected_adjacency_list.child_ids);
  EXPECT_EQ(adjacency_list.edge_ids, expected_adjacency_list.edge_ids);
}

TEST(BacktrackDecodePose, BacktrackDecodePoseIsCorrect) {
  // The scores tensors has size [height, width, num_keypoints].
  // The short_offsets tensors has size [height, width, num_keypoints * 2].
  // The mid_offsets tensors has size [height, width, 2 * 2 * num_edges].
  const int height = 10;
  const int width = 8;
  const int num_keypoints = 3;
  const int num_edges = 2 * (num_keypoints - 1);  // Forward-backward chain.

  // Create a scores tensor with all 0.8s.
  float scores[height * width * num_keypoints];
  std::fill(std::begin(scores), std::end(scores), 0.8f);
  // Create a short_offsets tensor with all 1s
  float short_offsets[height * width * num_keypoints * 2];
  std::fill(std::begin(short_offsets), std::end(short_offsets), 1.0f);
  // Create a mid_offsets tensor with all -1s
  float mid_offsets[height * width * 2 * 2 * num_edges];
  std::fill(std::begin(mid_offsets), std::end(mid_offsets), -1.0f);

  // Build an adjacency list for a forward-backward linear chain connectivity,
  // i.e., the edge set is: (0 -> 1), (1 -> 2), (2 -> 1), (1 -> 0).
  AdjacencyList adjacency_list;
  adjacency_list.child_ids = {{1}, {2, 0}, {1}};
  adjacency_list.edge_ids = {{0}, {1, 3}, {2}};

  PoseKeypoints pose_keypoints[num_keypoints];
  PoseKeypointScores keypoint_scores[num_keypoints];

  const float y1 = 7.1;
  const float x1 = 5.5;
  KeypointWithScore root(Point{y1, x1}, /*id=*/1, /*score=*/0);
  // score is a filler, will be set in BacktrackDecodePose

  BacktrackDecodePose(scores, short_offsets, mid_offsets, height, width,
                      num_keypoints, num_edges, root, adjacency_list,
                      /*mid_short_offset_refinement_steps=*/2, pose_keypoints,
                      keypoint_scores);

  Point expected_pose_keypoints[] = {
      Point{y1 + 1, x1 + 1},
      Point{y1, x1},
      Point{y1 + 1, x1 + 1},
  };

  for (int i = 0; i < num_keypoints; i++) {
    // Compare pose_keypoints Point values (y and x)
    EXPECT_FLOAT_EQ(pose_keypoints->keypoint[i].y,
                    expected_pose_keypoints[i].y);
    EXPECT_FLOAT_EQ(pose_keypoints->keypoint[i].x,
                    expected_pose_keypoints[i].x);
    // Compare keypoint_scores scores
    EXPECT_FLOAT_EQ(keypoint_scores->keypoint[i], 0.8f);
  }
}

TEST(BuildKeypointWithScoreQueueTest, BuildIsValidWithThreshold) {
  // The scores tensors has size [height, width, num_keypoints].
  // The short_offsets tensors has size [height, width, num_keypoints * 2].
  const int height = 5;
  const int width = 4;
  const int num_keypoints = 3;
  // Create a scores tensor with all 0s except for two spikes
  float scores[height * width * num_keypoints] = {0};
  KeypointWithScore p1(Point{1, 2}, /*id=*/2, /*score=*/1.0f);
  KeypointWithScore p2(Point{3, 0}, /*id=*/1, /*score=*/2.0f);
  scores[static_cast<int>(p1.point.y * width + p1.point.x) * num_keypoints +
         p1.id] = p1.score;
  scores[static_cast<int>(p2.point.y * width + p2.point.x) * num_keypoints +
         p2.id] = p2.score;
  // Create a short_offsets tensor with all 1s
  float short_offsets[height * width * num_keypoints * 2];
  std::fill(std::begin(short_offsets), std::end(short_offsets), 1.0f);

  DecreasingScoreKeypointPriorityQueue queue;
  BuildKeypointWithScoreQueue(scores, short_offsets, height, width,
                              num_keypoints, /*score_threshold=*/0.5f,
                              /*local_maximum_radius=*/1, &queue);

  // Only two positions in the score map have scores above threshold (0.5f)
  EXPECT_EQ(queue.size(), 2);
}

TEST(BuildKeypointWithScoreQueueTest, BuildIsValidScoresCorrect) {
  // The scores tensors has size [height, width, num_keypoints].
  // The short_offsets tensors has size [height, width, num_keypoints * 2].
  const int height = 5;
  const int width = 4;
  const int num_keypoints = 3;
  // Create a scores tensor with all 0s except for two spikes
  float scores[height * width * num_keypoints] = {0};
  KeypointWithScore p1(Point{1, 2}, /*id=*/2, /*score=*/1.0f);
  KeypointWithScore p2(Point{3, 0}, /*id=*/1, /*score=*/2.0f);
  scores[static_cast<int>(p1.point.y * width + p1.point.x) * num_keypoints +
         p1.id] = p1.score;
  scores[static_cast<int>(p2.point.y * width + p2.point.x) * num_keypoints +
         p2.id] = p2.score;
  // Create a short_offsets tensor with all 1s
  float short_offsets[height * width * num_keypoints * 2];
  std::fill(std::begin(short_offsets), std::end(short_offsets), 1.0f);
  // Create expected queue values
  // The keypoints in the queue will be displaced by (1, 1) due to the offsets
  KeypointWithScore expected_keypoint1(
      Point{p1.point.y + 1.0f, p1.point.x + 1.0f}, p1.id, p1.score);
  KeypointWithScore expected_keypoint2(
      Point{p2.point.y + 1.0f, p2.point.x + 1.0f}, p2.id, p2.score);

  DecreasingScoreKeypointPriorityQueue queue;
  BuildKeypointWithScoreQueue(scores, short_offsets, height, width,
                              num_keypoints, /*score_threshold=*/0.5f,
                              /*local_maximum_radius=*/1, &queue);

  // Queue stores the keypoints in order of decreasing score values
  EXPECT_FLOAT_EQ(queue.top().score, expected_keypoint2.score);
  EXPECT_FLOAT_EQ(queue.top().point.y, expected_keypoint2.point.y);
  EXPECT_FLOAT_EQ(queue.top().point.x, expected_keypoint2.point.x);
  EXPECT_FLOAT_EQ(queue.top().id, expected_keypoint2.id);
  queue.pop();
  EXPECT_FLOAT_EQ(queue.top().score, expected_keypoint1.score);
  EXPECT_FLOAT_EQ(queue.top().point.y, expected_keypoint1.point.y);
  EXPECT_FLOAT_EQ(queue.top().point.x, expected_keypoint1.point.x);
  EXPECT_FLOAT_EQ(queue.top().id, expected_keypoint1.id);
}

TEST(PassKeypointNMSTest, SquaredDistanceLessThanSquaredNMSRadius) {
  const int num_keypoints = 2;
  PoseKeypoints pose_keypoints[num_keypoints];
  pose_keypoints->keypoint[0] = Point{0, 0};
  pose_keypoints->keypoint[1] = Point{1, 1};
  KeypointWithScore keypoint1(Point{0.5, 0.5}, /*id=*/0, /*score=*/0);
  // score doesn't matter here. Function doesn't check the score
  // The squared distance of this keypoint to previously detected keypoints is
  // min(0.5, 0.5) = 0.50

  EXPECT_FALSE(PassKeypointNMS(pose_keypoints, num_keypoints, keypoint1,
                               /*squared_nms_radius=*/0.55));
}

TEST(PassKeypointNMSTest, SquaredDistanceGreaterThanSquaredNMSRadius) {
  const int num_keypoints = 2;
  PoseKeypoints pose_keypoints[num_keypoints];
  pose_keypoints->keypoint[0] = Point{0, 0};
  pose_keypoints->keypoint[1] = Point{1, 1};
  KeypointWithScore keypoint1(Point{0.5, 0.5}, /*id=*/0, /*score=*/0);
  // score doesn't matter here. Function doesn't check the score
  // The squared distance of this keypoint to previously detected keypoints is
  // min(0.5, 0.5) = 0.5

  EXPECT_TRUE(PassKeypointNMS(pose_keypoints, num_keypoints, keypoint1,
                              /*squared_nms_radius=*/0.45));
}

TEST(FindOverlappingKeypointsTest, KeypointsResultNoOverlap) {
  PoseKeypoints pose_keypoints1;
  pose_keypoints1.keypoint[0] = Point{0, 0};
  pose_keypoints1.keypoint[1] = Point{0, 1};
  PoseKeypoints pose_keypoints2;
  pose_keypoints2.keypoint[0] = Point{1, 1};
  pose_keypoints2.keypoint[1] = Point{1, 0};
  std::vector<bool> mask(2, false);

  FindOverlappingKeypoints(pose_keypoints1, pose_keypoints2,
                           /*squared_radius=*/1.0f, &mask);

  EXPECT_THAT(mask, ElementsAre(false, false));
}

TEST(FindOverlappingKeypointsTest, KeypointsResultSamePoints) {
  PoseKeypoints pose_keypoints1;
  pose_keypoints1.keypoint[0] = Point{0, 0};
  pose_keypoints1.keypoint[1] = Point{0, 1};
  PoseKeypoints pose_keypoints2;
  pose_keypoints2.keypoint[0] = Point{0, 0};
  pose_keypoints2.keypoint[1] = Point{0, 1};
  std::vector<bool> mask(2, false);

  FindOverlappingKeypoints(pose_keypoints1, pose_keypoints2,
                           /*squared_radius=*/1.0f, &mask);

  EXPECT_THAT(mask, ElementsAre(true, true));
}

TEST(FindOverlappingKeypointsTest, KeypointsResultOverlap) {
  PoseKeypoints pose_keypoints1;
  pose_keypoints1.keypoint[0] = Point{0, 0};
  pose_keypoints1.keypoint[1] = Point{1, 1};
  PoseKeypoints pose_keypoints2;
  pose_keypoints2.keypoint[0] = Point{0, 0.9};
  pose_keypoints2.keypoint[1] = Point{2, 2};
  std::vector<bool> mask(2, false);

  FindOverlappingKeypoints(pose_keypoints1, pose_keypoints2,
                           /*squared_radius=*/1.0f, &mask);

  EXPECT_THAT(mask, ElementsAre(true, false));
}

std::vector<float> TestSoftKeypointNMS(const float squared_nms_radius,
                                       const int topk) {
  const int num_keypoints = 2;
  std::vector<float> all_instance_scores(num_keypoints);
  PoseKeypoints all_keypoint_coords[] = {
      {Point{0, 0}, Point{1, 1}},
      {Point{0, 1}, Point{2, 2}},
  };
  PoseKeypointScores all_keypoint_scores[] = {
      {0.1, 0.2},
      {0.3, 0.4},
  };

  // Set the second instance to be stronger than the first instance
  const std::vector<int> decreasing_indices = {1, 0};

  PerformSoftKeypointNMS(decreasing_indices, all_keypoint_coords,
                         all_keypoint_scores, num_keypoints, squared_nms_radius,
                         topk, &all_instance_scores);

  return all_instance_scores;
}

TEST(PerformSoftKeypointNMSTest, PerformSoftKeypointNMSAverage) {
  // For PerformSoftKeypointNMS, set the instance-level score to the average of
  // the keypoint-level scores by setting topk=num_keypoints (set to 2)

  // The score of the second (stronger) instance will always be 0.35. The score
  // of the first (weaker) instance will depend of the value of the nms_radius,
  // which determines if the keypoints of the weaker instance are masked by the
  // corresponding keypoints of the stronger instance

  // If 0 < squared_nms_radius < 1: {false, false}
  EXPECT_THAT(TestSoftKeypointNMS(/*squared_nms_radius=*/0.9, /*topk=*/2),
              Pointwise(FloatEq(), {0.15, 0.35}));

  // If 1 < squared_nms_radius < 2: {true, false}
  EXPECT_THAT(TestSoftKeypointNMS(/*squared_nms_radius=*/1.5, /*topk=*/2),
              Pointwise(FloatEq(), {0.1, 0.35}));

  // If 2 < squared_nms_radius: {true, true}
  EXPECT_THAT(TestSoftKeypointNMS(/*squared_nms_radius=*/2.1, /*topk=*/2),
              Pointwise(FloatEq(), {0.0, 0.35}));
}

TEST(PerformSoftKeypointNMSTest, PerformSoftKeypointNMSMaximum) {
  // For PerformSoftKeypointNMS, set the instance-level score to the maximum of
  // the keypoint-level scores by setting topk=1

  // The score of the second (stronger) instance will always be 0.4. The score
  // of the first (weaker) instance will depend of the value of the nms_radius,
  // which determines if the keypoints of the weaker instance are masked by the
  // corresponding keypoints of the stronger instance

  // If 0 < squared_nms_radius < 1: {false, false}
  EXPECT_THAT(TestSoftKeypointNMS(/*squared_nms_radius=*/0.9, /*topk=*/1),
              Pointwise(FloatEq(), {0.2, 0.4}));

  // If 1 < squared_nms_radius < 2: {true, false}
  EXPECT_THAT(TestSoftKeypointNMS(/*squared_nms_radius=*/1.5, /*topk=*/1),
              Pointwise(FloatEq(), {0.2, 0.4}));

  // If 2 < squared_nms_radius: {true, true}
  EXPECT_THAT(TestSoftKeypointNMS(/*squared_nms_radius=*/2.1, /*topk=*/1),
              Pointwise(FloatEq(), {0.0, 0.4}));
}
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
