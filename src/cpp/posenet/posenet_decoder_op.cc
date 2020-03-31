#include "src/cpp/posenet/posenet_decoder_op.h"

#include <cmath>
#include <numeric>
#include <string>

#include "flatbuffers/flexbuffers.h"
#include "src/cpp/posenet/posenet_decoder.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace coral {
namespace posenet_decoder_op {

using tflite::GetInput;
using tflite::GetOutput;
using tflite::GetTensorData;
using tflite::NumDimensions;
using tflite::NumInputs;
using tflite::NumOutputs;

constexpr int kInputTensorHeatmaps = 0;
constexpr int kInputTensorShortOffsets = 1;
constexpr int kInputTensorMidOffsets = 2;

constexpr int kOutputTensorPoseKeypoints = 0;
constexpr int kOutputTensorPoseKeypointScores = 1;
constexpr int kOutputTensorPoseScores = 2;
constexpr int kOutputTensorPoseCount = 3;

struct OpData {
  // Decoder parameters
  int max_detections;
  float score_threshold;
  int stride;
  float nms_radius;

  // Temporary tensors (e.g for dequantized values)
  int heatmaps_float_index;
  int shorts_float_index;
  int mids_float_index;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData;
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();

  op_data->max_detections = m["max_detections"].AsInt32();
  op_data->score_threshold = m["score_threshold"].AsFloat();
  op_data->stride = m["stride"].AsInt32();
  op_data->nms_radius = m["nms_radius"].AsFloat();

  context->AddTensors(context, 1, &op_data->heatmaps_float_index);
  context->AddTensors(context, 1, &op_data->shorts_float_index);
  context->AddTensors(context, 1, &op_data->mids_float_index);
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus PrepTempTensor(TfLiteContext* context, int temp_tensor_index,
                            const TfLiteIntArray* dims) {
  TfLiteTensor* temp_tensor = &context->tensors[temp_tensor_index];
  temp_tensor->type = kTfLiteFloat32;
  temp_tensor->allocation_type = kTfLiteArenaRw;
  return context->ResizeTensor(context, temp_tensor, TfLiteIntArrayCopy(dims));
}

TfLiteStatus PrepOutputTensor(TfLiteContext* context,
                              TfLiteTensor* output_tensor,
                              std::initializer_list<int> dims) {
  output_tensor->type = kTfLiteFloat32;
  TfLiteIntArray* size = TfLiteIntArrayCreate(dims.size());
  std::copy(std::begin(dims), std::end(dims), size->data);
  return context->ResizeTensor(context, output_tensor, size);
}

void ScaleFloatTensor(const TfLiteTensor* src, TfLiteTensor* dst, float scale) {
  assert(src->type == kTfLiteFloat32);
  assert(dst->type == kTfLiteFloat32);
  const float* src_data = GetTensorData<float>(src);
  float* dst_data = GetTensorData<float>(dst);
  assert(src_data != nullptr);
  assert(dst_data != nullptr);
  const size_t tensor_elements = src->bytes / sizeof(float);
  for (int idx = 0; idx < tensor_elements; ++idx) {
    dst_data[idx] = src_data[idx] * scale;
  }
}

void DequantizeTensor(const TfLiteTensor* src, TfLiteTensor* dst,
                      float extra_scale = 1.0) {
  if (src->type == kTfLiteUInt8) {
    const int num_elements = src->bytes;
    assert(num_elements * sizeof(float) == dst->bytes);
    const float quant_zero_point = static_cast<float>(src->params.zero_point);
    const float quant_scale = src->params.scale * extra_scale;
    const uint8_t* src_data = GetTensorData<uint8_t>(src);
    assert(src_data != nullptr);
    float* dst_data = GetTensorData<float>(dst);
    assert(dst_data != nullptr);
    for (int idx = 0; idx < num_elements; ++idx) {
      dst_data[idx] = (src_data[idx] - quant_zero_point) * quant_scale;
    }
  } else if (src->type == kTfLiteFloat32) {
    ScaleFloatTensor(src, dst, extra_scale);
  } else {
    assert(false);
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 4);

  const TfLiteTensor* heatmaps = GetInput(context, node, kInputTensorHeatmaps);
  const TfLiteTensor* shorts =
      GetInput(context, node, kInputTensorShortOffsets);
  const TfLiteTensor* mids = GetInput(context, node, kInputTensorMidOffsets);

  TF_LITE_ENSURE(context, (heatmaps->type == kTfLiteUInt8 ||  //
                           heatmaps->type == kTfLiteFloat32));
  TF_LITE_ENSURE(context, (shorts->type == kTfLiteUInt8 ||  //
                           shorts->type == kTfLiteFloat32));
  TF_LITE_ENSURE(context, (mids->type == kTfLiteUInt8 ||  //
                           mids->type == kTfLiteFloat32));
  TF_LITE_ENSURE_EQ(context, NumDimensions(heatmaps), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(shorts), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(mids), 4);
  TF_LITE_ENSURE_EQ(context, heatmaps->dims->data[0], 1);
  TF_LITE_ENSURE_EQ(context, shorts->dims->data[0], 1);
  TF_LITE_ENSURE_EQ(context, mids->dims->data[0], 1);
  TF_LITE_ENSURE_EQ(context, heatmaps->dims->data[3], kNumKeypoints);
  TF_LITE_ENSURE_EQ(context, shorts->dims->data[3], 2 * kNumKeypoints);
  TF_LITE_ENSURE_EQ(context, mids->dims->data[3], 2 * 2 * kNumEdges);

  // Temporary tensors
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(3);
  node->temporaries->data[0] = op_data->heatmaps_float_index;
  node->temporaries->data[1] = op_data->shorts_float_index;
  node->temporaries->data[2] = op_data->mids_float_index;

  TF_LITE_ENSURE_OK(
      context,
      PrepTempTensor(context, op_data->heatmaps_float_index, heatmaps->dims));
  TF_LITE_ENSURE_OK(
      context,
      PrepTempTensor(context, op_data->shorts_float_index, shorts->dims));
  TF_LITE_ENSURE_OK(
      context, PrepTempTensor(context, op_data->mids_float_index, mids->dims));

  // Output tensor 0 will be max_detections*kNumKeypoints*2
  // The last dimension has the x and y coordinates of each keypoint.
  TF_LITE_ENSURE_OK(
      context,
      PrepOutputTensor(context,
                       GetOutput(context, node, kOutputTensorPoseKeypoints),
                       {1, op_data->max_detections, kNumKeypoints, 2}));

  // Output tensor 1 to be size max_detections*kNumKeypoints and contain
  // keypoints scores in the range [0,1].
  TF_LITE_ENSURE_OK(
      context,
      PrepOutputTensor(
          context, GetOutput(context, node, kOutputTensorPoseKeypointScores),
          {1, op_data->max_detections, kNumKeypoints}));

  // Output tensor 2 to be size max_detections and contain
  // pose scores in the range [0,1].

  TF_LITE_ENSURE_OK(
      context, PrepOutputTensor(
                   context, GetOutput(context, node, kOutputTensorPoseScores),
                   {1, op_data->max_detections}));

  // Output Tensor 3 is an int32 scalar, the number of detected poses.
  // Currently only float output tensors are supported so save this as a float.
  TF_LITE_ENSURE_OK(
      context,
      PrepOutputTensor(context,
                       GetOutput(context, node, kOutputTensorPoseCount), {1}));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE(context, op_data->stride > 0);
  const TfLiteTensor* heatmaps = GetInput(context, node, kInputTensorHeatmaps);
  const TfLiteTensor* shorts =
      GetInput(context, node, kInputTensorShortOffsets);
  const TfLiteTensor* mids = GetInput(context, node, kInputTensorMidOffsets);

  // Dequantize (and rescale) input tensors

  TfLiteTensor* heatmaps_float =
      &context->tensors[op_data->heatmaps_float_index];
  TfLiteTensor* shorts_float = &context->tensors[op_data->shorts_float_index];
  TfLiteTensor* mids_float = &context->tensors[op_data->mids_float_index];

  DequantizeTensor(heatmaps, heatmaps_float);
  DequantizeTensor(shorts, shorts_float, 1.0 / op_data->stride);
  DequantizeTensor(mids, mids_float, 1.0 / op_data->stride);

  const float* heatmaps_data = GetTensorData<float>(heatmaps_float);
  const float* mids_data = GetTensorData<float>(mids_float);
  const float* shorts_data = GetTensorData<float>(shorts_float);

  TfLiteTensor* pose_keypoints =
      GetOutput(context, node, kOutputTensorPoseKeypoints);
  TfLiteTensor* pose_keypoint_scores =
      GetOutput(context, node, kOutputTensorPoseKeypointScores);
  TfLiteTensor* pose_scores = GetOutput(context, node, kOutputTensorPoseScores);
  TfLiteTensor* pose_count = GetOutput(context, node, kOutputTensorPoseCount);
  float* pose_keypoints_data = GetTensorData<float>(pose_keypoints);
  float* pose_keypoint_scores_data = GetTensorData<float>(pose_keypoint_scores);
  float* pose_scores_data = GetTensorData<float>(pose_scores);
  float* pose_count_data = GetTensorData<float>(pose_count);

  const float nms_radius = op_data->nms_radius / op_data->stride;
  pose_count_data[0] = DecodeAllPoses(
      heatmaps_data, shorts_data, mids_data,
      /*height = */ heatmaps_float->dims->data[1],
      /*width = */ heatmaps_float->dims->data[2], op_data->max_detections,
      op_data->score_threshold,
      /*mid_short_offset_refinement_steps = */ 5, nms_radius, op_data->stride,
      reinterpret_cast<PoseKeypoints*>(pose_keypoints_data),
      reinterpret_cast<PoseKeypointScores*>(pose_keypoint_scores_data),
      pose_scores_data);

  return kTfLiteOk;
}

}  // namespace posenet_decoder_op

TfLiteRegistration* RegisterPosenetDecoderOp() {
  static TfLiteRegistration r = {
      posenet_decoder_op::Init, posenet_decoder_op::Free,
      posenet_decoder_op::Prepare, posenet_decoder_op::Eval};
  return &r;
}

}  // namespace coral
