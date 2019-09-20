#ifndef EDGETPU_CPP_POSENET_POSENET_DECODER_H_
#define EDGETPU_CPP_POSENET_POSENET_DECODER_H_

namespace coral {
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
//   short_offsets 23x31x2*kNumKeypoints
//   mid_offsets 23x31x2*kNumEdges
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
}  // namespace coral

#endif  // EDGETPU_CPP_POSENET_POSENET_DECODER_H_
