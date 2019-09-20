#ifndef EDGETPU_CPP_BBOX_UTILS_H_
#define EDGETPU_CPP_BBOX_UTILS_H_

#include <vector>

#include "absl/strings/substitute.h"

namespace coral {

// Object Detection model produces axis-aligned boxes in two formats:
// BoxCorner represents the lower left corner (xmin, ymin) and
// the upper right corner (xmax, ymax).
// CenterSize represents the center (xcenter, ycenter), height and width.
// BoxCornerEncoding and CenterSizeEncoding are related as follows:
// ycenter = y / y_scale * anchor.h + anchor.y;
// xcenter = x / x_scale * anchor.w + anchor.x;
// half_h = 0.5*exp(h/ h_scale)) * anchor.h;
// half_w = 0.5*exp(w / w_scale)) * anchor.w;
// ymin = ycenter - half_h
// ymax = ycenter + half_h
// xmin = xcenter - half_w
// xmax = xcenter + half_w
struct BoxCornerEncoding {
  float ymin;
  float xmin;
  float ymax;
  float xmax;

  inline std::string DebugString() const {
    return absl::Substitute("ymin=$0,xmin=$1,ymax=$2,xmax=$3", ymin, xmin, ymax,
                            xmax);
  }
};

inline bool operator==(const BoxCornerEncoding& x, const BoxCornerEncoding& y) {
  return x.xmin == y.xmin && x.xmax == y.xmax && x.ymin == y.ymin &&
         x.ymax == y.ymax;
}

inline bool operator!=(const BoxCornerEncoding& x, const BoxCornerEncoding& y) {
  return !(x == y);
}

struct CenterSizeEncoding {
  float y;
  float x;
  float h;
  float w;

  inline std::string DebugString() const {
    return absl::Substitute("y=$0,x=$1,h=$2,w=$3", y, x, h, w);
  }

  inline BoxCornerEncoding ConvertToBoxCornerEncoding() const {
    const float half_h = 0.5 * h;
    const float half_w = 0.5 * w;
    return BoxCornerEncoding({y - half_h, x - half_w, y + half_h, x + half_w});
  }
};

// Detection result.
struct DetectionCandidate {
  BoxCornerEncoding corners;
  int label;
  float score;
  inline std::string DebugString() const {
    return absl::Substitute("corners={$0},label=$1,score=$2",
                            corners.DebugString(), label, score);
  }
};

// Compare based on id and score only.
inline bool operator==(const DetectionCandidate& x,
                       const DetectionCandidate& y) {
  return x.score == y.score && x.label == y.label && x.corners == y.corners;
}

inline bool operator!=(const DetectionCandidate& x,
                       const DetectionCandidate& y) {
  return !(x == y);
}

// Defines a comparator which allows us to rank DetectionCandidate based on
// their score and id.
struct DetectionCandidateComparator {
  bool operator()(const DetectionCandidate& lhs,
                  const DetectionCandidate& rhs) const {
    return std::tie(lhs.score, lhs.label) > std::tie(rhs.score, rhs.label);
  }
};

inline bool IsBoxEmpty(const BoxCornerEncoding& box) {
  return (box.ymin >= box.ymax || box.xmin >= box.xmax);
}

inline float ComputeBoxArea(const BoxCornerEncoding& box) {
  if (IsBoxEmpty(box)) return 0;
  return (box.ymax - box.ymin) * (box.xmax - box.xmin);
}

// Returns intersection over union of two boxes.
inline float IntersectionOverUnion(const BoxCornerEncoding& box0,
                                   const BoxCornerEncoding& box1) {
  const float area0 = ComputeBoxArea(box0);
  const float area1 = ComputeBoxArea(box1);
  if (area0 <= 0 || area1 <= 0) return 0.0;
  const float intersection_ymin = std::max<float>(box0.ymin, box1.ymin);
  const float intersection_xmin = std::max<float>(box0.xmin, box1.xmin);
  const float intersection_ymax = std::min<float>(box0.ymax, box1.ymax);
  const float intersection_xmax = std::min<float>(box0.xmax, box1.xmax);
  const float intersection_area =
      std::max<float>(intersection_ymax - intersection_ymin, 0.0) *
      std::max<float>(intersection_xmax - intersection_xmin, 0.0);
  return intersection_area / (area0 + area1 - intersection_area);
}

}  // namespace coral

#endif  // EDGETPU_CPP_BBOX_UTILS_H_
