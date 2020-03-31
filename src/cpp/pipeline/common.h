#ifndef EDGETPU_CPP_PIPELINE_COMMON_H_
#define EDGETPU_CPP_PIPELINE_COMMON_H_

#include "tensorflow/lite/c/common.h"

namespace coral {

// A tensor in the pipeline system.
// This is a simplified version of `TfLiteTensor`.
struct PipelineTensor {
  // The data type specification for data stored in `data`. This affects
  // what member of `data` union should be used.
  TfLiteType type;
  // A union of data pointers. The appropriate type should be used for a typed
  // tensor based on `type`.
  TfLitePtrUnion data;
  // The number of bytes required to store the data of this tensor. That is:
  // `(bytes of each element) * dims[0] * ... * dims[n-1]`. For example, if
  // type is kTfLiteFloat32 and `dims = {3, 2}` then
  // `bytes = sizeof(float) * 3 * 2 = 4 * 3 * 2 = 24`.
  size_t bytes;
};

// Performance statistics for one segment of model pipeline.
struct SegmentStats {
  // Total time spent traversing this segment so far (in nanoseconds).
  int64_t total_time_ns = 0;
  // Number of inferences processed so far.
  uint64_t num_inferences = 0;
};

}  // namespace coral

#endif  // EDGETPU_CPP_PIPELINE_COMMON_H_
