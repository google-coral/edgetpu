#ifndef EDGETPU_CPP_POSENET_POSENET_DECODER_OP_H_
#define EDGETPU_CPP_POSENET_POSENET_DECODER_OP_H_

#include "tensorflow/lite/context.h"

namespace coral {

static const char kPosenetDecoderOp[] = "PosenetDecoderOp";

TfLiteRegistration* RegisterPosenetDecoderOp();

}  // namespace coral

#endif  // EDGETPU_CPP_POSENET_POSENET_DECODER_OP_H_
