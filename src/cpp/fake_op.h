#ifndef EDGETPU_CPP_FAKE_OP_H_
#define EDGETPU_CPP_FAKE_OP_H_

#include "tensorflow/lite/context.h"

namespace coral {

static const char kFakeOpDouble[] = "fake-op-double";

TfLiteRegistration* RegisterFakeOpDouble();

}  // namespace coral

#endif  // EDGETPU_CPP_FAKE_OP_H_
