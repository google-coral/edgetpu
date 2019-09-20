#include <cassert>
#include <cstring>
#include <vector>

#include "src/cpp/posenet/posenet_decoder_op.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/util.h"

namespace {
using tflite::ConvertVectorToTfLiteIntArray;
using tflite::TfLiteIntArrayView;

typedef void (*ErrorHandler)(const char*);

constexpr char kDelegateName[] = "PosenetDelegateForCustomOp";
constexpr int kDelegateVersion = 1;

void* DelegateInit(TfLiteContext* context, const char* buffer, size_t length) {
  const TfLiteDelegateParams* params =
      reinterpret_cast<const TfLiteDelegateParams*>(buffer);
  assert(params);

  TfLiteIntArray* nodes = params->nodes_to_replace;
  assert(nodes->size == 1);
  const int node_index = nodes->data[0];

  TfLiteNode* node;
  TfLiteRegistration* registration;
  context->GetNodeAndRegistration(context, node_index, &node, &registration);

  return coral::RegisterPosenetDecoderOp()->init(
      context, static_cast<const char*>(node->custom_initial_data),
      node->custom_initial_data_size);
}

TfLiteStatus PrepareImpl(TfLiteContext* context, TfLiteDelegate* delegate) {
  TfLiteIntArray* plan;
  TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan));

  std::vector<int> supported_nodes;
  for (int node_index : TfLiteIntArrayView(plan)) {
    TfLiteNode* node;
    TfLiteRegistration* registration;
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_index, &node, &registration));

    if (registration->custom_name &&
        std::strcmp(registration->custom_name, coral::kPosenetDecoderOp) == 0) {
      supported_nodes.push_back(node_index);
    }
  }

  TfLiteRegistration registration = *coral::RegisterPosenetDecoderOp();
  registration.init = DelegateInit;
  registration.custom_name = kDelegateName;
  registration.version = kDelegateVersion;

  for (int node_index : supported_nodes) {
    TfLiteIntArray* nodes = ConvertVectorToTfLiteIntArray({node_index});
    context->ReplaceNodeSubsetsWithDelegateKernels(context, registration, nodes,
                                                   delegate);
    TfLiteIntArrayFree(nodes);
  }

  return kTfLiteOk;
}

class PosenetDelegateForCustomOp : public TfLiteDelegate {
 public:
  PosenetDelegateForCustomOp() : TfLiteDelegate(TfLiteDelegateCreate()) {
    this->Prepare = PrepareImpl;
    this->flags = kTfLiteDelegateFlagsAllowDynamicTensors;
  }
};

}  // namespace

extern "C" {

TfLiteDelegate* tflite_plugin_create_delegate(char** options_keys,
                                              char** options_values,
                                              size_t num_options,
                                              ErrorHandler error_handler) {
  return new PosenetDelegateForCustomOp();
}

void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate) {
  delete static_cast<PosenetDelegateForCustomOp*>(delegate);
}

}  // extern "C"
