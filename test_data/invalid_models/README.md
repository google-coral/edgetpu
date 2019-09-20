model_invoke_error.tflite is constructed via following method.

The input is a list of uint8, length = 3. It always returns kTfLiteError when
executing FakeOp.

```
#include "third_party/tensorflow/lite/kernels/test_util.h"

class FakeOpModel : public SingleOpModel {
 public:
  FakeOpModel(const TensorData& input, const TensorData& output,
              bool throw_error) {
    input_ = AddInput(input);
    output_ = AddOutput(output);

    flexbuffers::Builder fbb;
    fbb.Map([&]() { fbb.Bool("throw_error", throw_error); });
    fbb.Finish();
    SetCustomOp(kFakeOpDouble, fbb.GetBuffer(), RegisterFakeOpDouble);
    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }

  template <class T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input_, data);
  }

  template <class T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }
  TfLiteStatus InvokeWithoutCheck() { return interpreter_->Invoke(); }

 protected:
  int input_;
  int output_;
};

FakeOpModel m(
    // input tensors
    {TensorType_UINT8, {1, 3}},
    // output tensors
    {TensorType_FLOAT32, {}},
    // throw error
    true);
```
