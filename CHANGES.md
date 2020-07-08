# edgetpu release notes

This is a non-exhaustive summary of changes to the Edge TPU library, compiler, and runtime.

For pre-built downloads, see [coral.ai/software](https://coral.ai/software/).

## Edge TPU runtime and Edge TPU compiler 14.1 (July 2020 "eel2")

+   The Edge TPU runtime now supports PCIe interface on 64-bit Windows 10 systems (x86-64)
+   The Edge TPU compiler includes bug fixes and the version number now aligns with our Debian
    package versions

## Source code update, Edge TPU Python library 2.14.0 and Edge TPU compiler 2.1.302470888 (March 2020 "eel")

+   C++ support for model pipelining
+   Update Edge TPU python library to 2.14.0. New `size` parameter for `run_inference`
+   Compiler update (new compiler options `num_segments` and `intermediate_tensors`)
+   EfficientNet embedding extractor available for on-device learning with backprop


## Edge TPU runtime v13 and Edge TPU compiler 2.0.291256449 (January 2020)

+   Bug fixes
+   Added support for ops: ExpandDims, Pack, Quantize, Sum, and TransposeConv
+   First runtime release made available for Mac and Windows (USB interface only) — the compiler
    is still Linux only


## Edge TPU Python library 2.13.0 (January 2020 "diploria3")

+   Updated the build based on the latest TensorFlow source (no API changes)
+   First release made available for Mac and Windows


## Edge TPU Python library 2.12.2 (November 2019 "diploria2")

+   Performance optimizations


## Edge TPU Python library 2.12.1 (September 2019)

+   Python API signature style changes.
+   New `read_label_file()` utility function.


## Edge TPU compiler 2.0.267685300 (September 2019)

+   Improved support for models built with [full integer post-training quantization](
    https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization_of_weights_and_activations)—especially
    those built with the Keras API.
+   Added support for the DeepLab v3 semantic segmentation model.
+   Still compatible with Edge TPU runtime v12.


## Edge TPU Python library 2.11.1 (July 2019 "diploria")

+   New [`SoftmaxRegression`](
    https://coral.ai/docs/reference/edgetpu.learn.backprop.softmax_regression/) API that allows you
    to perform [transfer learning with on-device backpropagation](
    https://coral.ai/docs/edgetpu/retrain-classification-ondevice-backprop/).
+   Re-implementation of the [`ImprintingEngine`](
    https://coral.ai/docs/reference/edgetpu.learn.imprinting.engine/) API so you can keep the
    pre-trained classes from the provided model, retrain existing classes without starting from
    scratch, and immediately perform inference without exporting to a new `.tflite` file. For
    details, see [Retrain a classification model on-device with weight imprinting](
    https://coral.ai/docs/edgetpu/retrain-classification-ondevice/).


## Edge TPU compiler 2.0 and Edge TPU runtime v12 (July 2019)

+   Added support in `libedgetpu.so` for the [TensorFlow Lite delegate API](
    https://www.tensorflow.org/lite/performance/delegates), allowing you to perform inferences
    directly from the TensorFlow Lite Python API (instead of using the Edge TPU Python API). For
    instructions, see [Run inference with TensorFlow Lite in Python](
    https://coral.ai/docs/edgetpu/tflite-python/).
+   Added support for models built with [full integer post-training quantization](
    https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization_of_weights_and_activations).
+   Added support for new [EfficientNet-EdgeTPU](
    https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu) models.


## Edge TPU compiler 1.0 (May 2019)

+   New offline [Edge TPU Compiler](https://coral.ai/docs/edgetpu/compiler/).


## Edge TPU C++ runtime library and Edge TPU Python library 1.9.2 (April 2019)

+   New `edgetpu.h` C++ API to perform inferencing on the Edge TPU, using the
    TensorFlow Lite C++ API.
+   You can now [run multiple models with multiple Edge TPUs](
    https://coral.ai/docs/edgetpu/multiple-edgetpu/).
+   New `edgetpu.utils.image_processing` APIs to process images before running an inference.
