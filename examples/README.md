# Edge TPU Python API examples

This directory contains several examples for the [Edge TPU
Python API](https://github.com/google-coral/edgetpu/tree/master/edgetpu).

**NOTICE:** The Edge TPU Python library is deprecated. We recommend you
instead use [PyCoral](https://github.com/google-coral/pycoral).

## Prerequisites

Before you begin, you must [set up your Coral device](https://coral.ai/docs/setup/).


## Download the Edge TPU API examples

Although you can clone this GitHub repo, it's actually very big, so we suggest you download
the Edge TPU Python examples with this Debian package:

```.language-bash
sudo apt-get update

sudo apt-get install edgetpu-examples
```

The examples are saved at `/usr/share/edgetpu/examples/` and include pre-compiled models
and images useful for each of the code examples.

For more pre-compiled models, see [coral.ai/models](https://coral.ai/models/).


## Run the example code

Each `.py` file includes documentation at the top with an example command you can use to run it,
using a model included with the Debian package and this repo's `test_data` directory.

For more information about building models and running inference on the Edge TPU, see the
[Coral documentation](https://coral.ai/docs/).
