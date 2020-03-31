# C++ examples using Edge TPU

To build all the examples in this directory, you first need to
[install Bazel](https://docs.bazel.build/versions/master/install.html) and
(optional but we recommend)
[Docker](https://docs.docker.com/install/).

Then navigate up to the root `edgetpu` directory and run the following command:

```
make DOCKER_IMAGE=debian:stretch DOCKER_CPUS="aarch64" DOCKER_TARGETS="examples" docker-build
```

When done, you'll find the example binaries in `edgetpu/out/aarch64/examples/`.

The above command builds for `aarch64` (compatible with the Coral Dev Board),
but alternative CPU options are `k8`, `armv7a`, and `darwin`.

**Tip:** Instead of building on your computer, just
[run this Colab notebook](https://colab.sandbox.google.com/github/google-coral/tutorials/blob/master/build_cpp_examples.ipynb)
to build the examples and download the binaries.


## Build without Bazel

The `minimal.cc` file also has its own `Makefile` in this
directory, which offers a simple example of how build a project with the
`libedgetpu` library, using Make and the g++ compiler.

For details, see the documentation inside the `Makefile`.

**Note:** Inside the Makefile, you must specify `TENSORFLOW_DIR` for your local
copy of the TensorFlow repo, which must be cloned using the
[`TENSORFLOW_COMMIT` version specified
here](https://github.com/google-coral/edgetpu/blob/master/WORKSPACE)—that's the
version used to build the `libedgetpu` library, so your TensorFlow version
must match.
