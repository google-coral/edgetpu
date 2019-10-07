cmake_minimum_required(VERSION 3.11)
project(edgetpu-cpp)


set(CMAKE_C_FLAGS "-Wall -pthread ")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++11  -lstdc++")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})


# Set default build type.
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

# When using Ninja, compiler output won't be colorized without this.
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-fdiagnostics-color=always SUPPORTS_COLOR_ALWAYS)
if(SUPPORTS_COLOR_ALWAYS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
endif()

find_program(LSB_RELEASE_EXEC lsb_release)
execute_process(COMMAND ${LSB_RELEASE_EXEC} -is
    OUTPUT_VARIABLE LSB_RELEASE_ID_SHORT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(COMMAND grep "TENSORFLOW_COMMIT =" ../../../WORKSPACE
	RESULT_VARIABLE TF_SHA
	OUTPUT_VARIABLE TF_SHA1
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE " " ";" SHA_LIST ${TF_SHA1})
list(GET SHA_LIST 2 TF_SHA)
string(REPLACE "\"" "" TF_SHA ${TF_SHA})
message("TF SHA set to ${TF_SHA} bazel from WORKSPACE")

if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    set(EDGE_ARCH "k8")
    set(TF_COMMAND make -j8 BUILD_WITH_NNAPI=false -C ${CMAKE_BINARY_DIR}/tensorflow/src/tf -f tensorflow/lite/tools/make/Makefile lib)
    set(TF_INSTALL_PREFIX "linux_x86_64")
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    set(EDGE_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
    set(TF_INSTALL_PREFIX "generic-aarch64_armv8-a")
    if(${LSB_RELEASE_ID_SHORT} STREQUAL "Gentoo")
      set(CROSS_PREFIX "aarch64-unknown-linux-gnu-")
    else()
      set(CROSS_PREFIX "aarch64-linux-gnu-")
    endif()
    #TODO: Export CROSS_PREFIX, TARGET and TARGET_ARCH as CMAKE Vars
    #FIXME: Fix upstream TF issue requiring -fpermissive with GCC 9.2
    set(TF_COMMAND make -j4 TARGET=generic-aarch64 TARGET_ARCH=armv8-a -C ${CMAKE_BINARY_DIR}/tensorflow/src/tf -f tensorflow/lite/tools/make/Makefile CC=${CROSS_PREFIX}g++ CXX=${CROSS_PREFIX}g++ AR=${CROSS_PREFIX}ar CFLAGS=-fpermissive lib)
else()
    message(FATAL_ERROR "Not implemented for this ARCH. Should be easy to extend. Just look above..")
endif()
message("System ARCH is ${CMAKE_SYSTEM_PROCESSOR} ${EDGE_ARCH}")

include(ExternalProject)

ExternalProject_Add(absl_src
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp
    GIT_PROGRESS 1
    GIT_SHALLOW 1
    PREFIX "absl"
    PATCH_COMMAND sed -i /ABSL_RANDOM_HWAES_X64_FLAGS/d
    ${CMAKE_BINARY_DIR}/absl/src/absl_src/absl/copts/AbseilConfigureCopts.cmake
    && sed -i /random/d ${CMAKE_BINARY_DIR}/absl/src/absl_src/absl/CMakeLists.txt
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/absl_install -DCMAKE_INSTALL_LIBDIR=lib 
    BUILD_BYPRODUCTS libabsl_synchronization.a libabsl_symbolize.a
)
set(ABSL_LIBS absl_synchronization absl_stacktrace absl_symbolize absl_demangle_internal absl_debugging_internal absl_dynamic_annotations absl_time absl_time_zone absl_graphcycles_internal absl_failure_signal_handler absl_malloc_internal absl_base absl_spinlock_wait)

ExternalProject_Add(glogsrc
    GIT_REPOSITORY https://github.com/google/glog
    GIT_PROGRESS 1
    GIT_SHALLOW 1
    CMAKE_ARGS -DWITH_GFLAGS=OFF
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/glog_install
    -DCMAKE_INSTALL_LIBDIR=lib
    PREFIX "glog_src"
    BUILD_BYPRODUCTS  libglog.a
)
set(LIBGLOG ${CMAKE_BINARY_DIR}/glog_install/lib/libglog.a)

ExternalProject_Add(benchmark_src
    GIT_REPOSITORY https://github.com/google/benchmark
    GIT_PROGRESS 1
    GIT_SHALLOW 1
    CMAKE_ARGS "-DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON -DBENCHMARK_ENABLE_TESTING=OFF -DBENCHMARK_ENABLE_INSTALL=OFF"
    PREFIX "benchmark_src"
    BUILD_BYPRODUCTS   libgtest.a libbenchmark.a libbenchmark_main.a
    INSTALL_COMMAND cp -f ${CMAKE_BINARY_DIR}/benchmark_src/src/benchmark_src-build/src/libbenchmark.a ${CMAKE_BINARY_DIR}/ &&
    cp -f ${CMAKE_BINARY_DIR}/benchmark_src/src/benchmark_src-build/lib/libgtest.a ${CMAKE_BINARY_DIR}/ &&
    cp -f ${CMAKE_BINARY_DIR}/benchmark_src/src/benchmark_src-build/lib/libgmock.a ${CMAKE_BINARY_DIR}/
)

ExternalProject_Add(tf
    GIT_REPOSITORY https://github.com/tensorflow/tensorflow
    GIT_TAG ${TF_SHA}
    PREFIX "tensorflow"
    CONFIGURE_COMMAND ./tensorflow/lite/tools/make/download_dependencies.sh
    BUILD_IN_SOURCE 1
    BUILD_COMMAND ${TF_COMMAND}
    BUILD_BYPRODUCTS libtensorflow-lite.a ${CMAKE_BINARY_DIR}/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c ${CMAKE_BINARY_DIR}/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc
    INSTALL_COMMAND cp -f ${CMAKE_BINARY_DIR}/tensorflow/src/tf/tensorflow/lite/tools/make/gen/${TF_INSTALL_PREFIX}/lib/libtensorflow-lite.a ${CMAKE_BINARY_DIR}/
)

set(TF_ET_SRC_LIB "${CMAKE_SOURCE_DIR}/../../libedgetpu/direct/${EDGE_ARCH}/libedgetpu.so.1.0")
set(TF_ET_LIB edgetpu)
set(TF_LITE_LIB "${CMAKE_BINARY_DIR}/libtensorflow-lite.a")
include_directories(${CMAKE_SOURCE_DIR}/../..)
include_directories(${CMAKE_SOURCE_DIR}/../../libedgetpu)
include_directories(${CMAKE_BINARY_DIR}/tensorflow/src/tf/)
include_directories(${CMAKE_BINARY_DIR}/benchmark_src/src/benchmark_src/include/)
include_directories(${CMAKE_BINARY_DIR}/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/absl/)
include_directories(${CMAKE_BINARY_DIR}/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/flatbuffers/include/)
include_directories(${CMAKE_BINARY_DIR}/glog_install/include)
include_directories(${CMAKE_BINARY_DIR}/benchmark_src/src/benchmark_src-build/third_party/googletest/src/googletest/include/)
link_directories(${CMAKE_BINARY_DIR})
link_directories(${CMAKE_BINARY_DIR}/absl_install/lib)
link_directories(${CMAKE_BINARY_DIR}/glog_install/lib)

if(NOT DOWNLOAD_HAPPENED)
  execute_process(
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMAND ln -s ${TF_ET_SRC_LIB} ${CMAKE_BINARY_DIR}/libedgetpu.so
  COMMAND ln -s ${TF_ET_SRC_LIB} ${CMAKE_BINARY_DIR}/libedgetpu.so.1
  COMMAND wget https://dl.google.com/coral/canned_models/mobilenet_v2_1.0_224_quant_edgetpu.tflite
  COMMAND wget http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz
  COMMAND tar xfz mobilenet_v2_1.0_224_quant.tgz
  COMMAND wget https://github.com/google-coral/edgetpu/raw/master/test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite
  COMMAND wget https://github.com/google-coral/edgetpu/raw/master/test_data/mobilenet_v2_1.0_224_inat_bird_quant.tflite
  COMMAND wget https://github.com/google-coral/edgetpu/raw/master/test_data/inat_bird_labels.txt
  COMMAND wget https://coral.withgoogle.com/static/docs/images/parrot.jpg
  COMMAND convert parrot.jpg parrot.bmp
  )
  set(DOWNLOAD_HAPPENED TRUE CACHE BOOL "Has the download happened?" FORCE)
endif()

add_library(error_reporter
  error_reporter.cc
  error_reporter.h)
target_link_libraries(error_reporter ${TF_LITE_LIB})
add_dependencies(error_reporter tf benchmark_src)

add_library(test_utils
  test_utils.cc
  test_utils.h)
target_link_libraries(test_utils utils basic_engine detection_engine error_reporter classification_engine gtest ${TF_ET_LIB} ${TF_LITE_LIB})
add_dependencies(test_utils utils basic_engine detection_engine classification_engine tf)

add_library(version
  version.cc
  version.h)
target_link_libraries(version ${TF_LITE_LIB})
add_dependencies(version tf)

add_library(utils
  utils.cc
  utils.h)
target_link_libraries(utils error_reporter ${TF_LITE_LIB} ${TF_ET_LIB})
add_dependencies(utils error_reporter tf)

add_library(fake_op
  fake_op.cc
  fake_op.h)
target_link_libraries(fake_op ${TF_LITE_LIB})
add_dependencies(fake_op tf)

add_library(basic_engine_native
  basic/basic_engine_native.cc
  basic/basic_engine_native.h)
target_link_libraries(basic_engine_native ${TF_ET_LIB} posenet_decoder_op error_reporter edgetpu_resource_manager ${TF_LITE_LIB})
add_dependencies(basic_engine_native tf)

add_library(basic_engine
  basic/basic_engine.cc
  basic/basic_engine.h)
target_link_libraries(basic_engine basic_engine_native glog ${TF_LITE_LIB} ${TF_ET_LIB} ${ABSL_LIBS})
add_dependencies(basic_engine tf glogsrc absl_src)

add_library(edgetpu_resource_manager
  basic/edgetpu_resource_manager.cc
  basic/edgetpu_resource_manager.h)
target_link_libraries(edgetpu_resource_manager error_reporter)
add_dependencies(edgetpu_resource_manager tf)

add_library(classification_engine
  classification/engine.cc
  classification/engine.h)
target_link_libraries(classification_engine basic_engine ${TF_LITE_LIB})
add_dependencies(classification_engine tf)

add_library(detection_engine
  detection/engine.cc
  detection/engine.h)
target_link_libraries(detection_engine basic_engine ${TF_LITE_LIB})
add_dependencies(detection_engine tf)

add_library(model_utils
  examples/model_utils.cc
  examples/model_utils.h)
target_link_libraries(model_utils ${TF_LITE_LIB})
add_dependencies(model_utils tf)

add_library(label_utils
  examples/label_utils.cc
  examples/label_utils.h)
target_link_libraries(label_utils ${TF_LITE_LIB})
add_dependencies(label_utils tf)


add_library(learning_utils
  learn/utils.cc
  learn/utils.h)
target_link_libraries(learning_utils error_reporter utils ${TF_LITE_LIB})
add_dependencies(learning_utils tf)

add_library(imprinting_engine
  learn/imprinting/engine.cc
  learn/imprinting/engine.h)
target_link_libraries(imprinting_engine engine_native ${TF_LITE_LIB})
add_dependencies(imprinting_engine tf)

add_library(engine_native
  learn/imprinting/engine_native.cc
  learn/imprinting/engine_native.h)
target_link_libraries(engine_native error_reporter basic_engine_native learning_utils ${TF_LITE_LIB})
add_dependencies(engine_native tf)

add_library(posenet_decoder_op
  posenet/posenet_decoder_op.cc
  posenet/posenet_decoder_op.h)
target_link_libraries(posenet_decoder_op posenet_decoder ${TF_LITE_LIB})
add_library(posenet_decoder
  posenet/posenet_decoder.cc
  posenet/posenet_decoder.h)
add_dependencies(posenet_decoder_op tf)

add_library(tflite_graph_util
  tools/tflite_graph_util.cc
  tools/tflite_graph_util.h)
target_link_libraries(tflite_graph_util utils ${TF_LITE_LIB})
add_dependencies(tflite_graph_util tf)

add_executable(classify_image
  examples/classify_image.cc
  ${CMAKE_BINARY_DIR}/tensorflow/src/tf/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c
  ${CMAKE_BINARY_DIR}/tensorflow/src/tf/tensorflow/lite/tools/optimize/sparsity/format_converter.cc
)
target_link_libraries(classify_image test_utils benchmark label_utils model_utils
	classification_engine ${ABSL_LIBS} ${TF_LITE_LIB} ${TF_ET_LIB})
add_dependencies(classify_image tf test_utils benchmark_src absl_src)

message("You can build with ninja (or make) and then run...")
message("sudo LD_LIBRARY_PATH=\".\" ./classify_image -image_path parrot.bmp -labels_path inat_bird_labels.txt -model_path mobilenet_v2_1.0_224_quant_edgetpu.tflite")
