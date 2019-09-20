#include "src/cpp/tools/tflite_graph_util.h"

#include <map>
#include <vector>

#include "glog/logging.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace coral {
namespace tools {

namespace {

using flatbuffers::FlatBufferBuilder;
using flatbuffers::Offset;
using flatbuffers::Vector;

// Appends clones of model buffers to vector.
void CloneBuffers(const tflite::Model& model,
                  std::vector<Offset<tflite::Buffer>>* buffer_vector,
                  FlatBufferBuilder* builder) {
  CHECK(buffer_vector);
  CHECK(builder);
  for (int i = 0; i < model.buffers()->size(); ++i) {
    auto* buffer = model.buffers()->Get(i);
    if (buffer->data() == nullptr) {
      // Many transient tensors don't have data in the flatbuffer. Their
      // buffers will be allocated by the interpreter at run-time.
      VLOG(1) << "Buffer " << i << " is empty.";
      buffer_vector->push_back(tflite::CreateBuffer(*builder));
    } else {
      tflite::BufferT buffer_t;
      buffer->UnPackTo(&buffer_t);
      VLOG(1) << "Buffer " << i << " size in bytes: " << buffer_t.data.size();
      Offset<Vector<uint8_t>> data_buffer =
          builder->CreateVector(buffer_t.data.data(), buffer_t.data.size());
      buffer_vector->push_back(tflite::CreateBuffer(*builder, data_buffer));
    }
  }
}

// Appends clones of model tensors to vector, assuming the model has only one
// subgraph.
// |tensor_buffer_start_offset| specifies the offset to be added to the buffer
// index of all tensors of this model.
// |tensor_name_to_buffer_index_map| is the tensor to buffer index map, and
// |tensor_name_to_tensor_index_map| is the tensor name to tensor index map.
// Both maps will be updated by this function.
void CloneTensors(
    const tflite::Model& model, uint32_t tensor_buffer_start_offset,
    std::vector<Offset<tflite::Tensor>>* tensor_vector,
    FlatBufferBuilder* builder,
    std::map<std::string, uint32_t>* tensor_name_to_buffer_index_map,
    std::map<std::string, int32_t>* tensor_name_to_tensor_index_map) {
  CHECK(tensor_vector);
  CHECK(builder);
  CHECK(tensor_name_to_buffer_index_map);
  CHECK(tensor_name_to_tensor_index_map);

  const auto* subgraphs = model.subgraphs();
  const auto* tensors = subgraphs->Get(0)->tensors();
  for (int i = 0; i < tensors->size(); ++i) {
    const auto* tensor = tensors->Get(i);
    CHECK(tensor);
    tflite::TensorT tensor_t;
    tensor->UnPackTo(&tensor_t);
    if (tensor_name_to_buffer_index_map->count(tensor_t.name) > 0) {
      VLOG(1) << "Tensor " << tensor_t.name << " already exists.";
      continue;
    }

    const auto* q_param = tensor->quantization();
    const auto& q_param_t = tensor_t.quantization;

    auto new_q_param = tflite::CreateQuantizationParameters(
        *builder, q_param->min() ? builder->CreateVector(q_param_t->min) : 0,
        q_param->max() ? builder->CreateVector(q_param_t->max) : 0,
        q_param->scale() ? builder->CreateVector(q_param_t->scale) : 0,
        q_param->zero_point() ? builder->CreateVector(q_param_t->zero_point)
                              : 0);

    // Update tensor name to buffer index map. Note that buffer index must be
    // recalcualted.
    const uint32_t buffer_index = tensor_buffer_start_offset + tensor_t.buffer;
    (*tensor_name_to_buffer_index_map)[tensor_t.name] = buffer_index;
    VLOG(1) << "Tensor " << tensor_vector->size() << " name: " << tensor_t.name
            << ", buffer index from " << tensor_t.buffer << " to "
            << buffer_index;

    // Update tensor name to tensor index map.
    CHECK_EQ(tensor_name_to_tensor_index_map->count(tensor_t.name), 0);
    (*tensor_name_to_tensor_index_map)[tensor_t.name] = tensor_vector->size();

    tensor_vector->push_back(tflite::CreateTensor(
        *builder, builder->CreateVector(tensor_t.shape), tensor_t.type,
        /*buffer=*/buffer_index, builder->CreateString(tensor_t.name),
        new_q_param));
  }
}

// Recalcuates tensor indices given a new tensor name to tensor index map.
std::vector<int32_t> RecalcualteTensorIndices(
    const std::vector<int32_t>& old_tensor_indices, const tflite::Model& model,
    const std::map<std::string, int32_t>& new_tensor_name_to_tensor_index_map) {
  const auto* subgraphs = model.subgraphs();
  const auto* old_tensor_vector = subgraphs->Get(0)->tensors();
  std::vector<int32_t> result;
  result.reserve(old_tensor_indices.size());
  for (uint32_t i : old_tensor_indices) {
    const auto* tensor = old_tensor_vector->Get(i);
    const std::string tensor_name = tensor->name()->str();
    CHECK_GT(new_tensor_name_to_tensor_index_map.count(tensor_name), 0);
    const uint32_t new_index =
        new_tensor_name_to_tensor_index_map.at(tensor_name);
    VLOG(1) << "Change tensor " << tensor_name << " index from " << i << " to "
            << new_index;
    result.push_back(new_index);
  }
  return result;
}

// Appends clones of model operator codes to vector. No deduping.
void CloneOperatorCodes(
    const tflite::Model& model,
    std::vector<Offset<tflite::OperatorCode>>* opcode_vector,
    FlatBufferBuilder* builder) {
  CHECK(opcode_vector);
  CHECK(builder);
  for (int i = 0; i < model.operator_codes()->size(); ++i) {
    const auto* opcode = model.operator_codes()->Get(i);
    tflite::OperatorCodeT opcode_t;
    opcode->UnPackTo(&opcode_t);
    opcode_vector->push_back(tflite::CreateOperatorCode(
        *builder, opcode_t.builtin_code,
        opcode->custom_code() ? builder->CreateString(opcode_t.custom_code) : 0,
        opcode_t.version));
  }
}

// Appends clones of model operators to vector.
// |opcode_index_start_offset| specifies the offset to be added to the opcode
// index of all operators.
void CloneOperators(
    const tflite::Model& model, uint32_t opcode_index_start_offset,
    const std::map<std::string, int32_t>& tensor_name_to_tensor_index_map,
    std::vector<Offset<tflite::Operator>>* op_vector,
    FlatBufferBuilder* builder) {
  CHECK(op_vector);
  CHECK(builder);
  const auto* ops = model.subgraphs()->Get(0)->operators();
  for (int i = 0; i < ops->size(); ++i) {
    const auto* op = ops->Get(i);
    CHECK(op->inputs());
    CHECK(op->outputs());

    tflite::OperatorT op_t;
    op->UnPackTo(&op_t);
    uint32_t new_opcode_index = op_t.opcode_index + opcode_index_start_offset;
    VLOG(1) << "Change operator " << i << " opcode index from "
            << op_t.opcode_index << " to " << new_opcode_index;

    // Recalculate input and output indices of this operator.
    Offset<Vector<int32_t>> new_input_index_vector =
        builder->CreateVector<int32_t>(RecalcualteTensorIndices(
            op_t.inputs, model, tensor_name_to_tensor_index_map));
    Offset<Vector<int32_t>> new_output_index_vector =
        builder->CreateVector<int32_t>(RecalcualteTensorIndices(
            op_t.outputs, model, tensor_name_to_tensor_index_map));

    const auto builtin_options_type = op_t.builtin_options.type;

    const auto custom_options_format = op_t.custom_options_format;

    op_vector->push_back(tflite::CreateOperator(
        *builder, new_opcode_index, new_input_index_vector,
        new_output_index_vector, builtin_options_type,
        op->builtin_options() ? op_t.builtin_options.Pack(*builder) : 0,
        op->custom_options() ? builder->CreateVector(op_t.custom_options.data(),
                                                     op_t.custom_options.size())
                             : 0,
        custom_options_format));
  }
}

// Concanetate two tflite models, assuming each model has only one subgraph.
// |builder| contains the result model.
void ConcatModels(const tflite::Model& model0, const tflite::Model& model1,
                  flatbuffers::FlatBufferBuilder* builder) {
  CHECK(builder);

  CHECK(model0.subgraphs());
  CHECK_EQ(model0.subgraphs()->size(), 1);
  CHECK(model1.subgraphs());
  CHECK_EQ(model1.subgraphs()->size(), 1);

  // Merge all buffers.
  const int num_model0_buffers = model0.buffers()->size();
  const int num_model1_buffers = model1.buffers()->size();
  VLOG(1) << "model0 # buffers: " << num_model0_buffers
          << ", model1 # buffers: " << num_model1_buffers;
  std::vector<Offset<tflite::Buffer>> buffer_vector;
  CloneBuffers(model0, &buffer_vector, builder);
  CloneBuffers(model1, &buffer_vector, builder);
  VLOG(1) << "merged # buffers: " << buffer_vector.size();

  // Merge all tensors.
  const tflite::SubGraph& subgraph0 = *(*model0.subgraphs())[0];
  const tflite::SubGraph& subgraph1 = *(*model1.subgraphs())[0];
  const int num_model0_tensors = subgraph0.tensors()->size();
  const int num_model1_tensors = subgraph1.tensors()->size();
  VLOG(1) << "model0 # tensors: " << num_model0_tensors
          << ", model1 # tensors: " << num_model1_tensors;
  std::vector<Offset<tflite::Tensor>> tensor_vector;
  std::map<std::string, uint32_t> tensor_name_to_buffer_index_map;
  std::map<std::string, int32_t> tensor_name_to_tensor_index_map;
  CloneTensors(model0, /*tensor_buffer_start_offset=*/0, &tensor_vector,
               builder, &tensor_name_to_buffer_index_map,
               &tensor_name_to_tensor_index_map);
  CloneTensors(model1, /*tensor_buffer_start_offset=*/num_model0_buffers,
               &tensor_vector, builder, &tensor_name_to_buffer_index_map,
               &tensor_name_to_tensor_index_map);
  VLOG(1) << "merged # tensors: " << tensor_vector.size();
  CHECK_EQ(tensor_name_to_buffer_index_map.size(), tensor_vector.size());

  // Create vectors of input and output tensors indices.
  tflite::SubGraphT subgraph0_t, subgraph1_t;
  subgraph0.UnPackTo(&subgraph0_t);
  subgraph1.UnPackTo(&subgraph1_t);
  std::vector<int32_t> inputs = RecalcualteTensorIndices(
      subgraph0_t.inputs, model0, tensor_name_to_tensor_index_map);
  std::vector<int32_t> outputs = RecalcualteTensorIndices(
      subgraph1_t.outputs, model1, tensor_name_to_tensor_index_map);

  // Merge operator codes.
  const int num_model0_opcodes = model0.operator_codes()->size();
  const int num_model1_opcodes = model1.operator_codes()->size();
  VLOG(1) << "model0 # opcodes: " << num_model0_opcodes
          << ", model1 # opcodes: " << num_model1_opcodes;
  std::vector<Offset<tflite::OperatorCode>> opcode_vector;
  CloneOperatorCodes(model0, &opcode_vector, builder);
  CloneOperatorCodes(model1, &opcode_vector, builder);
  CHECK_EQ(num_model0_opcodes + num_model1_opcodes, opcode_vector.size());

  // Merge operators.
  const int num_model0_ops = subgraph0.operators()->size();
  const int num_model1_ops = subgraph1.operators()->size();
  VLOG(1) << "model0 # ops: " << num_model0_ops
          << ", model1 # ops: " << num_model1_ops;
  std::vector<Offset<tflite::Operator>> op_vector;
  CloneOperators(model0, /*opcode_index_start_offset=*/0,
                 tensor_name_to_tensor_index_map, &op_vector, builder);
  CloneOperators(model1, /*opcode_index_start_offset=*/num_model0_opcodes,
                 tensor_name_to_tensor_index_map, &op_vector, builder);
  CHECK_EQ(num_model0_ops + num_model1_ops, op_vector.size());

  Offset<Vector<int32_t>> merged_inputs =
      builder->CreateVector<int32_t>(inputs);
  Offset<Vector<int32_t>> merged_outputs =
      builder->CreateVector<int32_t>(outputs);
  Offset<Vector<Offset<tflite::Tensor>>> merged_tensors =
      builder->CreateVector(tensor_vector);
  Offset<Vector<Offset<tflite::Operator>>> merged_ops =
      builder->CreateVector(op_vector);
  Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      *builder, merged_tensors, merged_inputs, merged_outputs, merged_ops,
      (subgraph0.name() ? builder->CreateString(subgraph0.name()->str()) : 0));
  Offset<Vector<Offset<tflite::SubGraph>>> merged_subgraphs =
      builder->CreateVector<Offset<tflite::SubGraph>>({subgraph});

  Offset<Vector<Offset<tflite::Buffer>>> merged_buffers =
      builder->CreateVector(buffer_vector);
  Offset<Vector<Offset<tflite::OperatorCode>>> merged_opcodes =
      builder->CreateVector(opcode_vector);
  auto merged_model = tflite::CreateModel(
      *builder, model0.version(), merged_opcodes, merged_subgraphs,
      (model0.description() ? builder->CreateString(model0.description()->str())
                            : 0),
      merged_buffers);

  // TODO: find out whether need to set metadata_buffer.

  tflite::FinishModelBuffer(*builder, merged_model);
}
}  // namespace

void ConcatTfliteModels(const std::string& model0_path,
                        const std::string& model1_path,
                        const std::string& output_path) {
  std::string model0_contents;

  ReadFileOrDie(model0_path, &model0_contents);
  const tflite::Model* model0 = tflite::GetModel(model0_contents.data());
  CHECK_EQ(model0->subgraphs()->size(), 1);

  std::string model1_contents;
  ReadFileOrDie(model1_path, &model1_contents);
  const tflite::Model* model1 = tflite::GetModel(model1_contents.data());
  CHECK_EQ(model1->subgraphs()->size(), 1);

  // Merge the two models.
  flatbuffers::FlatBufferBuilder builder(/*initial_size=*/1024 * 1024);
  ConcatModels(*model0, *model1, &builder);

  // Write result model to file.
  const uint8_t* buffer = builder.GetBufferPointer();
  int size = builder.GetSize();
  WriteFileOrDie(std::string(reinterpret_cast<const char*>(buffer), size),
                 output_path);
}

}  // namespace tools
}  // namespace coral
