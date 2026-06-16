module {
  func.func @llama_decoder_block(%arg0: tensor<1x4xi8>) -> (tensor<1x4xi8>) {
    %0 = "llm.RMSNorm"(%arg0) {onnx_node_name = "input_layernorm"} : (tensor<1x4xi8>) -> tensor<1x4xi8>
    %1 = "onnx.MatMul"(%0, %q_weight) {onnx_node_name = "q_proj"} : (tensor<1x4xi8>, tensor<4x4xi4>) -> tensor<1x4xi32>
    %2 = "onnx.MatMul"(%0, %k_weight) {onnx_node_name = "k_proj"} : (tensor<1x4xi8>, tensor<4x4xi4>) -> tensor<1x4xi32>
    %3 = "llm.RoPE"(%1) {onnx_node_name = "rope_q"} : (tensor<1x4xi32>) -> tensor<1x4xi32>
    %4 = "llm.Softmax"(%3, %2) {onnx_node_name = "attention_softmax"} : (tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4xi8>
    %5 = "onnx.MatMul"(%4, %o_weight) {onnx_node_name = "o_proj"} : (tensor<1x4xi8>, tensor<4x4xi4>) -> tensor<1x4xi32>
    %6 = "llm.RMSNorm"(%5) {onnx_node_name = "post_attention_layernorm"} : (tensor<1x4xi32>) -> tensor<1x4xi8>
    %7 = "onnx.MatMul"(%6, %up_weight) {onnx_node_name = "up_proj"} : (tensor<1x4xi8>, tensor<8x4xi4>) -> tensor<1x8xi32>
    %8 = "llm.SiLU"(%7) {onnx_node_name = "silu"} : (tensor<1x8xi32>) -> tensor<1x8xi32>
    %9 = "onnx.MatMul"(%8, %down_weight) {onnx_node_name = "down_proj"} : (tensor<1x8xi32>, tensor<4x8xi4>) -> tensor<1x4xi32>
    return %9 : tensor<1x4xi32>
  }
}
