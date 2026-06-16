from pathlib import Path

from nl2hdl.mlir import analyze_mlir


def test_analyze_mlir_reports_supported_ops(tmp_path: Path):
    mlir = tmp_path / "graph.mlir"
    mlir.write_text(
        '''
module {
  func.func @main_graph(%arg0: tensor<1x4xf32>) -> (tensor<1x3xf32>) {
    %0 = "onnx.Gemm"(%arg0, %w, %b) {onnx_node_name = "dense"} : (tensor<1x4xf32>, tensor<3x4xf32>, tensor<3xf32>) -> tensor<1x3xf32>
    %1 = "onnx.Relu"(%0) : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }
}
''',
        encoding="utf-8",
    )
    analysis = analyze_mlir(mlir)
    assert analysis.entry == "main_graph"
    assert [op.op_type for op in analysis.ops] == ["Gemm", "Relu"]
    assert analysis.unsupported_ops == tuple()
    assert analysis.to_dict()["op_partition"] == {"gemm": ["Gemm"], "non_gemm": ["Relu"]}


def test_analyze_mlir_reports_unsupported_ops(tmp_path: Path):
    mlir = tmp_path / "graph.mlir"
    mlir.write_text(
        '''
module {
  func.func @main_graph(%arg0: tensor<1x4xf32>) -> (tensor<1x4xf32>) {
    %0 = "onnx.Conv"(%arg0, %w) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
  }
}
''',
        encoding="utf-8",
    )
    analysis = analyze_mlir(mlir)
    assert analysis.unsupported_ops == ("Conv",)


def test_analyze_mlir_uses_loc_as_node_name_when_onnx_node_name_missing(tmp_path: Path):
    mlir = tmp_path / "graph.mlir"
    mlir.write_text(
        '''
module {
  func.func @main_graph(%arg0: tensor<1x4xf32>) -> (tensor<1x4xf32>) {
    %0 = "onnx.MatMul"(%arg0, %w) : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32> loc("/model/layers.0/self_attn/q_proj/MatMul")
    return %0 : tensor<1x4xf32>
  }
}
''',
        encoding="utf-8",
    )
    analysis = analyze_mlir(mlir)
    assert analysis.ops[0].node_name == "/model/layers.0/self_attn/q_proj/MatMul"


def test_analyze_mlir_invalid_text_has_no_entry_or_ops(tmp_path: Path):
    mlir = tmp_path / "bad.mlir"
    mlir.write_text("not mlir", encoding="utf-8")
    analysis = analyze_mlir(mlir)
    assert analysis.entry is None
    assert analysis.ops == tuple()
