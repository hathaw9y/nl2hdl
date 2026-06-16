import numpy as np

from nl2hdl.config import OptimizationConfig
from nl2hdl.graph import DenseLayer, ModelGraph
from nl2hdl.pruning import apply_pruning


def test_magnitude_pruning_zeros_small_weights():
    graph = ModelGraph(
        name="g",
        input_name="x",
        input_size=2,
        output_name="y",
        layers=[
            DenseLayer(
                name="dense_0",
                input_name="x",
                output_name="y",
                weights=np.array([[0.05, -0.2], [0.3, -0.01]], dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            )
        ],
    )
    pruned, report = apply_pruning(
        graph,
        OptimizationConfig(pruning="magnitude_unstructured", pruning_threshold=0.05),
    )
    assert report["zeroed_weights"] == 2
    np.testing.assert_allclose(pruned.layers[0].weights, np.array([[0.0, -0.2], [0.3, 0.0]], dtype=np.float32))
