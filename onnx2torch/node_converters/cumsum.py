__all__ = ['OnnxCumSum']

import torch
from torch import nn
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import get_const_value
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import OnnxToTorchModule


class OnnxCumSum(nn.Module, OnnxToTorchModule):
    def __init__(self, axis, reverse: int = 0, exclusive: int = 0):
        self.axis = axis
        self.reverse = reverse
        self.exclusive = exclusive
        super().__init__()

    def forward(self, input_tensor: torch.Tensor):
        if self.reverse:
            raise NotImplemented('not supported reverse attribute int pytorch cumsum')
        if self.exclusive:
            raise NotImplemented('not supported exclusive attribute int pytorch cumsum ')
        out = torch.cumsum(input_tensor, dim=self.dim)
        return out

@add_converter(operation_type='CumSum', version=14)
@add_converter(operation_type='CumSum', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    node_attributes = node.attributes
    exclusive = node_attributes.get('exclusive', 0)
    reverse = node_attributes.get('reverse', 0)

    axis_name = node.input_values[1]
    try:
        axis = get_const_value(axis_name, graph) if axis_name is not None else None
    except KeyError:
        raise NotImplementedError('Dynamic value of axis is not implemented')

    return OperationConverterResult(
        torch_module=OnnxCumSum(axis, reverse, exclusive),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )