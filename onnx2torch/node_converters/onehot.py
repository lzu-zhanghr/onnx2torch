__all__ = ['OnnxOneHot']


from torch import nn
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from torch.nn import functional as F
from onnx2torch.utils.common import onnx_mapping_from_node
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import get_const_value


class OnnxOneHot(nn.Module, OnnxToTorchModule):
    def __init__(self, depth, values, indices, axis=-1, non_zero_values_only=False):
        self.axis = axis
        self.depth = depth
        self.values = values
        self.indices = indices
        self.non_zero_values_only = non_zero_values_only
        super().__init__()

    def forward(self,):
        if self.non_zero_values_only:
            off_value, on_value = -1, 1
        else:
            off_value, on_value = self.values
        out = F.one_hot(self.indices.to(int), self.depth.to(int).item())
        out = out * (on_value - off_value) + off_value

        rank = len(self.indices.shape)
        if self.axis < 0:
            self.axis += rank + 1
        if not rank == self.axis:  # permute only if dim not last dimension
            order = list(range(len(self.indices.shape)))
            order.insert(self.axis, -1)
            out = out.permute(order)
        return out

@add_converter(operation_type='OneHot', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    node_attributes = node.attributes
    axis = node_attributes.get('axis', -1)

    indices_value_name = node.input_values[0]
    depth_value_name = node.input_values[1]
    values_value_name = node.input_values[2]

    try:
        indices = get_const_value(indices_value_name, graph) if indices_value_name is not None else None
        print(indices)
        depth = get_const_value(depth_value_name, graph) if depth_value_name is not None else None
        print(depth)
        values = get_const_value(values_value_name, graph) if values_value_name is not None else None
        print(values)
    except KeyError:
        raise NotImplementedError('Dynamic value is not implemented')


    return OperationConverterResult(
        torch_module=OnnxOneHot(depth, values, indices, axis),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )