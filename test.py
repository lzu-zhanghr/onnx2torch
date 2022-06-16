from onnx2torch import convert
import onnxruntime as ort
import onnx
import torch
from torch import nn

model_path ='D:/workspace/onnx2pytorch/tests/.tmp/models/bertsquad-12.onnx'
onnx_model = onnx.load_model(str(model_path))
# graph = onnx_model.graph
# ns = []
# for node in graph.node:
#     ns.append(node.op_type)
# print(ns)
torch_model = convert(onnx_model)
nn.Embedding
nn.TransformerEncoderLayer

# from torch import nn
# import torch.functional as F
# from torchtext.models import RobertaEncoderConf, RobertaModelBundle, RobertaClassificationHead
# model_weights_path = "https://download.pytorch.org/models/text/xlmr.base.encoder.pt"
# encoder_conf = RobertaEncoderConf(vocab_size=250002)
# classifier_head = RobertaClassificationHead(num_classes=2, input_dim=768)
# model = RobertaModelBundle.build_model(encoder_conf=encoder_conf, head=classifier_head, checkpoint=model_weights_path)

