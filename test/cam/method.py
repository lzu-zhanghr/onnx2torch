import copy

import onnx
import torch
from onnx import ModelProto
from typing import Tuple
from numpy.typing import NDArray
import numpy as np
import scipy
import onnxruntime as ort
import torch.nn.functional as F
from torch.nn import Module
import tqdm


class ONNXCAM:
    def __init__(
        self,
        onnx_model: ModelProto,
        target_layer: str = None,
    ) -> None:
        self.model = onnx_model
        self.target_names = target_layer

    def _forward(self, x: NDArray) -> Tuple:
        model = copy.deepcopy(self.model)
        for node in model.graph.node:
            if node.name == self.target_names:
                for output in node.output:
                    model.graph.output.extend([onnx.ValueInfoProto(name=output)])
        ort_session = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        ort_inputs = {"input": x}
        ort_outputs = ort_session.run(output_names=None, input_feed=ort_inputs)

        ort_outputs = tuple(ort_outputs)
        return ort_outputs

    def _batch_forward(self, x: NDArray) -> NDArray:
        ort_session = ort.InferenceSession(
            self.model.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        ort_inputs = {"input": x}
        ort_outputs = ort_session.run(output_names=None, input_feed=ort_inputs)

        return ort_outputs

    def cam(self, x: NDArray, class_idx: int = None) -> NDArray:
        b, c, h, w = x.shape
        logit, activations = self._forward(x)

        b, k, u, v = activations.shape

        if class_idx is None:
            classes = np.argmax(logit, -1)
        else:
            classes = class_idx

        score_saliency_map: NDArray = np.zeros((1, 1, h, w))

        for i in tqdm.trange(k):
            saliency_map = np.expand_dims(activations[:, i, :, :], 1)
            saliency_map = F.interpolate(
                torch.from_numpy(saliency_map),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).numpy()
            norm_saliency_map = (saliency_map - saliency_map.min()) / (
                saliency_map.max() - saliency_map.min()
            )

            output = self._batch_forward(x * norm_saliency_map)[0]
            output = scipy.special.softmax(output)
            score = output[0][classes]

            score_saliency_map += score * saliency_map

        score_saliency_map = F.relu(torch.from_numpy(score_saliency_map)).numpy()
        return score_saliency_map

    def __call__(self, x: NDArray, class_idx: int = None) -> NDArray:
        return self.cam(x, class_idx)


class TorchCAM:
    def __init__(
        self,
        torch_model: Module,
        target_layer: str = None,
    ) -> None:
        self.model = torch_model.eval()
        self.target_names = target_layer
        self.activations = dict()

        def forward_hook(module, input, output) -> None:
            self.activations["value"] = output

        self.layer = self.model._modules[self.target_names]
        self.layer.register_forward_hook(forward_hook)

    def cam(self, x: NDArray, class_idx=None) -> NDArray:
        x = torch.from_numpy(x)
        b, c, h, w = x.size()

        # predication on raw input
        logit = self.model(x)

        if class_idx is None:
            classes = logit.max(1)[-1]
        else:
            classes = torch.LongTensor([class_idx])

        self.model.zero_grad()
        activations = self.activations["value"]
        b, k, u, v = activations.size()

        score_saliency_map = torch.zeros((1, 1, h, w))

        with torch.no_grad():
            for i in tqdm.trange(k):
                # upsampling
                saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
                saliency_map = F.interpolate(
                    saliency_map, size=(h, w), mode="bilinear", align_corners=False
                )

                # normalize to 0-1
                norm_saliency_map = (saliency_map - saliency_map.min()) / (
                    saliency_map.max() - saliency_map.min()
                )

                # how much increase if keeping the highlighted region
                # predication on masked input
                output = self.model(x * norm_saliency_map)
                output = F.softmax(output)
                score = output[0][classes]

                score_saliency_map += score * saliency_map

        score_saliency_map = F.relu(score_saliency_map)
        return score_saliency_map.numpy()

    def __call__(self, x, class_idx=None) -> NDArray:
        return self.cam(x, class_idx)
