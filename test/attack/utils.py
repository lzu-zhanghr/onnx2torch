import io
from typing import Any, Union

import numpy as np
import onnx
from test.attack.method import RayS
import onnxruntime as ort
import torch
from torch import nn
from onnx import ModelProto
from onnx2torch.converter import convert
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from test import DATASETS_DIR, logger

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def create_imagenet_test(  # pylint: disable=missing-function-docstring
    root: Any,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )

    dataset = datasets.ImageFolder(
        root=root,
        transform=transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataloader


def get_model(  # pylint: disable=missing-function-docstring,unused-argument
    model: torch.nn.Module,
    batch_size: int,
    resolution: int,
    opset_version: int,
) -> Union[ModelProto, torch.nn.Module]:
    inputs = torch.randn(batch_size, 3, resolution, resolution, requires_grad=True)

    with io.BytesIO() as tmp_file:
        torch.onnx.export(
            model=model,
            args=inputs,
            f=tmp_file,
            export_params=True,
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        onnx_model = onnx.load_from_string(tmp_file.getvalue())
    torch_model = convert(onnx_model)

    return onnx_model, torch_model


class Network(nn.Module):
    def __init__(self, model, n_class=10, im_mean=None, im_std=None):
        super(Network, self).__init__()
        self.model = model
        self.num_queries = 0
        self.im_mean = im_mean
        self.im_std = im_std
        self.n_class = n_class

    def forward(self, image):
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        image = self.preprocess(image)
        if isinstance(self.model, nn.Module):
            logits = self.model(image)
        else:
            onnx_input = {"input": image.numpy()}
            ort_session = ort.InferenceSession(
                self.SerializeToString(),
                providers=["CPUExecutionProvider"],
            )
            onnx_output = ort_session.run(
                output_names=None,
                input_feed=onnx_input,
            )
            logits = torch.from_numpy(onnx_output)
        return logits

    def preprocess(self, image):
        if isinstance(image, np.ndarray):
            processed = torch.from_numpy(image).type(torch.FloatTensor)
        else:
            processed = image

        if self.im_mean is not None and self.im_std is not None:
            im_mean = (
                torch.tensor(self.im_mean)
                .view(1, processed.shape[1], 1, 1)
                .repeat(processed.shape[0], 1, 1, 1)
            )
            im_std = (
                torch.tensor(self.im_std)
                .view(1, processed.shape[1], 1, 1)
                .repeat(processed.shape[0], 1, 1, 1)
            )
            processed = (processed - im_mean) / im_std
        return processed

    def predict_prob(self, image):
        with torch.no_grad():
            if len(image.size()) != 4:
                image = image.unsqueeze(0)
            image = self.preprocess(image)
            logits = self.model(image)
            self.num_queries += image.size(0)
        return logits

    def predict_label(self, image):
        logits = self.predict_prob(image)
        _, predict = torch.max(logits, 1)
        return predict


def gen_advs(  # pylint: disable=missing-function-docstring,unused-argument
    model: torch.nn.Module,
    batch_size: int,
    model_name: str,
    num: int = 10000,
    norm: str = "linf",
    onnx: bool = True,
    query: int = 10000,
    epsilon: float = 0.05,
    targeted: bool = True,
    opset_version: int = 13,
    early_stopping: bool = True,
    resolution: int = 224,
) -> None:
    root = DATASETS_DIR / "ILSVRC2012_img_val"
    dataloader = create_imagenet_test(root, batch_size, num_workers=4)

    onnx_model, torch_model = get_model(model, batch_size, resolution, opset_version)
    onnx_net = Network(
        onnx_model, n_class=1000, im_mean=_IMAGENET_MEAN, im_std=_IMAGENET_STD
    )
    torch_net = Network(
        torch_model, n_class=1000, im_mean=_IMAGENET_MEAN, im_std=_IMAGENET_STD
    )
    net = onnx_net if onnx else torch_net

    order = 2 if norm == "l2" else np.inf

    attack = RayS(net, epsilon=epsilon, order=order)

    advs = []

    adbds = []

    count = 0
    for i, (x, y) in enumerate(dataloader):
        if count >= num:
            break

        if net.predict_label(x) != y:
            continue

        if targeted:

            target = (
                np.random.randint(net.n_class) * torch.ones(y.shape, dtype=torch.long)
                if targeted
                else None
            )
        else:
            target = None

        adv_b, queries_b, adbd_b, succ_b = attack(
            x, y, target=target, query_limit=query
        )

        advs.extend(adv.numpy() for adv in adv_b)

        count += x.shape[0]
        adbds.extend([adbd.numpy() for adbd in adbd_b])

    logger.info(
        f"model: {model_name:30s}; ADBD: {np.mean(np.array(adbds), axis=0):.6f};"
    )
    np.save(
        DATASETS_DIR.parent
        / (model_name + "_" + ("onnx" if onnx else "torch") + "advs.npy"),
        np.stack(advs),
    )
