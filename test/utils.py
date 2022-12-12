import io
from typing import Any, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchmetrics
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
    opset_version: int,
) -> Union[ModelProto, torch.nn.Module]:
    inputs = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

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


def check_model(  # pylint: disable=missing-function-docstring,unused-argument
    model: torch.nn.Module, batch_size, model_name: str, opset_version: int = 13
) -> None:

    onnx_model, torch_model = get_model(model, batch_size, opset_version)
    if not DATASETS_DIR.exists():
        DATASETS_DIR.mkdir()
    root = DATASETS_DIR / "ILSVRC2012_img_val"
    dataloader = create_imagenet_test(root, batch_size, num_workers=4)
    ort_session = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )

    onnx_top1 = torchmetrics.Accuracy(num_classes=1000)
    onnx_top5 = torchmetrics.Accuracy(num_classes=1000, top_k=5)

    torch_top1 = torchmetrics.Accuracy(num_classes=1000)
    torch_top5 = torchmetrics.Accuracy(num_classes=1000, top_k=5)

    mse = torchmetrics.MeanSquaredError()
    err = 0

    for _, (image, target) in enumerate(dataloader):
        onnx_input = {"input": image.detach().cpu().numpy()}
        torch_input = image

        onnx_output = ort_session.run(
            output_names=None,
            input_feed=onnx_input,
        )
        onnx_output = torch.from_numpy(np.array(onnx_output)).squeeze()
        torch_output = torch_model(torch_input)

        onnx_top1.update(onnx_output, target)
        onnx_top5.update(onnx_output, target)

        torch_top1.update(torch_output, target)
        torch_top5.update(torch_output, target)

        mse.update(torch_output, onnx_output)
        err += torch.count_nonzero(torch_output.argmax(1) - onnx_output.argmax(1))

    onnx_top1_acc, onnx_top5_acc = (
        onnx_top1.compute(),
        onnx_top5.compute(),
    )
    torch_top1_acc, torch_top5_acc = (
        torch_top1.compute(),
        torch_top5.compute(),
    )

    error = mse.compute()

    logger.info(
        f"model:{model_name:18s}; mse: {error:.2e}; err: {err:5d}; "
        f"onnx_top1_acc: {onnx_top1_acc:.6f},  torch_top1_acc: {torch_top1_acc:.6f}; "
        f" onnx_top5_acc: {onnx_top5_acc:.6f}, torch_top5_acc: {torch_top5_acc:.6f}; "
    )
    assert onnx_top1_acc == torch_top1_acc and onnx_top5_acc == torch_top5_acc
