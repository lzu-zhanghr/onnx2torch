import io
from typing import Any, Union, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchmetrics
import torchvision.datasets
from onnx import ModelProto
from onnx2torch.converter import convert
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

from test import DATASETS_DIR, logger

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CIFAR10_MEAN = (0.491, 0.482, 0.447)
_CIFAR10_STD = (0.202, 0.199, 0.201)


def create_imagenet_test(  # pylint: disable=missing-function-docstring
    root: Any,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> Tuple[int, int, DataLoader]:
    resolution: int = 224
    transform = transforms.Compose(
        [
            transforms.Resize(size=256),
            transforms.CenterCrop(size=resolution),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )

    dataset = datasets.ImageFolder(
        root=root,
        transform=transform,
    )
    classes: int = 1000
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return classes, resolution, dataloader


def create_cifar10_test(  # pylint: disable=missing-function-docstring
    root: Any,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> Tuple[int, int, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )

    dataset = torchvision.datasets.CIFAR10(
        root, train=False, transform=transform, download=False
    )
    classes: int = 10
    resolution: int = 32
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return classes, resolution, dataloader


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


def check_model(  # pylint: disable=missing-function-docstring,unused-argument
    model: torch.nn.Module,
    batch_size: int,
    dataset: str,
    model_name: str,
    opset_version: int = 13,
) -> None:
    if not DATASETS_DIR.exists():
        DATASETS_DIR.mkdir()
    if dataset == "imagenet":
        root = DATASETS_DIR / "ILSVRC2012_img_val"
        classes, resolution, dataloader = create_imagenet_test(
            root, batch_size, num_workers=4
        )
    else:
        root = DATASETS_DIR
        classes, resolution, dataloader = create_cifar10_test(
            root, batch_size, num_workers=4
        )

    onnx_model, torch_model = get_model(model, batch_size, resolution, opset_version)

    ort_session = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )

    onnx_top1 = torchmetrics.Accuracy(num_classes=classes)
    onnx_top5 = torchmetrics.Accuracy(num_classes=classes, top_k=5)

    torch_top1 = torchmetrics.Accuracy(num_classes=classes)
    torch_top5 = torchmetrics.Accuracy(num_classes=classes, top_k=5)

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
        f"dataset:{dataset:8s}; model:{model_name:30s}; mse: {error:.2e}; err: {err:5d}; "
        f"onnx_top1_acc: {onnx_top1_acc:.6f},  torch_top1_acc: {torch_top1_acc:.6f}; "
        f" onnx_top5_acc: {onnx_top5_acc:.6f}, torch_top5_acc: {torch_top5_acc:.6f}; "
    )
    assert onnx_top1_acc == torch_top1_acc and onnx_top5_acc == torch_top5_acc
