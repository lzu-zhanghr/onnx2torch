import io
from typing import Union, Any, Optional
import onnx
import torch
from onnx import ModelProto
from onnx2torch.converter import convert
from numpy.typing import NDArray
from matplotlib import cm
import numpy as np

from torchvision.transforms import transforms
from PIL import Image
import torch.nn.functional as F
from method import ONNXCAM, TorchCAM


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def get_model(  # pylint: disable=missing-function-docstring,unused-argument
    model: torch.nn.Module,
    batch_size: int = 1,
    resolution: int = 224,
    opset_version: int = 13,
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


def load_img(path: str, resolution: int = 224) -> Image.Image:
    image = Image.open(path).convert("RGB").resize((resolution, resolution))
    return image


def apply_transforms(image: Any):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )

    tensor = transform(image).unsqueeze(0)

    tensor.requires_grad = False

    return tensor


def overlay_mask(
    img: Image.Image,
    mask: NDArray,
    alpha: float = 0.7,
) -> Image.Image:

    if alpha < 0 or alpha >= 1:
        raise ValueError(
            "alpha argument is expected to be of type float between 0 and 1"
        )
    overlayed_img = Image.fromarray(
        (alpha * np.asarray(img) + (1 - alpha) * mask).astype(np.uint8)
    )

    return overlayed_img


def check_model(  # pylint: disable=missing-function-docstring,unused-argument
    model: torch.nn.Module,
    model_name: str,
    layer_name: str,
    image_path: str,
    resolution: int = 224,
    opset_version: int = 13,
    class_idx: int = None,
) -> None:
    onnx_model, torch_model = get_model(model, opset_version)
    onnxcam = ONNXCAM(onnx_model, layer_name)
    torchcam = TorchCAM(torch_model, layer_name)
    img = load_img(image_path, resolution)
    x = apply_transforms(img).numpy()

    onnx_mask = onnxcam.cam(x, class_idx)
    torch_mask = torchcam.cam(x, class_idx)
    abs_mask = np.abs(onnx_mask - torch_mask)
    onnx_heatmap = np.squeeze(np.uint8(cm.jet(onnx_mask[0])[..., :3] * 255))
    torch_heatmap = np.squeeze(np.uint8(cm.jet(torch_mask[0])[..., :3] * 255))
    abs_heatmap = np.squeeze(np.uint8(cm.jet(abs_mask[0])[..., :3] * 255))
    onnx_img = overlay_mask(img, onnx_heatmap)
    torch_img = overlay_mask(img, torch_heatmap)
    abs_img = overlay_mask(img, abs_heatmap)

    images = {
        "onnx_img": onnx_img,
        "torch_img": torch_img,
        "abs_img": abs_img,
        "abs_heatmap": Image.fromarray(abs_heatmap),
    }
    save_images("", model_name, images)


def save_images(path: str, model_name: str, images: dict) -> None:
    for name, img in images:
        img.save(path + "/" + model_name + "_" + name + ".jpg")
