import io
from typing import Union, Any, Tuple
import onnx
import torch
from onnx import ModelProto
from onnx2torch.converter import convert
from numpy.typing import NDArray
from matplotlib import cm
import numpy as np
import onnxruntime as ort


from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from test import IMG_DIR, DATASETS_DIR
from test.cam.method import ONNXCAM, TorchCAM


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


def load_img(path: Any, resolution: int = 224) -> Image.Image:
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


def normalize(mask: NDArray) -> NDArray:
    mask_min, mask_max = (
        mask.min(),
        mask.max(),
    )

    mask = np.divide(
        mask - mask_min,
        mask_max - mask_min,
    )
    return mask


def check_model(  # pylint: disable=missing-function-docstring,unused-argument
    model: torch.nn.Module,
    model_name: str,
    onnx_layer: str,
    torch_layer: str,
    image_name: str,
    resolution: int = 224,
    opset_version: int = 13,
    class_idx: Tuple[int, int] = None,
) -> None:
    onnx_model, torch_model = get_model(model, opset_version)
    onnxcam = ONNXCAM(onnx_model, onnx_layer)
    torchcam = TorchCAM(torch_model, torch_layer)
    image_path = IMG_DIR / "raw" / (image_name + ".JPEG")
    img = load_img(image_path, resolution)
    x = apply_transforms(img).numpy()

    onnx_mask = onnxcam.cam(x, class_idx[0])
    torch_mask = torchcam.cam(x, class_idx[1])
    abs_mask = np.abs(onnx_mask - torch_mask)
    onnx_mask = normalize(onnx_mask)
    torch_mask = normalize(torch_mask)
    abs_mask = normalize(abs_mask)

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
    save_images(IMG_DIR / "cam", model_name, image_name, images)


def check_diff(  # pylint: disable=missing-function-docstring,unused-argument
    model: torch.nn.Module,
    model_name: str,
    onnx_layer: str,
    torch_layer: str,
    image_name: str,
    resolution: int = 224,
    opset_version: int = 13,
    class_idx: Tuple[int, int] = None,
) -> None:
    onnx_model, torch_model = get_model(model, opset_version)
    onnxcam = ONNXCAM(onnx_model, onnx_layer)
    torchcam = TorchCAM(torch_model, torch_layer)
    image_path = IMG_DIR / "raw" / (image_name + ".JPEG")
    img = load_img(image_path, resolution)
    x = apply_transforms(img).numpy()

    onnx_mask_class1 = onnxcam.cam(x, class_idx[0])
    onnx_mask_class2 = onnxcam.cam(x, class_idx[1])
    torch_mask_class1 = torchcam.cam(x, class_idx[0])
    torch_mask_class2 = torchcam.cam(x, class_idx[1])
    abs_onnx_mask = np.abs(onnx_mask_class1 - onnx_mask_class2)
    abs_torch_mask = np.abs(torch_mask_class1 - torch_mask_class2)
    onnx_mask_class1 = normalize(onnx_mask_class1)
    onnx_mask_class2 = normalize(onnx_mask_class2)
    torch_mask_class1 = normalize(torch_mask_class1)
    torch_mask_class2 = normalize(torch_mask_class2)
    abs_onnx_mask = normalize(abs_onnx_mask)
    abs_torch_mask = normalize(abs_torch_mask)

    onnx_heatmap_class1 = np.squeeze(
        np.uint8(cm.jet(onnx_mask_class1[0])[..., :3] * 255)
    )
    onnx_heatmap_class2 = np.squeeze(
        np.uint8(cm.jet(onnx_mask_class2[0])[..., :3] * 255)
    )
    torch_heatmap_class1 = np.squeeze(
        np.uint8(cm.jet(torch_mask_class1[0])[..., :3] * 255)
    )
    torch_heatmap_class2 = np.squeeze(
        np.uint8(cm.jet(torch_mask_class2[0])[..., :3] * 255)
    )
    abs_onnx_heatmap = np.squeeze(np.uint8(cm.jet(abs_onnx_mask[0])[..., :3] * 255))
    abs_torch_heatmap = np.squeeze(np.uint8(cm.jet(abs_torch_mask[0])[..., :3] * 255))
    onnx_img_class1 = overlay_mask(img, onnx_heatmap_class1)
    onnx_img_class2 = overlay_mask(img, onnx_heatmap_class2)
    torch_img_class1 = overlay_mask(img, torch_heatmap_class1)
    torch_img_class2 = overlay_mask(img, torch_heatmap_class2)
    abs_onnx_img = overlay_mask(img, abs_onnx_heatmap)
    abs_torch_img = overlay_mask(img, abs_torch_heatmap)

    images = {
        "onnx_img_class1": onnx_img_class1,
        "onnx_img_class2": onnx_img_class2,
        "torch_img_class1": torch_img_class1,
        "torch_img_class2": torch_img_class2,
        "abs_onnx_img": abs_onnx_img,
        "abs_torch_img": abs_torch_img,
        "abs_onnx_heatmap": Image.fromarray(abs_onnx_heatmap),
        "abs_torch_heatmap": Image.fromarray(abs_torch_heatmap),
    }
    save_images(IMG_DIR / "cam", model_name, image_name, images)


def create_imagenet_test(  # pylint: disable=missing-function-docstring
    root: Any,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> Tuple[int, DataLoader]:
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
    print(dataset.class_to_idx, dataset.imgs)
    return classes, dataloader


def get_case(  # pylint: disable=missing-function-docstring,unused-argument
    model: torch.nn.Module,
    opset_version: int = 13,
) -> None:
    onnx_model, torch_model = get_model(model, opset_version)
    root = DATASETS_DIR / "ILSVRC2012_img_val"
    classes, dataloader = create_imagenet_test(root, 1, num_workers=4)
    ort_session = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )

    for _, (image, target) in enumerate(dataloader):
        onnx_input = {"input": image.detach().cpu().numpy()}
        torch_input = image

        onnx_output = ort_session.run(
            output_names=None,
            input_feed=onnx_input,
        )
        onnx_output = torch.from_numpy(np.array(onnx_output)).squeeze()
        torch_output = torch_model(torch_input)
        torch_class = torch_output.argmax(-1)
        onnx_class = onnx_output.argmax(-1)

        err = torch.count_nonzero(torch_class - onnx_class)

        if err:
            torch.save(
                {
                    "x": image,
                    "torch_class": torch_class,
                    "onnx_class": torch.from_numpy(onnx_class),
                },
                IMG_DIR / "case.pkl",
            )
            break


def save_images(path: Any, model_name: str, image_name: str, images: dict) -> None:
    for name, img in images.items():
        img.save(path / (model_name + "_" + image_name + "_" + name + ".jpg"))
