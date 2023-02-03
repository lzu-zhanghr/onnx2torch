from test.cam.utils import apply_transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
import pytest
from typing import Tuple

import torchvision
from utils import check_model

# model = getattr(torchvision.models, "wide_resnet101_2")(pretrained=True)
# onnx_model, torch_model = get_model(
#     model, batch_size=1, resolution=224, opset_version=13
# )
#
# scorecam = ONNXCAM(onnx_model, "Conv_235")
#
#
# if __name__ == "__main__":
#     image = Image.open("/img/OIP.jfif").convert("RGB")
#
#     x = apply_transforms(image, 224).numpy()
#
#     # mask = scorecam.cam(x)
#     mask = np.load("/img/mask.npy")
#
#     # np.save("D:\workspace\onnx2torch\img\mask.npy", mask)
#
#     x = np.squeeze(x).swapaxes(0, 1).swapaxes(1, 2)
#     plt.imshow(x)
#     heatmap = np.squeeze(np.uint8(cm.jet(mask[0])[..., :3] * 255))
#
#     plt.imshow(heatmap, cmap="jet", alpha=0.5)
#     plt.show()


@pytest.mark.filterwarnings("ignore::torch.jit._trace.TracerWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "model",
    (
        ("resnet50", ''),
        ("resnet50", ''),

    ),
)
def test_cam(
    model: Tuple,
) -> None:  # pylint: disable=missing-function-docstring
    model = getattr(torchvision.models, model[0])(pretrained=True)
    check_model(model, model_name=model[0], layer_name=model[1], image_path='')


@pytest.mark.filterwarnings("ignore::torch.jit._trace.TracerWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "layer_name",
    (
        '',
        '',
    ),
)
def test_cam(
        layer_name: str,
) -> None:  # pylint: disable=missing-function-docstring
    model_name = ''
    model = getattr(torchvision.models, model_name)(pretrained=True)
    check_model(model, model_name=model_name, layer_name=model[1], image_path='')

