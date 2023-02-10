import pytest
from typing import Tuple

import torchvision
from test.cam.utils import check_model, check_diff

#
# @pytest.mark.filterwarnings("ignore::torch.jit._trace.TracerWarning")
# @pytest.mark.filterwarnings("ignore::DeprecationWarning")
# @pytest.mark.parametrize(
#     "net",
#     (
#         ("googlenet", "Conv_158", "Conv_56"),
#         ("inception_v3", "Conv_256", "Conv_93"),
#         ("resnet18", "Conv_43", "Conv_19"),
#         ("resnet34", "Conv_81", "Conv_35"),
#         ("resnet50", "Conv_116", "Conv_52"),
#         ("resnet101", "Conv_235", "Conv_103"),
#         ("resnet152", "Conv_354", "Conv_154"),
#         ("resnext50_32x4d", "Conv_116", "Conv_52"),
#         ("resnext101_32x8d", "Conv_235", "Conv_103"),
#         ("wide_resnet50_2", "Conv_116", "Conv_52"),
#         ("mobilenet_v2", "Conv_163", "Conv_51"),
#         ("squeezenet1_0", "Conv_61", "Conv_25"),
#         ("squeezenet1_1", "Conv_61", "Conv_25"),
#     ),
# )
# def test_cam(
#     net: Tuple,
# ) -> None:  # pylint: disable=missing-function-docstring
#     model = getattr(torchvision.models, net[0])(pretrained=True)
#     image_name = "ILSVRC2012_val_00037081"
#     check_model(
#         model,
#         model_name=net[0],
#         onnx_layer=net[1],
#         torch_layer=net[2],
#         image_name=image_name,
#     )
#
#
# @pytest.mark.filterwarnings("ignore::torch.jit._trace.TracerWarning")
# @pytest.mark.filterwarnings("ignore::DeprecationWarning")
# @pytest.mark.parametrize(
#     "net",
#     ((282, 281, "Conv_235", "Conv_103", "layer-1"),),
# )
# def test_case(
#     net: Tuple[int, int, str, str, str],
# ) -> None:  # pylint: disable=missing-function-docstring
#     model = getattr(torchvision.models, "wide_resnet101_2")(pretrained=True)
#     image_name = "ILSVRC2012_val_00016399"
#     check_model(
#         model,
#         model_name="wide_resnet101_2" + "_" + net[4],
#         onnx_layer=net[2],
#         torch_layer=net[3],
#         image_name=image_name,
#         class_idx=(net[0], net[1]),
#     )
#

@pytest.mark.filterwarnings("ignore::torch.jit._trace.TracerWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "net",
    ((282, 281, "Conv_235", "Conv_103", "layer-1_diff"),),
)
def test_diff(
    net: Tuple[int, int, str, str, str],
) -> None:  # pylint: disable=missing-function-docstring
    model = getattr(torchvision.models, "wide_resnet101_2")(pretrained=True)
    image_name = "ILSVRC2012_val_00016399"
    check_diff(
        model,
        model_name="wide_resnet101_2" + "_" + net[4],
        onnx_layer=net[2],
        torch_layer=net[3],
        image_name=image_name,
        class_idx=(net[0], net[1]),
    )
