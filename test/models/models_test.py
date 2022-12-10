from test.utils import check_model

import pytest
import torchvision


@pytest.mark.filterwarnings("ignore::torch.jit._trace.TracerWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "model_name",
    (
        "resnet18",
        "resnet50",
        "mobilenet_v2",
        "mobilenet_v3_large",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "wide_resnet50_2",
        "resnext50_32x4d",
        "vgg16",
        "googlenet",
        "mnasnet1_0",
        "regnet_y_400mf",
        "regnet_y_16gf",
    ),
)
def test_torchvision_classification(
    model_name: str,
) -> None:  # pylint: disable=missing-function-docstring
    model = getattr(torchvision.models, model_name)(pretrained=True)
    check_model(model, batch_size=32, model_name=model_name)
