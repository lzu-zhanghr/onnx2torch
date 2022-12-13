import torch.hub

from test.utils import check_model

import pytest
import torchvision


@pytest.mark.filterwarnings("ignore::torch.jit._trace.TracerWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "model_name",
    (
        "alexnet",
        "googlenet",
        "inception_v3",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "wide_resnet50_2",
        "wide_resnet101_2",
        "vgg11",
        "vgg11_bn",
        "vgg13",
        "vgg13_bn",
        "vgg16",
        "vgg16_bn",
        "vgg19",
        "vgg19_bn",
        "mobilenet_v2",
        "squeezenet1_0",
        "squeezenet1_1",
        "mnasnet0_5",
        "mnasnet1_0",
    ),
)
def test_imagenet_classification(
    model_name: str,
) -> None:  # pylint: disable=missing-function-docstring
    model = getattr(torchvision.models, model_name)(pretrained=True)
    check_model(model, batch_size=32, model_name=model_name)


@pytest.mark.filterwarnings("ignore::torch.jit._trace.TracerWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "model_name",
    (
        "vgg11_bn",
        "vgg13_bn",
        "vgg16_bn",
        "vgg19_bn",
        "resnet20",
        "resnet32",
        "resnet44",
        "resnet56",
        "mobilenetv2_x0_5",
        "mobilenetv2_x0_75",
        "mobilenetv2_x1_0",
        "mobilenetv2_x1_4",
    ),
)
def test_cifar10_classification(
    model_name: str,
) -> None:  # pylint: disable=missing-function-docstring
    name = "cifar10_".join(model_name)
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", name, pretrained=True)
    check_model(model, batch_size=32, model_name=model_name, dataset="cifar10")


@pytest.mark.filterwarnings("ignore::torch.jit._trace.TracerWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "model_name",
    (
        "vgg11_bn",
        "vgg13_bn",
        "vgg16_bn",
        "vgg19_bn",
        "resnet20",
        "resnet32",
        "resnet44",
        "resnet56",
        "mobilenetv2_x0_5",
        "mobilenetv2_x0_75",
        "mobilenetv2_x1_0",
        "mobilenetv2_x1_4",
    ),
)
def test_cifar100_classification(
    model_name: str,
) -> None:  # pylint: disable=missing-function-docstring
    name = "cifar10_".join(model_name)
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", name, pretrained=True)
    check_model(model, batch_size=32, model_name=model_name, dataset="cifar100")
