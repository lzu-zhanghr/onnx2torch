from attack.utils import zoo_attack_model

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
        "wide_resnet101_2" "vgg11",
        "vgg11_bn",
        "vgg13",
        "vgg13_bn",
        "vgg16",
        "vgg16_bn",
        "vgg19",
        "vgg19_bn",
        "mobilenet_v2",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "squeezenet1_0",
        "squeezenet1_1",
        "mnasnet0_5",
        "mnasnet0_75",
        "mnasnet1_0",
        "mnasnet1_3",
        "densenet121",
        "densenet169",
        "densenet201",
        "densenet161",
        "shufflenet_v2_x0_5",
        "shufflenet_v2_x1_0",
        "shufflenet_v2_x1_5",
        "shufflenet_v2_x2_0",
    ),
)
def targeted_attack_torchvision_classification(
    model_name: str,
) -> None:  # pylint: disable=missing-function-docstring
    model = getattr(torchvision.models, model_name)(pretrained=True)
    zoo_attack_model(model, batch_size=32, model_name=model_name, targeted=True)


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
        "wide_resnet101_2" "vgg11",
        "vgg11_bn",
        "vgg13",
        "vgg13_bn",
        "vgg16",
        "vgg16_bn",
        "vgg19",
        "vgg19_bn",
        "mobilenet_v2",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "squeezenet1_0",
        "squeezenet1_1",
        "mnasnet0_5",
        "mnasnet0_75",
        "mnasnet1_0",
        "mnasnet1_3",
        "densenet121",
        "densenet169",
        "densenet201",
        "densenet161",
        "shufflenet_v2_x0_5",
        "shufflenet_v2_x1_0",
        "shufflenet_v2_x1_5",
        "shufflenet_v2_x2_0",
    ),
)
def notarget_attack_torchvision_classification(
    model_name: str,
) -> None:  # pylint: disable=missing-function-docstring
    model = getattr(torchvision.models, model_name)(pretrained=True)
    zoo_attack_model(model, batch_size=32, model_name=model_name, targeted=False)
