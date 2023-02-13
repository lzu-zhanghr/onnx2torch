import pytest
import torchvision
from test.attack.utils import gen_advs


@pytest.mark.filterwarnings("ignore::torch.jit._trace.TracerWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "model_name",
    (
        # "alexnet",
        # "googlenet",
        # "inception_v3",
        "resnet18",
        # "resnet34",
        # "resnet50",
        # "resnet101",
        # "resnet152",
    ),
)
def test_torch(
    model_name: str,
) -> None:  # pylint: disable=missing-function-docstring
    model = getattr(torchvision.models, model_name)(pretrained=True)
    gen_advs(
        model,
        batch_size=1,
        model_name=model_name,
        targeted=False,
        num=10000,
        query=10000,
        epsilon=0.05,
        onnx=False,
        early_stopping=True,
    )
