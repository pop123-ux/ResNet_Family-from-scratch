"""Smoke tests for the four ResNet implementations.

These tests intentionally avoid datasets and training loops. They only verify that each
architecture can be instantiated, execute a complete forward pass, produce the expected
classifier shape, propagate gradients through the network, and expose valid checkpoint
metadata for the pretrained models distributed by the repository.
"""

from pathlib import Path

import pytest
import torch

from models import ResNet_12, ResNet_18, ResNet_34, ResNet_50


MODEL_CASES = [
    # ResNet-12 is the custom 1 x 7 x 96 architecture used in this project.
    pytest.param(ResNet_12, (1, 1, 7, 96), 96, id="resnet12"),
    # ResNet-18 has a fixed 5 x 5 average-pooling head, so its intended 100 x 100
    # input is retained here.
    pytest.param(ResNet_18, (1, 3, 100, 100), 100, id="resnet18"),
    # ResNet-34 and ResNet-50 end in AdaptiveAvgPool2d, so a smaller image is
    # enough for a fast CPU-only structural smoke test while still traversing
    # every residual stage.
    pytest.param(ResNet_34, (1, 3, 64, 64), 100, id="resnet34"),
    pytest.param(ResNet_50, (1, 3, 64, 64), 100, id="resnet50"),
]

CHECKPOINT_CASES = [
    pytest.param(ResNet_12, "ResNet12_cifar100.pth", id="resnet12-checkpoint"),
    pytest.param(ResNet_18, "ResNet18_cifar100.pth", id="resnet18-checkpoint"),
    pytest.param(ResNet_34, "ResNet34_cifar100.pth", id="resnet34-checkpoint"),
]


@pytest.mark.parametrize("model_cls,input_shape,num_classes", MODEL_CASES)
def test_forward_shape(model_cls, input_shape, num_classes):
    """Every model should map a batch of images to one logit vector per sample."""
    torch.manual_seed(0)

    model = model_cls(num_classes=num_classes)
    model.eval()
    x = torch.randn(*input_shape)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (input_shape[0], num_classes)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("model_cls,input_shape,num_classes", MODEL_CASES)
def test_backward_reaches_trainable_parameters(model_cls, input_shape, num_classes):
    """A complete backward pass should create gradients for trainable parameters."""
    torch.manual_seed(0)

    model = model_cls(num_classes=num_classes)
    model.train()
    x = torch.randn(*input_shape)

    output = model(x)
    loss = output.square().mean()
    loss.backward()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)


@pytest.mark.parametrize("model_cls,_,__", MODEL_CASES)
def test_parameter_counter_matches_pytorch(model_cls, _, __):
    """The convenience params() method should report PyTorch's parameter count."""
    model = model_cls()

    expected = sum(parameter.numel() for parameter in model.parameters())

    assert model.params() == expected


@pytest.mark.parametrize("model_cls,checkpoint_name", CHECKPOINT_CASES)
def test_pretrained_checkpoint_paths(model_cls, checkpoint_name):
    """Distributed pretrained models should point to the root checkpoints directory."""
    checkpoint = Path(model_cls.DEFAULT_WEIGHTS)

    assert checkpoint.parent.name == "checkpoints"
    assert checkpoint.name == checkpoint_name


def test_resnet50_has_no_default_checkpoint():
    """ResNet-50 is structurally tested but has no distributed pretrained run."""
    assert ResNet_50.DEFAULT_WEIGHTS is None
