"""Unit tests for the custom BatchNorm implementations used by the ResNet models."""

import importlib

import pytest
import torch


BATCHNORM_MODULES = [
    pytest.param("models.ResNet_12", id="resnet12-batchnorm"),
    pytest.param("models.ResNet_18", id="resnet18-batchnorm"),
    pytest.param("models.ResNet_34", id="resnet34-batchnorm"),
    pytest.param("models.ResNet_50", id="resnet50-batchnorm"),
]


def get_batchnorm(module_name):
    module = importlib.import_module(module_name)
    return module.BatchNorm


@pytest.mark.parametrize("module_name", BATCHNORM_MODULES)
def test_batchnorm_has_affine_parameters_and_running_buffers(module_name):
    """Gamma/beta should be trainable, while running statistics should be buffers."""
    BatchNorm = get_batchnorm(module_name)
    layer = BatchNorm(4)

    parameters = dict(layer.named_parameters())
    buffers = dict(layer.named_buffers())

    assert parameters["weight"].shape == (4,)
    assert parameters["bias"].shape == (4,)
    assert parameters["weight"].requires_grad
    assert parameters["bias"].requires_grad

    assert buffers["running_mean"].shape == (4,)
    assert buffers["running_var"].shape == (4,)
    assert not buffers["running_mean"].requires_grad
    assert not buffers["running_var"].requires_grad


@pytest.mark.parametrize("module_name", BATCHNORM_MODULES)
def test_training_output_is_normalized_per_channel(module_name):
    """In training mode, each channel should be normalized using batch statistics."""
    torch.manual_seed(0)
    BatchNorm = get_batchnorm(module_name)
    layer = BatchNorm(3)
    layer.train()

    x = torch.randn(8, 3, 5, 5) * 3.0 + 2.0
    output = layer(x)

    channel_mean = output.mean(dim=(0, 2, 3))
    channel_var = output.var(dim=(0, 2, 3), unbiased=False)

    assert torch.allclose(channel_mean, torch.zeros_like(channel_mean), atol=1e-5)
    assert torch.allclose(channel_var, torch.ones_like(channel_var), atol=1e-4)


@pytest.mark.parametrize("module_name", BATCHNORM_MODULES)
def test_running_statistics_update_only_in_training(module_name):
    """Running statistics should update during training and remain fixed in eval mode."""
    torch.manual_seed(0)
    BatchNorm = get_batchnorm(module_name)
    layer = BatchNorm(2, momentum=0.1)

    x = torch.randn(6, 2, 4, 4) + 5.0

    layer.train()
    layer(x)

    mean_after_train = layer.running_mean.clone()
    var_after_train = layer.running_var.clone()

    assert not torch.equal(mean_after_train, torch.zeros_like(mean_after_train))

    layer.eval()
    layer(torch.randn(6, 2, 4, 4) - 5.0)

    assert torch.equal(layer.running_mean, mean_after_train)
    assert torch.equal(layer.running_var, var_after_train)


@pytest.mark.parametrize("module_name", BATCHNORM_MODULES)
def test_eval_mode_uses_running_statistics(module_name):
    """Eval output should follow the stored running mean/variance formula."""
    BatchNorm = get_batchnorm(module_name)
    layer = BatchNorm(2, eps=1e-5)

    with torch.no_grad():
        layer.running_mean.copy_(torch.tensor([1.0, -2.0]))
        layer.running_var.copy_(torch.tensor([4.0, 9.0]))
        layer.weight.copy_(torch.tensor([2.0, 0.5]))
        layer.bias.copy_(torch.tensor([-1.0, 3.0]))

    x = torch.tensor(
        [[[[5.0]], [[4.0]]]],
        dtype=torch.float32,
    )

    layer.eval()
    output = layer(x)

    mean = layer.running_mean.view(1, -1, 1, 1)
    var = layer.running_var.view(1, -1, 1, 1)
    weight = layer.weight.view(1, -1, 1, 1)
    bias = layer.bias.view(1, -1, 1, 1)
    expected = ((x - mean) / torch.sqrt(var + layer.eps)) * weight + bias

    assert torch.allclose(output, expected, atol=1e-6)


@pytest.mark.parametrize("module_name", BATCHNORM_MODULES)
def test_gradients_flow_through_batchnorm(module_name):
    """Input, gamma, and beta should all participate in backpropagation."""
    torch.manual_seed(0)
    BatchNorm = get_batchnorm(module_name)
    layer = BatchNorm(3)

    x = torch.randn(4, 3, 5, 5, requires_grad=True)
    loss = layer(x).square().mean()
    loss.backward()

    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()
    assert torch.isfinite(layer.bias.grad).all()
