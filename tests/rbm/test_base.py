import pytest
import torch

from ebm.rbm.base import RBMBase


def test_default_init():
    rbm = RBMBase(num_visible=6, num_hidden=3)
    assert rbm.weight.shape == (6, 3)
    assert rbm.visible_bias.shape == (6,)
    assert rbm.hidden_bias.shape == (3,)
    assert rbm.mle_visible_bias.shape == (6,)


@pytest.mark.parametrize(
    "strategy",
    [
        "xavier_uniform",
        "xavier_normal",
        "he_uniform",
        "he_normal",
        "orthogonal",
        "sparse",
        "zeros",
        "ones",
        "constant",
        "normal",
    ],
)
def test_parameter_initialization_strategies(strategy):
    rbm = RBMBase(num_visible=4, num_hidden=2, weight_init=strategy)
    assert rbm.weight.shape == (4, 2)


def test_parameter_initialization_with_numbers():
    rbm = RBMBase(num_visible=5, num_hidden=4, weight_init=0.05)
    assert rbm.weight.shape == (5, 4)
    # Just ensuring we get a normal distribution scaled by 0.05
    assert torch.abs(rbm.weight).mean() > 0  # non-zero mean
    # The precise mean is not guaranteed, but it should be in the ballpark


def test_parameter_initialization_with_tensor():
    custom_init = torch.ones(5, 5)
    rbm = RBMBase(num_visible=5, num_hidden=5, weight_init=custom_init)
    assert torch.allclose(rbm.weight, custom_init)


def test_parameter_initialization_with_callable():
    def custom_init_fn(t):
        with torch.no_grad():
            t.fill_(2.0)

    rbm = RBMBase(num_visible=3, num_hidden=2, weight_init=custom_init_fn)
    assert torch.allclose(rbm.weight, torch.full((3, 2), 2.0))


def test_visible_activation():
    rbm = RBMBase(num_visible=4, num_hidden=4)
    input_tensor = torch.randn(2, 4)  # batch_size=2, num_visible=4
    out = rbm.visible_activation(input_tensor)
    assert out.shape == (2, 4)


def test_hidden_activation():
    rbm = RBMBase(num_visible=4, num_hidden=4)
    input_tensor = torch.randn(3, 4)  # batch_size=3, num_hidden=4
    out = rbm.hidden_activation(input_tensor)
    assert out.shape == (3, 4)


def test_init_visible_bias_from_means():
    rbm = RBMBase(num_visible=5, num_hidden=2)
    means = torch.tensor([0.2, 0.5, 0.9, 0.01, 0.99])
    rbm.init_visible_bias_from_means(means)
    assert torch.allclose(rbm.mle_visible_bias, rbm.visible_bias, atol=1e-5)
    assert rbm.visible_bias.shape == (5,)


def test_forward_raises():
    rbm = RBMBase(num_visible=4, num_hidden=2)
    x = torch.randn(3, 4)
    with pytest.raises(NotImplementedError):
        _ = rbm(x)


def test_clone():
    rbm = RBMBase(num_visible=5, num_hidden=2)
    rbm.weight.data.fill_(1.23)
    rbm.visible_bias.data.fill_(-1.0)
    rbm.hidden_bias.data.fill_(2.0)

    rbm2 = rbm.clone()

    # Make sure clone didn't just copy references
    assert rbm2 is not rbm
    assert rbm2.weight is not rbm.weight
    assert rbm2.visible_bias is not rbm.visible_bias
    assert rbm2.hidden_bias is not rbm.hidden_bias

    # Check values
    assert torch.allclose(rbm.weight, rbm2.weight)
    assert torch.allclose(rbm.visible_bias, rbm2.visible_bias)
    assert torch.allclose(rbm.hidden_bias, rbm2.hidden_bias)


def test_reset_parameters():
    rbm = RBMBase(num_visible=3, num_hidden=2, weight_init="zeros")
    rbm.weight_init = "ones"
    rbm.reset_parameters()
    assert torch.all(rbm.weight == 1.0)
