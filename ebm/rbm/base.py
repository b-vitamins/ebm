"""Restricted Boltzmann machine base class."""

import copy
from collections.abc import Callable

import torch
import torch.nn as nn

InitStrategy = str | float | int | torch.Tensor | Callable[[torch.Tensor], None]


class RBMBase(nn.Module):
    """
    Base class for a Restricted Boltzmann Machine (RBM).

    Parameters
    ----------
    num_visible : int
        Number of visible units.
    num_hidden : int
        Number of hidden units.
    weight_init : InitStrategy, optional
        Initialization strategy for weights. Defaults to "xavier_uniform".
    visible_bias_init : InitStrategy, optional
        Initialization strategy for visible bias. Defaults to "zeros".
    hidden_bias_init : InitStrategy, optional
        Initialization strategy for hidden bias. Defaults to "zeros".
    visible_activation_fn : Callable[[torch.Tensor], torch.Tensor] or None, optional
        Activation function for the visible layer. Defaults to torch.sigmoid.
    hidden_activation_fn : Callable[[torch.Tensor], torch.Tensor] or None, optional
        Activation function for the hidden layer. Defaults to torch.sigmoid.
    """

    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        weight_init: InitStrategy = "normal",
        visible_bias_init: InitStrategy = "zeros",
        hidden_bias_init: InitStrategy = "zeros",
        visible_activation_fn: Callable[[torch.Tensor], torch.Tensor]
        | None = None,
        hidden_activation_fn: Callable[[torch.Tensor], torch.Tensor]
        | None = None,
    ) -> None:
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        self.weight_init = weight_init
        self.visible_bias_init = visible_bias_init
        self.hidden_bias_init = hidden_bias_init

        self.visible_activation_fn = visible_activation_fn or torch.sigmoid
        self.hidden_activation_fn = hidden_activation_fn or torch.sigmoid

        # shape: (num_visible, num_hidden)
        self.weight = nn.Parameter(torch.empty(num_visible, num_hidden))
        # shape: (num_visible,)
        self.visible_bias = nn.Parameter(torch.empty(num_visible))
        # shape: (num_hidden,)
        self.hidden_bias = nn.Parameter(torch.empty(num_hidden))
        # shape: (num_visible,)
        self.register_buffer("mle_visible_bias", torch.empty(num_visible))

        self.reset_parameters()

    @staticmethod
    @torch.no_grad()
    def _initialize_parameter(
        tensor: torch.Tensor,
        strategy: InitStrategy,
        constant_value: float = 0.0,
    ) -> None:
        # shape: tensor (whatever shape is passed in) -> remains the same shape after init
        if isinstance(strategy, float | int):
            tensor.normal_()  # shape stays the same
            tensor.mul_(float(strategy))  # shape stays the same
        elif isinstance(strategy, torch.Tensor):
            # shape must match 'tensor'
            if strategy.shape != tensor.shape:
                raise ValueError(
                    f"Tensor init shape {strategy.shape} != param shape {tensor.shape}"
                )
            tensor.copy_(strategy)  # shape stays the same
        elif callable(strategy):
            strategy(tensor)  # shape stays the same
        else:
            if strategy == "xavier_uniform":
                nn.init.xavier_uniform_(tensor)  # shape stays the same
            elif strategy == "xavier_normal":
                nn.init.xavier_normal_(tensor)  # shape stays the same
            elif strategy == "he_uniform":
                nn.init.kaiming_uniform_(
                    tensor, nonlinearity="relu"
                )  # shape stays
            elif strategy == "he_normal":
                nn.init.kaiming_normal_(
                    tensor, nonlinearity="relu"
                )  # shape stays
            elif strategy == "orthogonal":
                nn.init.orthogonal_(tensor)  # shape stays
            elif strategy == "sparse":
                nn.init.sparse_(tensor, sparsity=0.1, std=0.01)  # shape stays
            elif strategy == "zeros":
                nn.init.zeros_(tensor)  # shape stays
            elif strategy == "ones":
                nn.init.ones_(tensor)  # shape stays
            elif strategy == "constant":
                nn.init.constant_(tensor, constant_value)  # shape stays
            elif strategy == "normal":
                nn.init.normal_(tensor, mean=0.0, std=0.01)  # shape stays
            else:
                raise ValueError(f"Unknown init strategy: {strategy}")

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset parameters using the specified initialization strategies."""
        # shape: self.weight (num_visible, num_hidden)
        # shape: self.visible_bias (num_visible,)
        # shape: self.hidden_bias (num_hidden,)
        self._initialize_parameter(self.weight, self.weight_init)
        self._initialize_parameter(self.visible_bias, self.visible_bias_init)
        self._initialize_parameter(self.hidden_bias, self.hidden_bias_init)

    def visible_activation(self, pre_activation: torch.Tensor) -> torch.Tensor:
        """
        Apply the visible activation function to the given pre-activation tensor.

        Parameters
        ----------
        pre_activation : torch.Tensor
            The pre-activation values for visible units.

        Returns
        -------
        torch.Tensor
            The activated visible units.
        """
        # shape: pre_activation (batch_size, num_visible) -> return (batch_size, num_visible)
        return self.visible_activation_fn(pre_activation)

    def hidden_activation(self, pre_activation: torch.Tensor) -> torch.Tensor:
        """
        Apply the hidden activation function to the given pre-activation tensor.

        Parameters
        ----------
        pre_activation : torch.Tensor
            The pre-activation values for hidden units.

        Returns
        -------
        torch.Tensor
            The activated hidden units.
        """
        # shape: pre_activation (batch_size, num_hidden) -> return (batch_size, num_hidden)
        return self.hidden_activation_fn(pre_activation)

    @torch.no_grad()
    def init_visible_bias_from_means(self, means: torch.Tensor) -> None:
        """
        Update visible biases using the logit of the given means.

        Parameters
        ----------
        means : torch.Tensor
            A 1D tensor of shape (num_visible,) representing the means
            of the visible units.
        """
        # shape: means (num_visible,) -> clamped_means (num_visible,) -> logit_means (num_visible,)
        clamped_means = means.clamp(1e-3, 1.0 - 1e-3)
        logit_means = torch.logit(clamped_means)
        # shape: self.mle_visible_bias (num_visible,) <- logit_means (num_visible,)
        self.mle_visible_bias.copy_(logit_means)
        # shape: self.visible_bias (num_visible,) <- logit_means (num_visible,)
        self.visible_bias.copy_(logit_means)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Energy-based models do not have a typical forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_visible).

        Raises
        ------
        NotImplementedError
            Always raised, since an RBM typically does not use a forward pass
            in the conventional sense.
        """
        # shape: x (batch_size, num_visible) -> NotImplementedError
        raise NotImplementedError(
            "Energy-based model; no typical forward pass."
        )

    def clone(self) -> "RBMBase":
        """
        Return a deep copy of this RBM, including all parameters and buffers.

        Returns
        -------
        RBMBase
            A deep copy of the current RBM instance.
        """
        # deep copy preserves shapes of parameters, buffers
        return copy.deepcopy(self)
