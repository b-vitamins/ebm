"""Restricted Boltzmann machine base class."""

import copy
from collections.abc import Callable

import torch
import torch.nn as nn

# Type alias for user-provided initialization strategies.
# You may rename or expand this as you see fit.
InitStrategy = str | float | int | torch.Tensor | Callable[[torch.Tensor], None]


class InitStrategyFactory:
    """
    Factory for parameter initialization.

    This class transforms a user-given init strategy (string, numeric,
    tensor, or callable) into a callable that initializes a given tensor.

    Parameters
    ----------
    strategy : InitStrategy
        The user-specified initialization strategy.
    constant_value : float, optional
        Value used by the 'constant' strategy if you want a fixed
        fill value. Only used if strategy == "constant". Defaults to 0.0.

    Notes
    -----
    - If `strategy` is an int or float, we do constant initialization
      with that numeric value. (Old code sometimes used normal(0,1)*scale.)
    - If `strategy` is a `torch.Tensor`, it must have the same shape
      as the parameter being initialized.
    - If `strategy` is a callable, we directly call it on the tensor.
    - If `strategy` is a string, we map it to one of the standard
      PyTorch initializers (e.g. xavier_uniform, he_normal, etc.),
      or raise a ValueError if unknown.
    """

    def __init__(self, strategy: InitStrategy, constant_value: float = 0.0):
        self.strategy = strategy
        self.constant_value = constant_value

    def get_initializer(self) -> Callable[[torch.Tensor], None]:
        """Return a callable that takes a tensor and initializes it according to the specified strategy."""
        # Numeric => constant initialization
        if isinstance(self.strategy, float | int):
            value = float(self.strategy)

            def init_fn(tensor: torch.Tensor) -> None:
                nn.init.constant_(tensor, value)

            return init_fn

        # Directly copy from a user-provided tensor
        if isinstance(self.strategy, torch.Tensor):
            user_tensor = self.strategy

            def init_fn(tensor: torch.Tensor) -> None:
                if user_tensor.shape != tensor.shape:
                    raise ValueError(
                        f"Tensor init shape {user_tensor.shape} != param shape {tensor.shape}"
                    )
                tensor.copy_(user_tensor)

            return init_fn

        # Directly apply a callable
        if callable(self.strategy):
            return self.strategy

        # If we get here, strategy must be a string
        if self.strategy == "xavier_uniform":
            return nn.init.xavier_uniform_
        elif self.strategy == "xavier_normal":
            return nn.init.xavier_normal_
        elif self.strategy == "he_uniform":
            return lambda t: nn.init.kaiming_uniform_(t, nonlinearity="relu")
        elif self.strategy == "he_normal":
            return lambda t: nn.init.kaiming_normal_(t, nonlinearity="relu")
        elif self.strategy == "orthogonal":
            return nn.init.orthogonal_
        elif self.strategy == "sparse":
            return lambda t: nn.init.sparse_(t, sparsity=0.1, std=0.01)
        elif self.strategy == "zeros":
            return nn.init.zeros_
        elif self.strategy == "ones":
            return nn.init.ones_
        elif self.strategy == "constant":
            return lambda t: nn.init.constant_(t, self.constant_value)
        elif self.strategy == "normal":
            return lambda t: nn.init.normal_(t, mean=0.0, std=0.01)
        else:
            raise ValueError(f"Unknown init strategy: {self.strategy}")

    def apply(self, tensor: torch.Tensor) -> None:
        """Initialize the given tensor according to the stored strategy."""
        initializer = self.get_initializer()
        initializer(tensor)


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
        Initialization strategy for weights. Defaults to "normal".
    visible_bias_init : InitStrategy, optional
        Initialization strategy for visible bias. Defaults to "zeros".
    hidden_bias_init : InitStrategy, optional
        Initialization strategy for hidden bias. Defaults to "zeros".
    visible_activation_fn : Callable[[torch.Tensor], torch.Tensor] or None, optional
        Activation function for the visible layer. Defaults to torch.sigmoid.
    hidden_activation_fn : Callable[[torch.Tensor], torch.Tensor] or None, optional
        Activation function for the hidden layer. Defaults to torch.sigmoid.

    Notes
    -----
    This class does not call `reset_parameters()` during __init__.
    Subclasses should call `self.reset_parameters()` themselves
    (typically at the end of their own __init__).
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
        do_init: bool = True,
    ) -> None:
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        # Save initialization strategies for later usage
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

        # shape: (num_visible,) – used for optional means-based init
        self.register_buffer("mle_visible_bias", torch.empty(num_visible))

        # Conditional initialization
        if do_init:
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """
        Reset parameters using the specified initialization strategies.

        By default:
          - weight_init="normal"
          - visible_bias_init="zeros"
          - hidden_bias_init="zeros"
        """
        InitStrategyFactory(self.weight_init).apply(self.weight)
        InitStrategyFactory(self.visible_bias_init).apply(self.visible_bias)
        InitStrategyFactory(self.hidden_bias_init).apply(self.hidden_bias)

    def visible_activation(self, pre_v: torch.Tensor) -> torch.Tensor:
        """
        Apply the visible activation function to the given pre-activation tensor.

        Parameters
        ----------
        pre_v : torch.Tensor
            The pre-activation values for visible units of shape (N, num_visible, ...).

        Returns
        -------
        torch.Tensor
            The activated visible units, same shape as `pre_v`.
        """
        return self.visible_activation_fn(pre_v)

    def hidden_activation(self, pre_h: torch.Tensor) -> torch.Tensor:
        """
        Apply the hidden activation function to the given pre-activation tensor.

        Parameters
        ----------
        pre_h : torch.Tensor
            The pre-activation values for hidden units of shape (N, num_hidden, ...).

        Returns
        -------
        torch.Tensor
            The activated hidden units, same shape as `pre_h`.
        """
        return self.hidden_activation_fn(pre_h)

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
        clamped_means = means.clamp(1e-3, 1.0 - 1e-3)
        logit_means = torch.logit(clamped_means)
        self.mle_visible_bias.copy_(logit_means)
        self.visible_bias.copy_(logit_means)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Energy-based models do not have a typical forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, num_visible, ...).

        Raises
        ------
        NotImplementedError
            Always raised, since an RBM typically does not use a forward pass
            in the conventional sense.
        """
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
        return copy.deepcopy(self)
