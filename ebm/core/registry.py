"""Registry system for models, samplers, and other components.

This module provides a flexible plugin system that allows easy registration
and discovery of different model types, samplers, and other components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

from .config import BaseConfig
from .logging import logger

T = TypeVar("T")
ConfigT = TypeVar("ConfigT", bound=BaseConfig)


class Registry(ABC):
    """Abstract base class for component registries."""

    def __init__(self, name: str):
        """Initialize registry.

        Args:
            name: Name of this registry (e.g., 'models', 'samplers')
        """
        self.name = name
        self._registry: dict[str, type[T]] = {}
        self._aliases: dict[str, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    @abstractmethod
    def _validate_class(self, cls: type[T]) -> None:
        """Validate that a class can be registered.

        Args:
            cls: Class to validate

        Raises
        ------
            TypeError: If class is not valid for this registry
        """

    def register(
        self,
        name: str,
        cls: type[T] | None = None,
        *,
        aliases: list[str] | None = None,
        **metadata: Any,
    ) -> type[T] | Callable[[type[T]], type[T]]:
        """Register a class in the registry.

        Can be used as a decorator or called directly.

        Args:
            name: Name to register the class under
            cls: Class to register (if None, returns decorator)
            aliases: Alternative names for this class
            **metadata: Additional metadata to store

        Returns
        -------
            Registered class or decorator function

        Example:
            @registry.register('my_model', aliases=['mm'])
            class MyModel:
                pass

            # Or directly:
            registry.register('my_model', MyModel)
        """

        def decorator(cls_to_register: type[T]) -> type[T]:
            self._validate_class(cls_to_register)

            # Check for duplicate registration
            if name in self._registry:
                logger.warning(
                    "Overwriting existing registration",
                    registry=self.name,
                    name=name,
                    old_class=self._registry[name].__name__,
                    new_class=cls_to_register.__name__,
                )

            # Register the class
            self._registry[name] = cls_to_register
            self._metadata[name] = metadata

            # Register aliases
            if aliases:
                for alias in aliases:
                    self._aliases[alias] = name

            logger.debug(
                f"Registered {cls_to_register.__name__}",
                registry=self.name,
                name=name,
                aliases=aliases,
            )

            return cls_to_register

        if cls is None:
            return decorator
        return decorator(cls)

    def get(self, name: str) -> type[T]:
        """Get a registered class by name.

        Args:
            name: Name or alias of the class

        Returns
        -------
            Registered class

        Raises
        ------
            KeyError: If name is not registered
        """
        # Check if it's an alias
        if name in self._aliases:
            name = self._aliases[name]

        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: {available}"
            )

        return self._registry[name]

    def create(
        self, name: str, *args: Any, config: ConfigT | None = None, **kwargs: Any
    ) -> T:
        """Create an instance of a registered class.

        Args:
            name: Name of the class to instantiate
            *args: Positional arguments for the class
            config: Configuration object (if supported)
            **kwargs: Keyword arguments for the class

        Returns
        -------
            Instance of the registered class
        """
        cls = self.get(name)

        # Try different instantiation patterns
        if config is not None:
            # Try config-based instantiation first
            try:
                return cls(config, *args, **kwargs)
            except TypeError:
                pass

        # Standard instantiation
        return cls(*args, **kwargs)

    def list(self) -> list[str]:
        """List all registered names."""
        return sorted(self._registry.keys())

    def list_with_aliases(self) -> dict[str, list[str]]:
        """List all registered names with their aliases."""
        result = {}
        for name in self._registry:
            aliases = [
                alias
                for alias, target in self._aliases.items()
                if target == name
            ]
            result[name] = aliases
        return result

    def get_metadata(self, name: str) -> dict[str, Any]:
        """Get metadata for a registered class."""
        if name in self._aliases:
            name = self._aliases[name]
        return self._metadata.get(name, {})

    def __contains__(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._registry or name in self._aliases

    def __repr__(self) -> str:
        """Return representation with name and registered items."""
        return f"{self.__class__.__name__}(name='{self.name}', items={self.list()})"


class ModelRegistry(Registry):
    """Registry for energy-based models."""

    def _validate_class(self, cls: type[T]) -> None:
        """Validate that class implements the EnergyModel protocol."""
        # Import here to avoid circular imports

        required_methods = {"energy", "free_energy", "device", "dtype"}
        class_methods = set(dir(cls))

        missing = required_methods - class_methods
        if missing:
            raise TypeError(
                f"Model class {cls.__name__} missing required methods: {missing}"
            )


class SamplerRegistry(Registry):
    """Registry for sampling algorithms."""

    def _validate_class(self, cls: type[T]) -> None:
        """Validate that class implements the Sampler protocol."""
        required_methods = {"sample", "reset"}
        class_methods = set(dir(cls))

        missing = required_methods - class_methods
        if missing:
            raise TypeError(
                f"Sampler class {cls.__name__} missing required methods: {missing}"
            )


class OptimizerRegistry(Registry):
    """Registry for optimizers."""

    def _validate_class(self, cls: type[T]) -> None:
        """Validate that class is an optimizer."""
        # Check if it's a torch optimizer or has similar interface
        if not (hasattr(cls, "step") and hasattr(cls, "zero_grad")):
            raise TypeError(
                f"Optimizer class {cls.__name__} must have 'step' and 'zero_grad' methods"
            )


class TransformRegistry(Registry):
    """Registry for data transforms."""

    def _validate_class(self, cls: type[T]) -> None:
        """Validate that class implements the Transform protocol."""
        required_methods = {"__call__", "inverse"}
        class_methods = set(dir(cls))

        missing = required_methods - class_methods
        if missing:
            raise TypeError(
                f"Transform class {cls.__name__} missing required methods: {missing}"
            )


# Global registry instances
models = ModelRegistry("models")
samplers = SamplerRegistry("samplers")
optimizers = OptimizerRegistry("optimizers")
transforms = TransformRegistry("transforms")


# Convenience decorators
def register_model(name: str, **kwargs: Any) -> Callable[[type], type] | type:
    """Register a model class in the global registry."""
    return models.register(name, **kwargs)


def register_sampler(name: str, **kwargs: Any) -> Callable[[type], type] | type:
    """Register a sampler class in the global registry."""
    return samplers.register(name, **kwargs)


def register_optimizer(name: str, **kwargs: Any) -> Callable[[type], type] | type:
    """Register an optimizer class in the global registry."""
    return optimizers.register(name, **kwargs)


def register_transform(name: str, **kwargs: Any) -> Callable[[type], type] | type:
    """Register a transform class in the global registry."""
    return transforms.register(name, **kwargs)


# Registry discovery utilities
def discover_plugins(module_name: str) -> None:
    """Discover and load plugins from a module.

    This function imports a module and automatically registers any classes
    that have been decorated with registry decorators.

    Args:
        module_name: Name of the module to import
    """
    try:
        import importlib

        importlib.import_module(module_name)
        logger.info(f"Loaded plugins from {module_name}")
    except ImportError as e:
        logger.error(f"Failed to load plugins from {module_name}: {e}")


def get_all_registries() -> dict[str, Registry]:
    """Get all available registries."""
    return {
        "models": models,
        "samplers": samplers,
        "optimizers": optimizers,
        "transforms": transforms,
    }
