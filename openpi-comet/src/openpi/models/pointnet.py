import dataclasses
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp

Variant = Literal["pcd", "rgb_pcd"]


@dataclasses.dataclass
class Config:
    n_coordinates: int
    output_dim: int
    hidden_dim: int
    hidden_depth: int


def get_config(variant: Variant) -> Config:
    if variant == "pcd":
        return Config(
            n_coordinates=3,
            output_dim=2048,
            hidden_dim=1024,
            hidden_depth=2,
        )
    if variant == "rgb_pcd":
        return Config(
            n_coordinates=3,
            n_color=3,
            output_dim=2048,
            hidden_dim=1024,
            hidden_depth=2,
        )


class MLP(nn.Module):
    hidden_dim: int
    hidden_depth: int
    output_dim: int
    activation: str = "gelu"

    def setup(self):
        layers = []
        in_dim = 3
        for _ in range(self.hidden_depth):
            layers.append(nn.Dense(self.hidden_dim))
            layers.append(self.get_activation())
            in_dim = self.hidden_dim
        layers.append(nn.Dense(self.output_dim))
        layers.append(self.get_activation())
        self.mlp = nn.Sequential(layers)

    def __call__(self, x):
        return self.mlp(x)

    def get_activation(self):
        return getattr(jax.nn, self.activation)


class PointNetSimplified(nn.Module):
    point_channels: int
    output_dim: int
    hidden_dim: int
    hidden_depth: int
    activation: str = "gelu"

    def setup(self):
        self._mlp = MLP(
            hidden_dim=self.hidden_dim,
            hidden_depth=self.hidden_depth,
            output_dim=self.output_dim,
            activation=self.activation,
        )

    def __call__(self, x):
        x = self._mlp(x)
        return jnp.max(x, axis=-2)


class UncoloredPointNet(nn.Module):
    n_coordinates: int = 3
    output_dim: int = 2048
    hidden_dim: int = 2048
    hidden_depth: int = 2
    activation: str = "gelu"
    subtract_mean: bool = False

    def setup(self):
        pn_in_channels = self.n_coordinates
        if self.subtract_mean:
            pn_in_channels += self.n_coordinates
        self.pointnet = PointNetSimplified(
            point_channels=pn_in_channels,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            hidden_depth=self.hidden_depth,
            activation=self.activation,
        )

    def __call__(self, x):
        point = x
        if self.subtract_mean:
            mean = jnp.mean(point, axis=-2, keepdims=True)  # (..., 1, coordinates)
            point = point - mean
            point = jnp.concatenate([point, mean], axis=-1)  # (..., points, 2 * coordinates)
        return self.pointnet(point)


class PointNet(nn.Module):
    n_coordinates: int = 3
    n_color: int = 3
    output_dim: int = 2048
    hidden_dim: int = 2048
    hidden_depth: int = 2
    activation: str = "gelu"
    subtract_mean: bool = False

    def setup(self):
        pn_in_channels = self.n_coordinates + self.n_color
        if self.subtract_mean:
            pn_in_channels += self.n_coordinates
        self.pointnet = PointNetSimplified(
            point_channels=pn_in_channels,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            hidden_depth=self.hidden_depth,
            activation=self.activation,
        )

    def __call__(self, x):
        xyz = x["pcd_xyz"]
        rgb = x["pcd_rgb"]
        point = xyz
        if self.subtract_mean:
            mean = jnp.mean(point, axis=-2, keepdims=True)
            point = point - mean
            point = jnp.concatenate([point, mean], axis=-1)
        x = jnp.concatenate([point, rgb], axis=-1)
        return self.pointnet(x)


if __name__ == "__main__":
    from jax import random

    model = UncoloredPointNet(n_coordinates=3, output_dim=2048, hidden_dim=2048, hidden_depth=2)

    key = random.PRNGKey(0)
    input_data = jnp.ones((2, 256, 3))
    flax_params = model.init(key, input_data)

    __import__("ipdb").set_trace()
    output = model.apply(flax_params, input_data)

    print("Output shape:", output.shape)
