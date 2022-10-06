from typing import Dict, List, Tuple, Type, Union

import torch as th
from stable_baselines3.common.utils import get_device
from torch import nn


class MlpExtractor_Aux(nn.Module):
    """
    analogous to the below MlpExtractor
    """

    def __init__(
            self,
            feature_dim: int,
            net_arch: List[Union[int, Dict[str, List[int]]]],
            activation_fn: Type[nn.Module],
            device: Union[th.device, str] = "auto",
    ):
        super(MlpExtractor_Aux, self).__init__()
        device = get_device(device)
        last_layer_dim_aux = feature_dim
        aux_net = []

        # Simple auxilliary Network
        aux_net.append(nn.Linear(last_layer_dim_aux, 256))
        aux_net.append(activation_fn())
        aux_net.append(nn.Linear(256, 128))
        aux_net.append(activation_fn())
        aux_net.append(nn.Linear(128, 64))
        aux_net.append(activation_fn())
        aux_net.append(nn.Linear(64, 32))
        aux_net.append(activation_fn())
        last_layer_dim_aux = 32

        # Save dim, used to create the distributions
        self.latent_dim_aux = last_layer_dim_aux

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module

        self.aux_net = nn.Sequential(*aux_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """

        return self.aux_net(features)
