from typing import Dict, NamedTuple, Union

import torch as th

TensorDict = Dict[Union[str, int], th.Tensor]


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    aux_angle: th.Tensor
    aux_angle_gt: th.Tensor


class DictRolloutBufferSamples(RolloutBufferSamples):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    aux_angle: th.Tensor
    aux_angle_gt: th.Tensor
