from typing import Dict

import numpy
import numpy as np

from ray.rllib.utils.tf_utils import explained_variance, make_tf_callable
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_tf_policy import (
    PPOTF1Policy,
    PPOTF2Policy,
)

other_agents_observation_key: str = 'other_agents_observation'
other_agents_action_key: str = 'other_agents_action'


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        if self.config['framework'] != 'torch':
            self.compute_central_vf = make_tf_callable(self.get_session())(
                self.model.central_value_function
            )
        else:
            self.compute_central_vf = self.model.central_value_function


def centralized_critic_postprocessing(policy, sample_batch, other_agent_batches=None, episode=None, framework=None):
    pytorch = policy.config['framework'] == 'torch'
    agent_id_batch: Dict = {}
    other_agents_observation: Dict = {}
    other_agents_action: Dict = {}
    if (pytorch and hasattr(policy, 'compute_central_vf')) or (not pytorch and policy.loss_initialized()):
        assert other_agent_batches is not None

        if policy.config['enable_connectors']:
            index_batche = 2
        else:
            index_batche = 1

        i: int = 0
        for key in other_agent_batches.keys():
            agent_id_batch['other_agent_' + str(i)] = other_agent_batches[key][index_batche]
            other_agents_observation['other_agent_' + str(i)] = other_agent_batches[key][index_batche][SampleBatch.CUR_OBS]
            other_agents_action['other_agent_' + str(i)] = other_agent_batches[key][index_batche][SampleBatch.ACTIONS]
            i += 1

        # also record the opponent obs and actions in the trajectory
        sample_batch[other_agents_observation_key] = other_agents_observation
        sample_batch[other_agents_action_key] = other_agents_action

        # overwrite default VF prediction with the central VF
        if framework == 'torch':
            sample_batch[SampleBatch.VF_PREDS] = (
                policy.compute_central_vf(
                    convert_to_torch_tensor(sample_batch[SampleBatch.CUR_OBS], policy.device),
                    convert_to_torch_tensor(sample_batch[other_agents_observation_key], policy.device),
                    convert_to_torch_tensor(sample_batch[other_agents_action_key], policy.device),
                )
                .cpu()
                .detach()
                .numpy()
            )
        else:
            sample_batch[SampleBatch.VF_PREDS] = convert_to_numpy(
                policy.compute_central_vf(
                    sample_batch[SampleBatch.CUR_OBS],
                    sample_batch[other_agents_observation_key],
                    sample_batch[other_agents_action_key],
                )
            )
    else:
        # Policy hasn't been initialized yet, use zeros.
        other_agent_observation: Dict[str:numpy.ndarray] = {}
        for key in sample_batch[SampleBatch.CUR_OBS].keys():
            other_agent_observation[key] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS][key])

        other_agents_observation['other_agent_dummy'] = other_agent_observation
        other_agents_action['other_agent_dummy'] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])

        sample_batch[other_agents_observation_key] = other_agents_observation
        sample_batch[other_agents_action_key] = other_agents_action
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    completed = sample_batch[SampleBatch.TERMINATEDS][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config['gamma'],
        policy.config['lambda'],
        use_gae=policy.config['use_gae'],
    )
    return train_batch


def loss_with_central_critic(policy, base_policy, model, dist_class, train_batch):
    # Save original value function.
    vf_saved = model.value_function

    # Calculate loss with a custom value function.
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS],
        train_batch[other_agents_observation_key],
        train_batch[other_agents_action_key],
    )
    policy._central_value_out = model.value_function()
    loss = base_policy.loss(model, dist_class, train_batch)

    # Restore original value function.
    model.value_function = vf_saved

    return loss


def central_vf_stats(policy, train_batch):
    # Report the explained variance of the central value function.
    return {
        'vf_explained_var': explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], policy._central_value_out
        )
    }


def get_ccppo_policy(base):
    class CCPPOTFPolicy(CentralizedValueMixin, base):
        def __init__(self, observation_space, action_space, config):
            base.__init__(self, observation_space, action_space, config)
            CentralizedValueMixin.__init__(self)

        @override(base)
        def loss(self, model, dist_class, train_batch):
            # Use super() to get to the base PPO policy.
            # This special loss function utilizes a shared
            # value function defined on self, and the loss function
            # defined on PPO policies.
            return loss_with_central_critic(
                self, super(), model, dist_class, train_batch
            )

        @override(base)
        def postprocess_trajectory(
                self, sample_batch, other_agent_batches=None, episode=None
        ):
            return centralized_critic_postprocessing(
                self, sample_batch, other_agent_batches, episode, framework='tf'
            )

        @override(base)
        def stats_fn(self, train_batch: SampleBatch):
            stats = super().stats_fn(train_batch)
            stats.update(central_vf_stats(self, train_batch))
            return stats

    return CCPPOTFPolicy


CCPPOStaticGraphTFPolicy = get_ccppo_policy(PPOTF1Policy)
CCPPOEagerTFPolicy = get_ccppo_policy(PPOTF2Policy)


class CCPPOTorchPolicy(CentralizedValueMixin, PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        CentralizedValueMixin.__init__(self)

    @override(PPOTorchPolicy)
    def loss(self, model, dist_class, train_batch):
        return loss_with_central_critic(self, super(), model, dist_class, train_batch)

    @override(PPOTorchPolicy)
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        return centralized_critic_postprocessing(self, sample_batch, other_agent_batches, episode, framework='torch')


class CentralizedCritic(PPO):
    @classmethod
    @override(PPO)
    def get_default_policy_class(cls, config):
        if config['framework'] == 'torch':
            return CCPPOTorchPolicy
        elif config['framework'] == 'tf':
            return CCPPOStaticGraphTFPolicy
        else:
            return CCPPOEagerTFPolicy
