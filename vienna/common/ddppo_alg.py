from collections import defaultdict
from typing import Tuple

import torch
from habitat_baselines.rl.ddppo.algo.ddppo import DDPPO
from torch.functional import Tensor
from torch.nn.functional import l1_loss

import random
import lmdb
import numpy as np
import msgpack_numpy
from torch import distributed as distrib
import json

class WDDPPO(DDPPO):
    """Differences with DD-PPO:
    - expands entropy calculation and tracking to three variables
    - adds a regularization term to the offset prediction
    """

    def __init__(
        self,
        *args,
        offset_regularize_coef: float = 0.0,
        pano_entropy_coef: float = 1.0,
        offset_entropy_coef: float = 1.0,
        distance_entropy_coef: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.offset_regularize_coef = offset_regularize_coef
        self.pano_entropy_coef = pano_entropy_coef
        self.offset_entropy_coef = offset_entropy_coef
        self.distance_entropy_coef = distance_entropy_coef

    def get_advantages(self, rollouts) -> Tensor:
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    def update(self, rollouts) -> Tuple[float, float, float]:
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        entropy_loss_epoch = 0.0
        pano_entropy_epoch = 0.0
        offset_entropy_epoch = 0.0
        distance_entropy_epoch = 0.0

        for _e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample
                N, _, _ = recurrent_hidden_states_batch.shape
                # Reshape to do in a single forward pass for all steps
                # (
                #     values,
                #     action_log_probs,
                #     entropy,
                #     _,
                # ) = self.actor_critic.evaluate_actions(
                #     obs_batch,
                #     recurrent_hidden_states_batch,
                #     prev_actions_batch,
                #     masks_batch,
                #     actions_batch,
                # )
                obs_batch = dict(obs_batch)
                prev_actions_batch = dict(prev_actions_batch)
                actions_batch = dict(actions_batch)

                (
                    values,
                    action_log_probs,
                    entropy,
                    _,
                ) = self._evaluate_actions(
                    obs_batch,
                    # recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )
                # print('value', values.shape, 'log_prob', action_log_probs.shape)
                masks_batch = masks_batch.view(-1, N, 1)
                masks_batch[0,:,0] = 1.0
                masks_batch = masks_batch.view(-1, 1)

                entropy_loss = (
                    self.pano_entropy_coef * entropy["pano"]
                    + self.offset_entropy_coef * entropy["offset"]
                    + self.distance_entropy_coef * entropy["distance"]
                ).mean() * self.entropy_coef

                # action_log_probs = torch.clamp(action_log_probs, min=-5)
                # old_action_log_probs_batch = torch.clamp(old_action_log_probs_batch, min=-5)
                # if (action_log_probs.view(-1,1) - old_action_log_probs_batch.view(-1,1)).max() > 5:
                #     print('action_log_probs', action_log_probs.view(-1)) 
                #     print('old_action_log_probs_batch', old_action_log_probs_batch.view(-1))

                ratio = torch.exp(
                    action_log_probs.view(-1,1) - old_action_log_probs_batch.view(-1,1)
                )
                surr1 = ratio * adv_targ.view(-1,1)
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ.view(-1,1)
                )
                # action_loss = -torch.min(surr1, surr2).mean()
                action_loss =  -torch.min(surr1, surr2) * masks_batch.float()

                action_loss = action_loss.sum() / masks_batch.float().sum()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch.view(-1,1) + (
                        values.view(-1,1) - value_preds_batch.view(-1,1)
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values.view(-1,1) - return_batch.view(-1,1)).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped.view(-1,1) - return_batch.view(-1,1)
                    ).pow(2)
                    # value_loss = (
                    #     0.5
                    #     * torch.max(value_losses, value_losses_clipped).mean()
                    # )
                    value_loss = 0.5 * (torch.max(value_losses, value_losses_clipped) * masks_batch.float()).sum() / masks_batch.float().sum()
                else:
                    # value_loss = 0.5 * (return_batch - values).pow(2).mean()
                    value_loss = (0.5 * (return_batch.view(-1,1) - values.view(-1,1)).pow(2) * masks_batch.float()).sum() / masks_batch.float().sum() 
                value_loss = value_loss * self.value_loss_coef

                # slight regularization to the offset
                offset_loss = 0.0
                if "offset" in actions_batch:
                    offset_loss = self.offset_regularize_coef * (l1_loss(
                        self.actor_critic.net.offset_to_continuous(
                            actions_batch["offset"]
                        ),
                        torch.zeros_like(actions_batch["offset"]),
                        reduction='none'
                    ) * masks_batch.float()).sum() / masks_batch.float().sum() 

                self.optimizer.zero_grad()
                loss = value_loss + action_loss + offset_loss - entropy_loss

                self.before_backward(loss)
                loss.backward()
                self.after_backward(loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                entropy_loss_epoch += entropy_loss.item()
                pano_entropy_epoch += entropy["pano"].mean().item()
                offset_entropy_epoch += entropy["offset"].mean().item()
                distance_entropy_epoch += entropy["distance"].mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        return (
            value_loss_epoch / num_updates,
            action_loss_epoch / num_updates,
            entropy_loss_epoch / num_updates,
            pano_entropy_epoch / num_updates,
            offset_entropy_epoch / num_updates,
            distance_entropy_epoch / num_updates,
        )

    def update_teacher(self, rollouts) -> Tuple[float, float, float]:
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        entropy_loss_epoch = 0.0
        pano_entropy_epoch = 0.0
        offset_entropy_epoch = 0.0
        distance_entropy_epoch = 0.0

        for _e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample
                N, _, _ = recurrent_hidden_states_batch.shape
                # Reshape to do in a single forward pass for all steps
                # (
                #     values,
                #     action_log_probs,
                #     entropy,
                #     _,
                # ) = self.actor_critic.evaluate_actions(
                #     obs_batch,
                #     recurrent_hidden_states_batch,
                #     prev_actions_batch,
                #     masks_batch,
                #     actions_batch,
                # )
                obs_batch = dict(obs_batch)
                prev_actions_batch = dict(prev_actions_batch)
                actions_batch = dict(actions_batch)

                (
                    values,
                    action_log_probs,
                    entropy,
                    _,
                ) = self._evaluate_actions(
                    obs_batch,
                    # recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )
                # print('value', values.shape, 'log_prob', action_log_probs.shape)
                masks_batch = masks_batch.view(-1, N, 1)
                masks_batch[0,:,0] = 1.0
                masks_batch = masks_batch.view(-1, 1)

                entropy_loss = (
                    self.pano_entropy_coef * entropy["pano"]
                    + self.offset_entropy_coef * entropy["offset"]
                    + self.distance_entropy_coef * entropy["distance"]
                ).mean() * self.entropy_coef


                # ratio = torch.exp(
                #     action_log_probs.view(-1,1) - old_action_log_probs_batch.view(-1,1)
                # )
                # surr1 = ratio * adv_targ.view(-1,1)
                # surr2 = (
                #     torch.clamp(
                #         ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                #     )
                #     * adv_targ.view(-1,1)
                # )
                # action_loss = -torch.min(surr1, surr2).mean()

                # action_loss =  -torch.min(surr1, surr2) * masks_batch.float()
                action_loss = -action_log_probs.view(-1,1) * masks_batch.float()

                action_loss = action_loss.sum() / masks_batch.float().sum()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch.view(-1,1) + (
                        values.view(-1,1) - value_preds_batch.view(-1,1)
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values.view(-1,1) - return_batch.view(-1,1)).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped.view(-1,1) - return_batch.view(-1,1)
                    ).pow(2)
                    # value_loss = (
                    #     0.5
                    #     * torch.max(value_losses, value_losses_clipped).mean()
                    # )
                    value_loss = 0.5 * (torch.max(value_losses, value_losses_clipped) * masks_batch.float()).sum() / masks_batch.float().sum()
                else:
                    # value_loss = 0.5 * (return_batch - values).pow(2).mean()
                    value_loss = (0.5 * (return_batch.view(-1,1) - values.view(-1,1)).pow(2) * masks_batch.float()).sum() / masks_batch.float().sum() 
                value_loss = value_loss * self.value_loss_coef
                value_loss = value_loss * 0.

                # slight regularization to the offset
                offset_loss = 0.0
                if "offset" in actions_batch:
                    offset_loss = self.offset_regularize_coef * (l1_loss(
                        self.actor_critic.net.offset_to_continuous(
                            actions_batch["offset"]
                        ),
                        torch.zeros_like(actions_batch["offset"]),
                        reduction='none'
                    ) * masks_batch.float()).sum() / masks_batch.float().sum() 

                self.optimizer.zero_grad()
                loss = value_loss + action_loss + offset_loss - entropy_loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print('value loss', value_loss)
                    print('action loss', action_loss)
                    print('offset loss', offset_loss)
                    print('entropy loss', entropy_loss)
                    print('return_batch', return_batch)
                    print('values',values)

                self.before_backward(loss)
                loss.backward()
                self.after_backward(loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                entropy_loss_epoch += entropy_loss.item()
                pano_entropy_epoch += entropy["pano"].mean().item()
                offset_entropy_epoch += entropy["offset"].mean().item()
                distance_entropy_epoch += entropy["distance"].mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        return (
            value_loss_epoch / num_updates,
            action_loss_epoch / num_updates,
            entropy_loss_epoch / num_updates,
            pano_entropy_epoch / num_updates,
            offset_entropy_epoch / num_updates,
            distance_entropy_epoch / num_updates,
        )


class MTDDPPO(DDPPO):
    """Differences with DD-PPO:
    - expands entropy calculation and tracking to three variables
    - adds a regularization term to the offset prediction
    """

    def __init__(
        self,
        *args,
        offset_regularize_coef: float = 0.0,
        pano_entropy_coef: float = 1.0,
        offset_entropy_coef: float = 1.0,
        distance_entropy_coef: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.offset_regularize_coef = offset_regularize_coef
        self.pano_entropy_coef = pano_entropy_coef
        self.offset_entropy_coef = offset_entropy_coef
        self.distance_entropy_coef = distance_entropy_coef

    def get_advantages(self, rollouts) -> Tensor:
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    @property
    def ddpmodel(self):
        return self._evaluate_actions_wrapper.ddp.module.actor_critic

    def update(self, rollouts) -> Tuple[float, float, float]:
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        entropy_loss_epoch = 0.0
        pano_entropy_epoch = 0.0
        offset_entropy_epoch = 0.0
        distance_entropy_epoch = 0.0

        world_size = distrib.get_world_size()

        for _e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample
                N, _, _ = recurrent_hidden_states_batch.shape
                # Reshape to do in a single forward pass for all steps
                # (
                #     values,
                #     action_log_probs,
                #     entropy,
                #     _,
                # ) = self.actor_critic.evaluate_actions(
                #     obs_batch,
                #     recurrent_hidden_states_batch,
                #     prev_actions_batch,
                #     masks_batch,
                #     actions_batch,
                # )
                obs_batch = dict(obs_batch)
                prev_actions_batch = dict(prev_actions_batch)
                actions_batch = dict(actions_batch)

                (
                    values,
                    action_log_probs,
                    entropy,
                    _,
                ) = self._evaluate_actions(
                # ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    # recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )
                # print('value', values.shape, 'log_prob', action_log_probs.shape)
                masks_batch = masks_batch.view(-1, N, 1)
                masks_batch[0,:,0] = 1.0
                masks_batch = masks_batch.view(-1, 1)

                entropy_loss = (
                    self.pano_entropy_coef * entropy["pano"]
                    + self.offset_entropy_coef * entropy["offset"]
                    + self.distance_entropy_coef * entropy["distance"]
                ).mean() * self.entropy_coef

                # action_log_probs = torch.clamp(action_log_probs, min=-5)
                # old_action_log_probs_batch = torch.clamp(old_action_log_probs_batch, min=-5)
                # if (action_log_probs.view(-1,1) - old_action_log_probs_batch.view(-1,1)).max() > 5:
                #     print('action_log_probs', action_log_probs.view(-1)) 
                #     print('old_action_log_probs_batch', old_action_log_probs_batch.view(-1))

                ratio = torch.exp(
                    action_log_probs.view(-1,1) - old_action_log_probs_batch.view(-1,1)
                )
                surr1 = ratio * adv_targ.view(-1,1)
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ.view(-1,1)
                )
                # action_loss = -torch.min(surr1, surr2).mean()
                action_loss =  -torch.min(surr1, surr2) * masks_batch.float()

                action_loss = action_loss.sum() / masks_batch.float().sum()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch.view(-1,1) + (
                        values.view(-1,1) - value_preds_batch.view(-1,1)
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values.view(-1,1) - return_batch.view(-1,1)).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped.view(-1,1) - return_batch.view(-1,1)
                    ).pow(2)
                    # value_loss = (
                    #     0.5
                    #     * torch.max(value_losses, value_losses_clipped).mean()
                    # )
                    value_loss = 0.5 * (torch.max(value_losses, value_losses_clipped) * masks_batch.float()).sum() / masks_batch.float().sum()
                else:
                    # value_loss = 0.5 * (return_batch - values).pow(2).mean()
                    value_loss = (0.5 * (return_batch.view(-1,1) - values.view(-1,1)).pow(2) * masks_batch.float()).sum() / masks_batch.float().sum() 
                value_loss = value_loss * self.value_loss_coef

                # slight regularization to the offset
                offset_loss = 0.0
                if "offset" in actions_batch:
                    offset_loss = self.offset_regularize_coef * (l1_loss(
                        self.actor_critic.net.offset_to_continuous(
                            actions_batch["offset"]
                        ),
                        torch.zeros_like(actions_batch["offset"]),
                        reduction='none'
                    ) * masks_batch.float()).sum() / masks_batch.float().sum() 

                self.optimizer.zero_grad()
                loss = value_loss + action_loss + offset_loss - entropy_loss

                if 'instruction' in obs_batch:
                    task_id = 0
                elif 'spectrogram' in obs_batch:
                    task_id = 1
                elif 'imagegoal' in obs_batch:
                    task_id = 2
                elif 'objectgoal' in obs_batch:
                    task_id = 3

                # backward loss
                self.ddpmodel.net.model.backward(loss, task_id)

                # params = self.actor_critic.parameters()

                # for p in params:
                #     if p.grad is not None:
                #         grad = p.grad.clone()
                #     else:
                #         grad = torch.zeros_like(p.data)
                #     distrib.all_reduce(grad) 
                #     grad = grad / world_size
                #     p.grad = grad
                

                # self.before_backward(loss)
                # loss.backward()
                # self.after_backward(loss)

                ### start compute grad
                
                

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                entropy_loss_epoch += entropy_loss.item()
                pano_entropy_epoch += entropy["pano"].mean().item()
                offset_entropy_epoch += entropy["offset"].mean().item()
                distance_entropy_epoch += entropy["distance"].mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        return (
            value_loss_epoch / num_updates,
            action_loss_epoch / num_updates,
            entropy_loss_epoch / num_updates,
            pano_entropy_epoch / num_updates,
            offset_entropy_epoch / num_updates,
            distance_entropy_epoch / num_updates,
        )


class CMADDPPO(DDPPO):
    
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
    
    def get_advantages(self, rollouts) -> Tensor:
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + 1e-5)
    
    def update(self, rollouts) -> Tuple[float, float, float]:
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        entropy_loss_epoch = 0.0

        for _e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    entropy,
                    _,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )

                entropy_loss = entropy.mean() * self.entropy_coef

                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                value_loss = value_loss * self.value_loss_coef

    
                self.optimizer.zero_grad()
                loss = value_loss + action_loss - entropy_loss

                self.before_backward(loss)
                loss.backward()
                self.after_backward(loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                entropy_loss_epoch += entropy_loss.item()
    

        num_updates = self.ppo_epoch * self.num_mini_batch
        return (
            value_loss_epoch / num_updates,
            action_loss_epoch / num_updates,
            entropy_loss_epoch / num_updates,
        )


def fn(batch):
    rgb_feat_list = []
    depth_feat_list = []
    for rgb_feat, depth_feat in batch:
        rgb_feat_list.append(rgb_feat)
        depth_feat_list.append(depth_feat)

    rgb_feats = torch.stack(rgb_feat_list, 0) 
    depth_feats = torch.stack(depth_feat_list, 0)

    return rgb_feats, depth_feats


def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]



class FramesDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        lmdb_map_size=1e9,
        batch_size=100,
    ):
        super().__init__()
        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 10
        self._preload = []
        self.batch_size = batch_size

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
        ) as lmdb_env:
            self.length = lmdb_env.stat()["entries"]

    def _load_next(self):
        if len(self._preload) <  self.batch_size:
            if len(self.load_ordering) == 0:
                self.load_ordering = list(
                    reversed(_block_shuffle(list(range(self.start, self.end)), self.preload_size))
                )

            new_preload = []
            # lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                # for _ in range(self.preload_size):
                while len(new_preload) < self.preload_size:
                    if len(self.load_ordering) == 0:
                        break
                    item = msgpack_numpy.unpackb(
                            txn.get(str(self.load_ordering.pop()).encode()), raw=False
                        )
                    obs, _, _ = item
                    rgb_feat = obs['rgb_features']
                    depth_feat = obs['depth_features']
                    n,c,_,_,_ = rgb_feat.shape
                    rgb_feat = rgb_feat.reshape((n * c,*(rgb_feat.shape[2:])))
                    depth_feat = depth_feat.reshape((n * c,*(depth_feat.shape[2:])))
                    for rgb, depth in zip(rgb_feat, depth_feat):
                        
                        new_preload.append(
                            (rgb, depth)
                        )

                    # lengths.append(len(new_preload[-1][0]))

            # sort_priority = list(range(len(lengths)))
            # random.shuffle(sort_priority)

            # sorted_ordering = list(range(len(lengths)))
            # sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))
            sorted_ordering = list(range(len(new_preload)))
            random.shuffle(sorted_ordering)

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

        return self._preload.pop()

    def __next__(self):
        rgb_feat, depth_feat = self._load_next()

        rgb_feat = torch.from_numpy(np.copy(rgb_feat))
        depth_feat = torch.from_numpy(np.copy(depth_feat))

        return (rgb_feat, depth_feat)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.start = 0
            self.end = self.length
        else:
            per_worker = int(np.ceil(self.length / worker_info.num_workers))

            self.start = per_worker * worker_info.id
            self.end = min(self.start + per_worker, self.length)

        # Reverse so we can use .pop()
        self.load_ordering = list(
            reversed(_block_shuffle(list(range(self.start, self.end)), self.preload_size))
        )
        # print(len(self.load_ordering))

        return self


class CWDDPPO(WDDPPO):
    def __init__(self, *args, offset_regularize_coef: float = 0, pano_entropy_coef: float = 1, offset_entropy_coef: float = 1, distance_entropy_coef: float = 1, **kwargs) -> None:
        super().__init__(*args, offset_regularize_coef=offset_regularize_coef, pano_entropy_coef=pano_entropy_coef, offset_entropy_coef=offset_entropy_coef, distance_entropy_coef=distance_entropy_coef, **kwargs)
        batch_size = 200
        lmdb_features_dir = 'data/eccv_frame_lmdb_resnet18'
        self.dataset = FramesDataset(
                        lmdb_features_dir,
                        batch_size=200
                    )
        diter = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=fn,
                pin_memory=False,
                drop_last=True,  # drop last batch if smaller
                num_workers=3,
            )
        self.diter = iter(diter)
    
    def update(self, rollouts) -> Tuple[float, float, float]:
        advantages = self.get_advantages(rollouts)
        t_device = advantages.device
        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        entropy_loss_epoch = 0.0
        pano_entropy_epoch = 0.0
        offset_entropy_epoch = 0.0
        distance_entropy_epoch = 0.0

        for _e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample
                rgb_feats, depth_feats = next(self.diter)
                rgb_feats = rgb_feats.to(t_device)
                depth_feats = depth_feats.to(t_device)
                obs_batch['neg_rgb_features'] = rgb_feats
                obs_batch['neg_depth_features'] = depth_feats
                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    entropy,
                    _,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )

                entropy_loss = (
                    self.pano_entropy_coef * entropy["pano"]
                    + self.offset_entropy_coef * entropy["offset"]
                    + self.distance_entropy_coef * entropy["distance"]
                ).mean() * self.entropy_coef

                action_log_probs = torch.clamp(action_log_probs, min=-4)
                old_action_log_probs_batch = torch.clamp(old_action_log_probs_batch, min=-4)
                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                value_loss = value_loss * self.value_loss_coef

                # slight regularization to the offset
                offset_loss = 0.0
                if "offset" in actions_batch:
                    offset_loss = self.offset_regularize_coef * l1_loss(
                        self.actor_critic.net.offset_to_continuous(
                            actions_batch["offset"]
                        ),
                        torch.zeros_like(actions_batch["offset"]),
                    )

                self.optimizer.zero_grad()
                loss = value_loss + action_loss + offset_loss - entropy_loss


                self.before_backward(loss)
                loss.backward()
                self.after_backward(loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                entropy_loss_epoch += entropy_loss.item()
                pano_entropy_epoch += entropy["pano"].mean().item()
                offset_entropy_epoch += entropy["offset"].mean().item()
                distance_entropy_epoch += entropy["distance"].mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        return (
            value_loss_epoch / num_updates,
            action_loss_epoch / num_updates,
            entropy_loss_epoch / num_updates,
            pano_entropy_epoch / num_updates,
            offset_entropy_epoch / num_updates,
            distance_entropy_epoch / num_updates,
        )

class ILP(WDDPPO):
    def update(self, rollouts):
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        entropy_loss_epoch = 0.0
        pano_entropy_epoch = 0.0
        offset_entropy_epoch = 0.0
        distance_entropy_epoch = 0.0

        for _e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    entropy,
                    _,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )

                entropy_loss = (
                    self.pano_entropy_coef * entropy["pano"]
                    + self.offset_entropy_coef * entropy["offset"]
                    + self.distance_entropy_coef * entropy["distance"]
                ).mean() * self.entropy_coef

                # ratio = torch.exp(
                #     action_log_probs - old_action_log_probs_batch
                # )
                # surr1 = ratio * adv_targ
                # surr2 = (
                #     torch.clamp(
                #         ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                #     )
                #     * adv_targ
                # )
                # action_loss = -torch.min(surr1, surr2).mean()

                action_loss = -action_log_probs.mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                value_loss = value_loss * self.value_loss_coef

                # slight regularization to the offset
                offset_loss = 0.0
                if "offset" in actions_batch:
                    offset_loss = self.offset_regularize_coef * l1_loss(
                        self.actor_critic.net.offset_to_continuous(
                            actions_batch["offset"]
                        ),
                        torch.zeros_like(actions_batch["offset"]),
                    )

                self.optimizer.zero_grad()
                # loss = value_loss + action_loss + offset_loss - entropy_loss
                loss = action_loss
                # print('loss', loss, 'value_loss', value_loss, 'action_loss', action_loss, 'offset_loss', offset_loss, 'entropy loss', entropy_loss)
                self.before_backward(loss)
                loss.backward()
                self.after_backward(loss)
                
                params = list(filter(lambda p: p[1].requires_grad, self.actor_critic.named_parameters()))
                # for name, p in params:
                #     if p.grad is not None:
                #         if torch.isnan(p.grad).any():
                #             print(name)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                entropy_loss_epoch += entropy_loss.item()
                pano_entropy_epoch += entropy["pano"].mean().item()
                offset_entropy_epoch += entropy["offset"].mean().item()
                distance_entropy_epoch += entropy["distance"].mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        return (
            value_loss_epoch / num_updates,
            action_loss_epoch / num_updates,
            entropy_loss_epoch / num_updates,
            pano_entropy_epoch / num_updates,
            offset_entropy_epoch / num_updates,
            distance_entropy_epoch / num_updates,
        )
