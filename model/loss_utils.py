import torch
from torch import nn
from torch.nn import functional as F

from typing import Tuple, Dict, Optional


def dpo_loss(
    ref_logits: torch.Tensor,
    policy_logits: torch.Tensor,
    labels: torch.LongTensor,
    mask: torch.Tensor
):

    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    ref_log_probs = ref_log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_log_probs = policy_log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    ref_log_probs = (ref_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1e-8)
    policy_log_probs = (policy_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1e-8)
    
    chosen_ref_log_probs, rejected_ref_log_probs = ref_log_probs.chunk(2, dim=0)
    chosen_policy_log_probs, rejected_policy_log_probs = policy_log_probs.chunk(2, dim=0)

    pi_log_probs = chosen_policy_log_probs - rejected_policy_log_probs
    ref_log_probs = chosen_ref_log_probs - rejected_ref_log_probs
    logits = pi_log_probs - ref_log_probs

    return F.logsigmoid(logits).mean()


def compute_grpo_advantage(
    rewards: torch.Tensor,
    masks: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    advantages = rewards.clone()
    batch_size = advantages.shape[0]
    group_size = batch_size // (rewards.shape[0] // masks.sum(dim=1).long().max().item()) if masks.sum() > 0 else batch_size
    if group_size < 2:
        return advantages
    
    for i in range(0, batch_size, group_size):
        end_idx = min(i + group_size, batch_size)
        group_rewards = advantages[i:end_idx]
        valid_count = (masks[i:end_idx].sum() if masks.dim() > 0 else (end_idx - i))
        
        if valid_count > 1:
            mean_reward = group_rewards.mean()
            std_reward = group_rewards.std(unbiased=False)
            advantages[i:end_idx] = (group_rewards - mean_reward) / (std_reward + 1e-8)
    
    return advantages


def grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    masks: torch.Tensor,
    eps: float = 0.2,
    log_prob_clip_range: Tuple[float, float] = (-20, 20)
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    clip_range = torch.full_like(log_probs, eps)
    
    old_log_probs_clamped = old_log_probs.clamp(*log_prob_clip_range)
    log_ratio = (log_probs - old_log_probs_clamped).clamp(*log_prob_clip_range)
    ratio = torch.exp(log_ratio)
    
    policy_loss_1 = -advantages.unsqueeze(-1) * ratio
    policy_loss_2 = -advantages.unsqueeze(-1) * ratio.clamp(1 - eps, 1 + eps)
    policy_loss = torch.max(policy_loss_1, policy_loss_2)
    
    if masks.dim() > 1:
        masked_loss = (policy_loss.squeeze(-1) * masks).sum() / masks.sum().clamp_min(1e-8)
    else:
        valid_count = masks.sum().clamp_min(1)
        masked_loss = (policy_loss.squeeze(-1) * masks).sum() / valid_count
    
    with torch.no_grad():
        clip_fraction = ((ratio - 1.0).abs() > eps).float()
        if masks.dim() > 1:
            clip_fraction = (clip_fraction * masks).sum() / masks.sum().clamp_min(1e-8)
        else:
            clip_fraction = clip_fraction.mean()
    
    metrics = {
        "policy_loss": masked_loss,
        "clip_fraction": clip_fraction,
        "mean_advantage": advantages.mean(),
        "mean_ratio": ratio.mean()
    }
    
    return masked_loss, metrics


def compute_token_level_log_probs(
    logits: torch.Tensor,
    input_ids: torch.LongTensor,
    masks: torch.Tensor
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    
    if masks.dim() == 1:
        masks_2d = masks.unsqueeze(0).expand_as(token_log_probs)
    else:
        masks_2d = masks
    
    masked_log_probs = token_log_probs * masks_2d
    
    return masked_log_probs

