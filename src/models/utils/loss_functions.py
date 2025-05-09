import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

def huber_loss(x: torch.Tensor, y: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """
    ฟังก์ชันการสูญเสียแบบ Huber (Huber Loss)
    
    คล้ายกับ MSE สำหรับค่าที่เล็ก แต่จะมีความทนทานต่อ outliers มากกว่า
    
    Parameters:
    x (torch.Tensor): ค่าที่ทำนาย
    y (torch.Tensor): ค่าเป้าหมาย
    delta (float): พารามิเตอร์ที่กำหนดจุดเปลี่ยนจาก quadratic เป็น linear
    
    Returns:
    torch.Tensor: ค่าการสูญเสีย
    """
    # คำนวณความต่าง
    diff = x - y
    
    # คำนวณการสูญเสียแบบ Huber
    cond = diff.abs() < delta
    loss = torch.where(cond, 0.5 * diff.pow(2), delta * (diff.abs() - 0.5 * delta))
    
    return loss.mean()

def mse_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    ฟังก์ชันการสูญเสียแบบ Mean Squared Error (MSE)
    
    Parameters:
    x (torch.Tensor): ค่าที่ทำนาย
    y (torch.Tensor): ค่าเป้าหมาย
    
    Returns:
    torch.Tensor: ค่าการสูญเสีย
    """
    return F.mse_loss(x, y)

def quantile_huber_loss(x: torch.Tensor, y: torch.Tensor, tau: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """
    ฟังก์ชันการสูญเสียแบบ Quantile Huber
    
    ใช้ใน QR-DQN (Quantile Regression DQN)
    
    Parameters:
    x (torch.Tensor): ค่าที่ทำนาย
    y (torch.Tensor): ค่าเป้าหมาย
    tau (torch.Tensor): ค่า quantile
    kappa (float): พารามิเตอร์ที่กำหนดจุดเปลี่ยนจาก quadratic เป็น linear
    
    Returns:
    torch.Tensor: ค่าการสูญเสีย
    """
    # คำนวณความต่าง
    diff = y - x
    
    # คำนวณความผิดพลาดแบบ Huber
    huber_e = torch.where(diff.abs() <= kappa, 0.5 * diff.pow(2), kappa * (diff.abs() - 0.5 * kappa))
    
    # คำนวณการสูญเสียแบบ Quantile Huber
    loss = torch.abs(tau - (diff < 0).float()) * huber_e
    
    return loss.mean()

def ppo_loss(ratio: torch.Tensor, advantage: torch.Tensor, clip_param: float = 0.2) -> torch.Tensor:
    """
    ฟังก์ชันการสูญเสียสำหรับ Proximal Policy Optimization (PPO)
    
    Parameters:
    ratio (torch.Tensor): อัตราส่วนของความน่าจะเป็นการกระทำใหม่ต่อความน่าจะเป็นการกระทำเก่า
    advantage (torch.Tensor): ค่า advantage
    clip_param (float): พารามิเตอร์สำหรับการ clip
    
    Returns:
    torch.Tensor: ค่าการสูญเสีย
    """
    # คำนวณการสูญเสียแบบ clipped
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
    loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
    
    return loss.mean()

def entropy_loss(probs: torch.Tensor, log_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    ฟังก์ชันการสูญเสียแบบ entropy
    
    Parameters:
    probs (torch.Tensor): ความน่าจะเป็นของการกระทำ
    log_probs (torch.Tensor, optional): log ของความน่าจะเป็นของการกระทำ
    
    Returns:
    torch.Tensor: ค่าการสูญเสีย
    """
    if log_probs is None:
        # คำนวณ log_probs จาก probs
        log_probs = torch.log(probs + 1e-10)  # เพิ่ม epsilon เพื่อป้องกันการหาร 0
    
    # คำนวณ entropy
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    return -entropy.mean()  # ลบเพื่อให้ได้การสูญเสีย (ต้องการเพิ่ม entropy)

def log_probability_loss(log_probs: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
    """
    ฟังก์ชันการสูญเสียสำหรับ Policy Gradient
    
    Parameters:
    log_probs (torch.Tensor): log ของความน่าจะเป็นของการกระทำ
    advantage (torch.Tensor): ค่า advantage
    
    Returns:
    torch.Tensor: ค่าการสูญเสีย
    """
    return -(log_probs * advantage).mean()

def calculate_dqn_loss(current_q: torch.Tensor, target_q: torch.Tensor, 
                      loss_type: str = 'huber', **kwargs) -> torch.Tensor:
    """
    คำนวณการสูญเสียสำหรับ DQN
    
    Parameters:
    current_q (torch.Tensor): ค่า Q ปัจจุบัน
    target_q (torch.Tensor): ค่า Q เป้าหมาย
    loss_type (str): ประเภทของการสูญเสีย ('huber', 'mse')
    **kwargs: พารามิเตอร์เพิ่มเติมสำหรับฟังก์ชันการสูญเสีย
    
    Returns:
    torch.Tensor: ค่าการสูญเสีย
    """
    if loss_type == 'huber':
        return huber_loss(current_q, target_q, **kwargs)
    elif loss_type == 'mse':
        return mse_loss(current_q, target_q)
    elif loss_type == 'quantile_huber':
        return quantile_huber_loss(current_q, target_q, **kwargs)
    else:
        return huber_loss(current_q, target_q, **kwargs)

def calculate_policy_gradient_loss(log_probs: torch.Tensor, advantage: torch.Tensor, 
                                 entropy_coef: float = 0.01, probs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    คำนวณการสูญเสียสำหรับ Policy Gradient
    
    Parameters:
    log_probs (torch.Tensor): log ของความน่าจะเป็นของการกระทำ
    advantage (torch.Tensor): ค่า advantage
    entropy_coef (float): สัมประสิทธิ์สำหรับค่า entropy
    probs (torch.Tensor, optional): ความน่าจะเป็นของการกระทำ
    
    Returns:
    torch.Tensor: ค่าการสูญเสีย
    """
    # คำนวณการสูญเสียจาก log probability
    policy_loss = log_probability_loss(log_probs, advantage)
    
    # คำนวณ entropy loss ถ้ามีการระบุ probs
    if probs is not None:
        entropy = entropy_loss(probs, log_probs)
        return policy_loss + entropy_coef * entropy
    else:
        return policy_loss

def calculate_advantage(rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor, 
                       dones: torch.Tensor, gamma: float = 0.99, lambda_gae: float = 0.95) -> torch.Tensor:
    """
    คำนวณค่า advantage ด้วย Generalized Advantage Estimation (GAE)
    
    Parameters:
    rewards (torch.Tensor): รางวัล
    values (torch.Tensor): ค่า value ของสถานะปัจจุบัน
    next_values (torch.Tensor): ค่า value ของสถานะถัดไป
    dones (torch.Tensor): สถานะจบ
    gamma (float): discount factor
    lambda_gae (float): พารามิเตอร์ lambda สำหรับ GAE
    
    Returns:
    torch.Tensor: ค่า advantage
    """
    # คำนวณค่า TD error
    deltas = rewards + gamma * next_values * (1 - dones) - values
    
    # คำนวณค่า advantage ด้วย GAE
    advantages = torch.zeros_like(deltas)
    advantage = 0
    
    for t in reversed(range(len(deltas))):
        advantage = deltas[t] + gamma * lambda_gae * (1 - dones[t]) * advantage
        advantages[t] = advantage
    
    return advantages

def discounted_returns(rewards: torch.Tensor, dones: torch.Tensor, 
                      gamma: float = 0.99, normalize: bool = True) -> torch.Tensor:
    """
    คำนวณผลตอบแทนสะสม (discounted returns)
    
    Parameters:
    rewards (torch.Tensor): รางวัล
    dones (torch.Tensor): สถานะจบ
    gamma (float): discount factor
    normalize (bool): ทำ normalization หรือไม่
    
    Returns:
    torch.Tensor: ผลตอบแทนสะสม
    """
    # คำนวณผลตอบแทนสะสม
    returns = torch.zeros_like(rewards)
    return_sum = 0
    
    for t in reversed(range(len(rewards))):
        if dones[t]:
            return_sum = 0
        return_sum = rewards[t] + gamma * return_sum
        returns[t] = return_sum
    
    # ทำ normalization
    if normalize and len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    return returns

# โค้ดเพิ่มเติมสำหรับฟังก์ชันการสูญเสียเฉพาะสำหรับอัลกอริทึมที่ซับซ้อนมากขึ้น

def td3_actor_loss(actor: torch.nn.Module, critic: torch.nn.Module, 
                  states: torch.Tensor) -> torch.Tensor:
    """
    ฟังก์ชันการสูญเสียสำหรับ actor ใน TD3 (Twin Delayed DDPG)
    
    Parameters:
    actor (torch.nn.Module): เครือข่าย actor
    critic (torch.nn.Module): เครือข่าย critic
    states (torch.Tensor): สถานะ
    
    Returns:
    torch.Tensor: ค่าการสูญเสีย
    """
    # ได้การกระทำจาก actor
    actions = actor(states)
    
    # คำนวณค่า Q จาก critic
    q_values = critic(states, actions)
    
    # ต้องการเพิ่มค่า Q (ลบเพราะต้องการลดการสูญเสีย)
    actor_loss = -q_values.mean()
    
    return actor_loss

def td3_critic_loss(critic1: torch.nn.Module, critic2: torch.nn.Module, 
                   target_critic1: torch.nn.Module, target_critic2: torch.nn.Module,
                   target_actor: torch.nn.Module, 
                   states: torch.Tensor, actions: torch.Tensor, 
                   rewards: torch.Tensor, next_states: torch.Tensor, 
                   dones: torch.Tensor, gamma: float = 0.99,
                   target_noise: float = 0.2, noise_clip: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ฟังก์ชันการสูญเสียสำหรับ critic ใน TD3 (Twin Delayed DDPG)
    
    Parameters:
    critic1 (torch.nn.Module): เครือข่าย critic 1
    critic2 (torch.nn.Module): เครือข่าย critic 2
    target_critic1 (torch.nn.Module): เครือข่าย target critic 1
    target_critic2 (torch.nn.Module): เครือข่าย target critic 2
    target_actor (torch.nn.Module): เครือข่าย target actor
    states (torch.Tensor): สถานะปัจจุบัน
    actions (torch.Tensor): การกระทำ
    rewards (torch.Tensor): รางวัล
    next_states (torch.Tensor): สถานะถัดไป
    dones (torch.Tensor): สถานะจบ
    gamma (float): discount factor
    target_noise (float): ความแปรปรวนของสัญญาณรบกวนที่เพิ่มใน target action
    noise_clip (float): ค่าสูงสุดของสัญญาณรบกวน
    
    Returns:
    Tuple[torch.Tensor, torch.Tensor]: ค่าการสูญเสียของ critic 1 และ critic 2
    """
    with torch.no_grad():
        # ได้การกระทำถัดไปจาก target actor และเพิ่มสัญญาณรบกวน
        next_actions = target_actor(next_states)
        noise = torch.randn_like(next_actions) * target_noise
        noise = noise.clamp(-noise_clip, noise_clip)
        next_actions = (next_actions + noise).clamp(-1, 1)
        
        # คำนวณค่า Q ถัดไปจาก target critic
        next_q1 = target_critic1(next_states, next_actions)
        next_q2 = target_critic2(next_states, next_actions)
        
        # ใช้ค่า Q ที่น้อยกว่าเพื่อลดการประมาณค่าที่สูงเกินไป
        next_q = torch.min(next_q1, next_q2)
        target_q = rewards + gamma * (1 - dones) * next_q
    
    # คำนวณค่า Q ปัจจุบัน
    current_q1 = critic1(states, actions)
    current_q2 = critic2(states, actions)
    
    # คำนวณการสูญเสีย
    critic1_loss = F.mse_loss(current_q1, target_q)
    critic2_loss = F.mse_loss(current_q2, target_q)
    
    return critic1_loss, critic2_loss

def sac_actor_loss(actor_network: torch.nn.Module, critic1: torch.nn.Module, critic2: torch.nn.Module, 
                 states: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
    """
    ฟังก์ชันการสูญเสียสำหรับ actor ใน SAC (Soft Actor-Critic)
    
    Parameters:
    actor_network (torch.nn.Module): เครือข่าย actor
    critic1 (torch.nn.Module): เครือข่าย critic 1
    critic2 (torch.nn.Module): เครือข่าย critic 2
    states (torch.Tensor): สถานะ
    alpha (float): พารามิเตอร์ temperature
    
    Returns:
    torch.Tensor: ค่าการสูญเสีย
    """
    # ได้การกระทำและ log probability จาก actor
    actions, log_probs = actor_network.sample(states)
    
    # คำนวณค่า Q จาก critic
    q1 = critic1(states, actions)
    q2 = critic2(states, actions)
    q = torch.min(q1, q2)
    
    # คำนวณการสูญเสีย (ต้องการเพิ่มค่า Q และ entropy)
    actor_loss = (alpha * log_probs - q).mean()
    
    return actor_loss

def sac_critic_loss(critic1: torch.nn.Module, critic2: torch.nn.Module, 
                  target_critic1: torch.nn.Module, target_critic2: torch.nn.Module,
                  actor_network: torch.nn.Module, 
                  states: torch.Tensor, actions: torch.Tensor, 
                  rewards: torch.Tensor, next_states: torch.Tensor, 
                  dones: torch.Tensor, gamma: float = 0.99, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ฟังก์ชันการสูญเสียสำหรับ critic ใน SAC (Soft Actor-Critic)
    
    Parameters:
    critic1 (torch.nn.Module): เครือข่าย critic 1
    critic2 (torch.nn.Module): เครือข่าย critic 2
    target_critic1 (torch.nn.Module): เครือข่าย target critic 1
    target_critic2 (torch.nn.Module): เครือข่าย target critic 2
    actor_network (torch.nn.Module): เครือข่าย actor
    states (torch.Tensor): สถานะปัจจุบัน
    actions (torch.Tensor): การกระทำ
    rewards (torch.Tensor): รางวัล
    next_states (torch.Tensor): สถานะถัดไป
    dones (torch.Tensor): สถานะจบ
    gamma (float): discount factor
    alpha (float): พารามิเตอร์ temperature
    
    Returns:
    Tuple[torch.Tensor, torch.Tensor]: ค่าการสูญเสียของ critic 1 และ critic 2
    """
    with torch.no_grad():
        # ได้การกระทำถัดไปและ log probability จาก actor
        next_actions, next_log_probs = actor_network.sample(next_states)
        
        # คำนวณค่า Q ถัดไปจาก target critic
        next_q1 = target_critic1(next_states, next_actions)
        next_q2 = target_critic2(next_states, next_actions)
        
        # ใช้ค่า Q ที่น้อยกว่า
        next_q = torch.min(next_q1, next_q2)
        
        # คำนวณ target Q (รวม entropy)
        target_q = rewards + gamma * (1 - dones) * (next_q - alpha * next_log_probs)
    
    # คำนวณค่า Q ปัจจุบัน
    current_q1 = critic1(states, actions)
    current_q2 = critic2(states, actions)
    
    # คำนวณการสูญเสีย
    critic1_loss = F.mse_loss(current_q1, target_q)
    critic2_loss = F.mse_loss(current_q2, target_q)
    
    return critic1_loss, critic2_loss

def sac_alpha_loss(log_alpha: torch.Tensor, actor_network: torch.nn.Module, 
                 states: torch.Tensor, target_entropy: float) -> torch.Tensor:
    """
    ฟังก์ชันการสูญเสียสำหรับพารามิเตอร์ alpha ใน SAC (Soft Actor-Critic) ที่ปรับตัวได้
    
    Parameters:
    log_alpha (torch.Tensor): ค่า log ของพารามิเตอร์ alpha
    actor_network (torch.nn.Module): เครือข่าย actor
    states (torch.Tensor): สถานะ
    target_entropy (float): ค่า entropy เป้าหมาย
    
    Returns:
    torch.Tensor: ค่าการสูญเสีย
    """
    # ได้ log probability จาก actor
    _, log_probs = actor_network.sample(states)
    
    # คำนวณการสูญเสีย
    alpha = log_alpha.exp()
    alpha_loss = -(log_alpha * (log_probs + target_entropy).detach()).mean()
    
    return alpha_loss

def ppo_actor_critic_loss(ratio: torch.Tensor, advantage: torch.Tensor, 
                        value_pred: torch.Tensor, value_target: torch.Tensor,
                        entropy: torch.Tensor, clip_param: float = 0.2,
                        value_coef: float = 0.5, entropy_coef: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ฟังก์ชันการสูญเสียสำหรับ PPO (Proximal Policy Optimization) ที่รวม actor และ critic
    
    Parameters:
    ratio (torch.Tensor): อัตราส่วนของความน่าจะเป็นการกระทำใหม่ต่อความน่าจะเป็นการกระทำเก่า
    advantage (torch.Tensor): ค่า advantage
    value_pred (torch.Tensor): ค่า value ที่ทำนาย
    value_target (torch.Tensor): ค่า value เป้าหมาย
    entropy (torch.Tensor): ค่า entropy
    clip_param (float): พารามิเตอร์สำหรับการ clip
    value_coef (float): สัมประสิทธิ์สำหรับค่า value loss
    entropy_coef (float): สัมประสิทธิ์สำหรับค่า entropy
    
    Returns:
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ค่าการสูญเสียรวม, actor loss, critic loss, entropy loss
    """
    # คำนวณ policy loss
    policy_loss1 = ratio * advantage
    policy_loss2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
    policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
    
    # คำนวณ value loss
    value_pred_clipped = value_pred + (value_target - value_pred).clamp(-clip_param, clip_param)
    value_loss1 = F.mse_loss(value_pred, value_target)
    value_loss2 = F.mse_loss(value_pred_clipped, value_target)
    value_loss = torch.max(value_loss1, value_loss2)
    
    # คำนวณ entropy loss
    entropy_loss = -entropy.mean()
    
    # คำนวณการสูญเสียรวม
    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    
    return total_loss, policy_loss, value_loss, entropy_loss

def a2c_loss(log_probs: torch.Tensor, values: torch.Tensor, 
           returns: torch.Tensor, entropy: torch.Tensor,
           value_coef: float = 0.5, entropy_coef: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ฟังก์ชันการสูญเสียสำหรับ A2C (Advantage Actor-Critic)
    
    Parameters:
    log_probs (torch.Tensor): log ของความน่าจะเป็นของการกระทำ
    values (torch.Tensor): ค่า value ที่ทำนาย
    returns (torch.Tensor): ผลตอบแทนสะสม
    entropy (torch.Tensor): ค่า entropy
    value_coef (float): สัมประสิทธิ์สำหรับค่า value loss
    entropy_coef (float): สัมประสิทธิ์สำหรับค่า entropy
    
    Returns:
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ค่าการสูญเสียรวม, actor loss, critic loss, entropy loss
    """
    # คำนวณ advantage
    advantage = returns - values
    
    # คำนวณ policy loss
    policy_loss = -(log_probs * advantage.detach()).mean()
    
    # คำนวณ value loss
    value_loss = F.mse_loss(values, returns)
    
    # คำนวณ entropy loss
    entropy_loss = -entropy.mean()
    
    # คำนวณการสูญเสียรวม
    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    
    return total_loss, policy_loss, value_loss, entropy_loss

def rainbow_dqn_loss(current_q: torch.Tensor, target_q: torch.Tensor, 
                   weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ฟังก์ชันการสูญเสียสำหรับ Rainbow DQN
    
    Parameters:
    current_q (torch.Tensor): ค่า Q ปัจจุบัน
    target_q (torch.Tensor): ค่า Q เป้าหมาย
    weights (torch.Tensor, optional): น้ำหนักสำหรับ Prioritized Experience Replay
    
    Returns:
    Tuple[torch.Tensor, torch.Tensor]: ค่าการสูญเสีย, ค่า TD error (สำหรับ PER)
    """
    # คำนวณค่า TD error
    td_error = (current_q - target_q).abs()
    
    # คำนวณการสูญเสียแบบ Huber
    loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
    
    # ถ่วงน้ำหนักการสูญเสีย (สำหรับ PER)
    if weights is not None:
        loss = loss * weights
    
    # คำนวณค่าเฉลี่ย
    loss = loss.mean()
    
    return loss, td_error

def impala_loss(log_probs: torch.Tensor, old_log_probs: torch.Tensor, 
              values: torch.Tensor, old_values: torch.Tensor,
              actions: torch.Tensor, rewards: torch.Tensor, 
              gamma: float = 0.99, lambda_gae: float = 0.95,
              rho_bar: float = 1.0, c_bar: float = 1.0) -> torch.Tensor:
    """
    ฟังก์ชันการสูญเสียสำหรับ IMPALA (Importance Weighted Actor-Learner Architecture)
    
    Parameters:
    log_probs (torch.Tensor): log ของความน่าจะเป็นของการกระทำใหม่
    old_log_probs (torch.Tensor): log ของความน่าจะเป็นของการกระทำเก่า
    values (torch.Tensor): ค่า value ใหม่
    old_values (torch.Tensor): ค่า value เก่า
    actions (torch.Tensor): การกระทำ
    rewards (torch.Tensor): รางวัล
    gamma (float): discount factor
    lambda_gae (float): พารามิเตอร์ lambda สำหรับ GAE
    rho_bar (float): พารามิเตอร์สำหรับการ clip importance weights
    c_bar (float): พารามิเตอร์สำหรับการ clip correction weights
    
    Returns:
    torch.Tensor: ค่าการสูญเสีย
    """
    # คำนวณ importance weights
    rho = torch.exp(log_probs - old_log_probs)
    clipped_rho = torch.clamp(rho, 0, rho_bar)
    
    # คำนวณ correction weights
    c = torch.clamp(rho, 0, c_bar)
    
    # คำนวณ advantages และ returns ด้วย V-trace
    with torch.no_grad():
        # คำนวณ TD errors
        td_errors = rewards + gamma * values[1:] - values[:-1]
        
        # คำนวณ advantages
        advantages = torch.zeros_like(td_errors)
        last_advantage = 0
        
        for t in reversed(range(len(td_errors))):
            advantages[t] = td_errors[t] + gamma * lambda_gae * clipped_rho[t] * last_advantage
            last_advantage = advantages[t]
        
        # คำนวณ returns
        returns = advantages + values[:-1]
    
    # คำนวณ policy loss
    policy_loss = -(clipped_rho * log_probs * advantages.detach()).mean()
    
    # คำนวณ value loss
    value_loss = 0.5 * F.mse_loss(values[:-1], returns)
    
    # คำนวณการสูญเสียรวม
    loss = policy_loss + value_loss
    
    return loss