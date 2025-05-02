#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reinforcement Learning training module for language models.
"""

import os
import torch
import torch.nn as nn
import logging
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RLConfig:
    """Configuration class for reinforcement learning parameters."""
    
    def __init__(
        self,
        learning_rate=1e-5,
        batch_size=8,
        max_seq_length=512,
        num_episodes=1000,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        seed=42,
        **kwargs
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        
        for key, value in kwargs.items():
            setattr(self, key, value)

class RLTrainer:
    """Main trainer class for reinforcement learning."""
    
    def __init__(self, config, model, tokenizer=None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = self._get_optimizer()
        logger.info(f"Using device: {self.device}")

    def _get_optimizer(self):
        """Initialize the optimizer."""
        return torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """Compute discounted returns."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.config.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, device=self.device)

    def train(self, env):
        """Main training loop for reinforcement learning."""
        logger.info("Starting reinforcement learning training...")
        
        epsilon = self.config.epsilon_start
        best_reward = float('-inf')
        
        for episode in range(self.config.num_episodes):
            state = env.reset()
            episode_rewards = []
            episode_actions = []
            episode_states = []
            
            done = False
            while not done:
                # Convert state to model input format
                state_input = self.tokenizer(
                    state,
                    max_length=self.config.max_seq_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action_probs = self.model(**state_input).logits
                        action = torch.argmax(action_probs).item()
                
                # Take action in environment
                next_state, reward, done, _ = env.step(action)
                
                # Store transition
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                
                state = next_state
            
            # Compute returns and update policy
            returns = self.compute_returns(episode_rewards)
            loss = self._update_policy(episode_states, episode_actions, returns)
            
            # Decay epsilon
            epsilon = max(self.config.epsilon_end, epsilon * self.config.epsilon_decay)
            
            # Log progress
            total_reward = sum(episode_rewards)
            if total_reward > best_reward:
                best_reward = total_reward
                self._save_checkpoint("best")
            
            if episode % 10 == 0:
                logger.info(
                    f"Episode {episode}/{self.config.num_episodes} | "
                    f"Total Reward: {total_reward:.2f} | "
                    f"Epsilon: {epsilon:.2f} | "
                    f"Loss: {loss:.4f}"
                )
        
        self._save_checkpoint("final")
        return {"best_reward": best_reward}

    def _update_policy(self, states, actions, returns):
        """Update policy using the collected experience."""
        loss = 0
        self.optimizer.zero_grad()
        
        for state, action, R in zip(states, actions, returns):
            state_input = self.tokenizer(
                state,
                max_length=self.config.max_seq_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**state_input)
            action_probs = outputs.logits
            
            # Compute policy loss
            policy_loss = -torch.log(action_probs[0, action]) * R
            
            # Add entropy bonus for exploration
            entropy = -torch.sum(action_probs * torch.log(action_probs))
            loss += policy_loss - self.config.entropy_coef * entropy
        
        loss = loss / len(states)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        return loss.item()

    def _save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        output_dir = f"checkpoints/rl_{checkpoint_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving model checkpoint to {output_dir}")
        
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)
            if self.tokenizer:
                self.tokenizer.save_pretrained(output_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(output_dir, "model.bin"))
        
        # Save training config
        torch.save(self.config.__dict__, os.path.join(output_dir, "rl_config.bin"))
