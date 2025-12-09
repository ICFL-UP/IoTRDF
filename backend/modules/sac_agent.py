"""
sac_agent.py

Implements the FeatureAwareSAC agent with a hierarchical actor–critic
architecture for feature selection in IRDF. The agent uses separate
entropy temperatures for operation and feature selection, target critics,
and supports prioritized replay and checkpointing.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import logging
from typing import Dict, Tuple, Set, List, Optional
import os
import random
from enum import IntEnum

logger = logging.getLogger(__name__)

class Operation(IntEnum):
   
    ADD = 0
    REMOVE = 1
    NO_OP = 2

class HierarchicalActor(nn.Module):
 
    def __init__(self, state_dim: int, feature_dim: int, hidden_dim: int = 512, op_embed_dim: int = 32):
        
        super().__init__()
        self.num_operations = 3 

        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        self.operation_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_operations)
        )

        self.op_embedding = nn.Embedding(self.num_operations, op_embed_dim)

        self.feature_head = nn.Sequential(
      
            nn.Linear((hidden_dim // 2) + op_embed_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, feature_dim)
        )

    def forward(self, state: torch.Tensor):
       
        shared_embedding = self.shared_net(state)
        operation_logits = self.operation_head(shared_embedding)
        operation_dist = Categorical(logits=operation_logits)

        return operation_dist, shared_embedding

    def get_feature_dist(self, shared_embedding: torch.Tensor, op_idx: torch.Tensor) -> Categorical:
       
        op_embedded = self.op_embedding(op_idx)

        conditional_input = torch.cat([shared_embedding, op_embedded], dim=-1)

        feature_logits = self.feature_head(conditional_input)
        feature_dist = Categorical(logits=feature_logits)

        return feature_dist

class CriticNetwork(nn.Module):
   
    def __init__(self, state_dim: int, feature_dim: int, hidden_dim: int = 512, device: torch.device = None):
        super().__init__()
        if device is None:
            raise ValueError("CriticNetwork requires a device.")
        self.device = device
        self.feature_dim = feature_dim
        self.num_operations = len(Operation)

        self.input_norm = nn.LayerNorm(state_dim)
        
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.add_head = nn.Linear(hidden_dim, feature_dim)
        self.remove_head = nn.Linear(hidden_dim, feature_dim)
        self.noop_head = nn.Linear(hidden_dim, feature_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        
        normalized_state = self.input_norm(state)
        
        logger.debug(
            f"[Critic Input Stats] "
            f"Shape: {normalized_state.shape} | "
            f"Mean: {normalized_state.mean().item():.3f} | "
            f"Std: {normalized_state.std().item():.3f}"
        )
        
        state_embedding = self.shared_net(normalized_state)
        
        q_add_values = self.add_head(state_embedding)
        q_remove_values = self.remove_head(state_embedding)
        q_noop_values = self.noop_head(state_embedding)
        
        return torch.stack([q_add_values, q_remove_values, q_noop_values], dim=2)

class FeatureAwareSAC(nn.Module):
   
    def __init__(self,
                 state_dim: int,
                 feature_dim: int,
                 hidden_dim: int = 256,
                 op_embed_dim: int = 32,
                 gamma: float = 0.99,
                 tau: float = 0.001,
                 lr: float = 1e-5,
                 actor_lr: float = 3e-5,
                 alpha_lr: float = 1e-3,
                 target_entropy_op_ratio: float = 0.9,
                 target_entropy_feat_ratio: float = 0.5,
                 manual_alpha_control: bool = False):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim, self.feature_dim, self.num_operations = state_dim, feature_dim, len(Operation)
        self.gamma, self.tau = gamma, tau
        self.manual_alpha_control = manual_alpha_control

        self.actor = HierarchicalActor(state_dim, feature_dim, hidden_dim, op_embed_dim).to(self.device)
        self.critic1 = CriticNetwork(state_dim, feature_dim, hidden_dim, device=self.device).to(self.device)
        self.critic2 = CriticNetwork(state_dim, feature_dim, hidden_dim, device=self.device).to(self.device)
        self.critic1_target = CriticNetwork(state_dim, feature_dim, hidden_dim, device=self.device).to(self.device)
        self.critic2_target = CriticNetwork(state_dim, feature_dim, hidden_dim, device=self.device).to(self.device)
        self._hard_update(self.critic1_target, self.critic1)
        self._hard_update(self.critic2_target, self.critic2)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=lr)

        self.log_alpha_op = nn.Parameter(torch.full((1,), -1.0, device=self.device, requires_grad=True))
        self.log_alpha_feat = nn.Parameter(torch.full((1,), -1.0, device=self.device, requires_grad=True))
        
        if not self.manual_alpha_control:
 
            max_op_entropy = np.log(self.num_operations)
            self.target_entropy_op = target_entropy_op_ratio * max_op_entropy
            self.alpha_op_optim = optim.Adam([self.log_alpha_op], lr=alpha_lr)
            logger.info(f"Initialized separate α_op with target entropy: {self.target_entropy_op:.4f}")

            max_feat_entropy = np.log(self.feature_dim)
            self.target_entropy_feat = target_entropy_feat_ratio * max_feat_entropy
            self.alpha_feat_optim = optim.Adam([self.log_alpha_feat], lr=alpha_lr)
            logger.info(f"Initialized separate α_feat with target entropy: {self.target_entropy_feat:.4f}")
        else:
            self.target_entropy_op, self.target_entropy_feat = None, None
            self.alpha_op_optim, self.alpha_feat_optim = None, None
            logger.info("Using manual alpha control. Target entropy tuning is disabled.")
             
        self.current_features: Set[int] = set()

    @property
    def alpha_op(self) -> torch.Tensor:
        if self.manual_alpha_control:

            return torch.tensor(getattr(self, "current_alpha_value", 1.0), device=self.device)
        else:
            return self.log_alpha_op.exp()

    @property
    def alpha_feat(self) -> torch.Tensor:
        if self.manual_alpha_control:
            return torch.tensor(getattr(self, "current_alpha_value", 1.0), device=self.device)
        else:
            return self.log_alpha_feat.exp()

    def select_action(self, state: torch.Tensor, eval_mode: bool = False) -> Tuple[int, int]:
       
        self.actor.eval()

        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(self.device)

        with torch.no_grad():
        
            op_dist, shared_embedding = self.actor(state)
            op_logits = op_dist.logits.clone()  

            current_num_features = len(self.current_features)
            if current_num_features <= 1 and self.feature_dim > 1:
                
                op_logits[:, Operation.REMOVE.value] = -float('inf')
            if current_num_features == self.feature_dim:
              
                op_logits[:, Operation.ADD.value] = -float('inf')

            final_op_dist = Categorical(logits=op_logits)
            logger.info(
                f"[Action Select] Op probs: "
                f"ADD={final_op_dist.probs[0, Operation.ADD.value]:.3f}, "
                f"REMOVE={final_op_dist.probs[0, Operation.REMOVE.value]:.3f}, "
                f"NO_OP={final_op_dist.probs[0, Operation.NO_OP.value]:.3f}"
            )

            if eval_mode:
                op_idx = torch.argmax(final_op_dist.probs, dim=1)
            else:
                op_idx = final_op_dist.sample()

            feat_dist = self.actor.get_feature_dist(shared_embedding, op_idx)
            feat_logits = feat_dist.logits.clone()  

            chosen_op = Operation(op_idx.item())
            if chosen_op == Operation.ADD:
        
                if self.current_features:
                    feat_logits[:, list(self.current_features)] = -float('inf')
            elif chosen_op in (Operation.REMOVE, Operation.NO_OP):

                inactive_features = list(set(range(self.feature_dim)) - self.current_features)
                if inactive_features:
                    feat_logits[:, inactive_features] = -float('inf')

            if torch.all(feat_logits == -float('inf')):
                logger.warning(f"All features for operation '{chosen_op.name}' are masked. Forcing NO_OP.")
                op_idx = torch.tensor([Operation.NO_OP.value], device=self.device, dtype=torch.long)

                if self.current_features:
                    feat_idx = torch.tensor([random.choice(list(self.current_features))], device=self.device, dtype=torch.long)
                else:
                    feat_idx = torch.tensor([0], device=self.device, dtype=torch.long)
            else:
                final_feat_dist = Categorical(logits=feat_logits)
                if eval_mode:
                    feat_idx = torch.argmax(final_feat_dist.probs, dim=1)
                else:
                    feat_idx = final_feat_dist.sample()

            final_op = Operation(op_idx.item())

        self.actor.train()
        logger.info(f"--> Final selected action: {final_op.name} feature {feat_idx.item()}")

        return int(feat_idx.item()), int(op_idx.item())

    def update(self, batch: Dict[str, torch.Tensor], is_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        states, op_indices, feature_indices, rewards, next_states, dones = (
            batch['states'], batch['op_idx'], batch['feature_idx'],
            batch['rewards'], batch['next_states'], batch['dones']
        )
        if is_weights is None:
            is_weights = torch.ones_like(rewards, device=self.device)

        batch_size = states.shape[0]

        with torch.no_grad():
          
            next_op_dist, next_shared_embedding = self.actor(next_states)
            next_op_probs = next_op_dist.probs  

            q1_target_all = self.critic1_target(next_states)
            q2_target_all = self.critic2_target(next_states)
            q_target_min = torch.min(q1_target_all, q2_target_all)  

            expected_q_per_op = []
            expected_feat_entropy_per_op = []
            for op_val in range(self.num_operations):
                op_tensor = torch.full((batch_size,), op_val, device=self.device, dtype=torch.long)
             
                next_feat_dist = self.actor.get_feature_dist(next_shared_embedding, op_tensor)
                q_for_op = q_target_min[:, :, op_val]  
    
                expected_q = torch.sum(next_feat_dist.probs * q_for_op, dim=1)  
                expected_q_per_op.append(expected_q)
                expected_feat_entropy_per_op.append(next_feat_dist.entropy())   

            expected_q_per_op = torch.stack(expected_q_per_op, dim=1)                      
            expected_feat_entropy_per_op = torch.stack(expected_feat_entropy_per_op, dim=1) 

            expected_q_next = torch.sum(next_op_probs * expected_q_per_op, dim=1).unsqueeze(1)          
            expected_feat_entropy = torch.sum(next_op_probs * expected_feat_entropy_per_op, dim=1).unsqueeze(1)  
            op_entropy = next_op_dist.entropy().unsqueeze(1)                                          

            next_value = expected_q_next + (self.alpha_op * op_entropy) + (self.alpha_feat * expected_feat_entropy)

            q_target = rewards + self.gamma * (1 - dones.float()) * next_value

        op_indices = op_indices.long()
        feature_indices = feature_indices.long()

        q1_all, q2_all = self.critic1(states), self.critic2(states)  

        op_idx_exp = op_indices.view(-1, 1, 1).expand(-1, self.feature_dim, 1)     
        feat_idx_exp = feature_indices.view(-1, 1, 1)                                

        q1 = q1_all.gather(2, op_idx_exp).gather(1, feat_idx_exp).squeeze(2).squeeze(1).view(-1, 1)  
        q2 = q2_all.gather(2, op_idx_exp).gather(1, feat_idx_exp).squeeze(2).squeeze(1).view(-1, 1)  

        logger.debug(f"[Critic Q-values] Q1 mean: {q1.mean().item():.4f}, Q2 mean: {q2.mean().item():.4f}")

        critic1_loss = (is_weights * F.mse_loss(q1, q_target, reduction='none')).mean()
        critic2_loss = (is_weights * F.mse_loss(q2, q_target, reduction='none')).mean()

        self.critic1_optim.zero_grad(); critic1_loss.backward(); self.critic1_optim.step()
        self.critic2_optim.zero_grad(); critic2_loss.backward(); self.critic2_optim.step()

        op_dist, shared_embedding = self.actor(states)
        op_probs = op_dist.probs 

        with torch.no_grad():
            q_min_all_current = torch.min(self.critic1(states), self.critic2(states))  

        expected_q_per_op_current, expected_feat_entropy_per_op_current = [], []
        for op_val in range(self.num_operations):
            op_tensor = torch.full((batch_size,), op_val, device=self.device, dtype=torch.long)
            feat_dist = self.actor.get_feature_dist(shared_embedding, op_tensor)       
            q_for_op = q_min_all_current[:, :, op_val]                                 
            expected_q_current = torch.sum(feat_dist.probs * q_for_op, dim=1)          
            expected_q_per_op_current.append(expected_q_current)
            expected_feat_entropy_per_op_current.append(feat_dist.entropy())           

        expected_q_per_op_current = torch.stack(expected_q_per_op_current, dim=1)                       
        expected_feat_entropy_per_op_current = torch.stack(expected_feat_entropy_per_op_current, dim=1) 

        actor_objective_q = torch.sum(op_probs * expected_q_per_op_current, dim=1)                                  
        actor_objective_entropy_op = op_dist.entropy()                                                               
        actor_objective_entropy_feat = torch.sum(op_probs * expected_feat_entropy_per_op_current, dim=1)             
       
        actor_loss = (- self.alpha_op.detach() * actor_objective_entropy_op
                    - self.alpha_feat.detach() * actor_objective_entropy_feat
                    - actor_objective_q).mean()

        self.actor_optim.zero_grad(); actor_loss.backward(); self.actor_optim.step()

        alpha_loss_op, alpha_loss_feat = torch.tensor(0.0), torch.tensor(0.0)
        if not self.manual_alpha_control:
            alpha_loss_op = -(self.log_alpha_op * (actor_objective_entropy_op.detach() - self.target_entropy_op)).mean()
            self.alpha_op_optim.zero_grad(); alpha_loss_op.backward(); self.alpha_op_optim.step()

            alpha_loss_feat = -(self.log_alpha_feat * (actor_objective_entropy_feat.detach() - self.target_entropy_feat)).mean()
            self.alpha_feat_optim.zero_grad(); alpha_loss_feat.backward(); self.alpha_feat_optim.step()

        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)

        op_entropy_mean = actor_objective_entropy_op.mean().item()
        feat_entropy_mean = actor_objective_entropy_feat.mean().item()

        logger.info(
            f"[SAC Update] ActorLoss={actor_loss.item():.4f} | "
            f"CriticLoss=({critic1_loss.item():.4f}, {critic2_loss.item():.4f}) | "
            f"Alphas=({self.alpha_op.item():.3f}, {self.alpha_feat.item():.3f})"
        )
        if not self.manual_alpha_control:
            logger.info(
                f"[Entropy] Op={op_entropy_mean:.3f} (T={self.target_entropy_op:.3f}) | "
                f"Feat={feat_entropy_mean:.3f} (T={self.target_entropy_feat:.3f}) | "
                f"AlphaLoss=({alpha_loss_op.item():.4f}, {alpha_loss_feat.item():.4f})"
            )

        metrics = {
            'actor_loss': actor_loss.item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'alpha_op': self.alpha_op.item(),
            'alpha_feat': self.alpha_feat.item(),
            'op_entropy': op_entropy_mean,
            'feat_entropy': feat_entropy_mean,
            'alpha_loss_op': alpha_loss_op.item(),
            'alpha_loss_feat': alpha_loss_feat.item(),
        }

        new_priorities = torch.abs(q1 - q_target).detach().squeeze(-1)
        new_priorities = torch.clamp(new_priorities, min=1e-6)

        logger.debug(f"[TD Errors] Mean={new_priorities.mean().item():.4f} | Max={new_priorities.max().item():.4f}")

        return new_priorities, metrics
        
    def apply_action(self, feature_idx: int, operation_value: int):
        """Modifies the agent's internal feature set based on the chosen action."""
        op = Operation(operation_value)
        if op == Operation.ADD: self.current_features.add(feature_idx)
        elif op == Operation.REMOVE: self.current_features.discard(feature_idx)

    def initialize_features(self, k: int):
        """Initializes the agent with a random subset of k features."""
        logger.info(f"Initializing feature set to a random subset of size {k}...")
        if not (0 < k <= self.feature_dim):
            k = min(max(1, k), self.feature_dim)
            logger.warning(f"Invalid k provided. Adjusted to {k}.")
        all_feature_indices = list(range(self.feature_dim))
        self.current_features = set(random.sample(all_feature_indices, k))
        logger.info(f"Agent features initialized to: {sorted(list(self.current_features))}")

    def get_feature_action_probs(self, state: torch.Tensor, operation: Operation, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Gets the top_k feature probabilities for a given operation, compatible with HierarchicalActor.
        """
        self.actor.eval()
        with torch.no_grad():
           
            _, shared_embedding = self.actor(state)
            
            op_idx = torch.tensor([operation.value], device=self.device, dtype=torch.long)

            feat_dist = self.actor.get_feature_dist(shared_embedding, op_idx)
            
            feat_probs = feat_dist.probs.squeeze()
            probs = feat_probs.cpu().numpy()
            feature_probs = [(idx, float(prob)) for idx, prob in enumerate(probs)]
            top_features = sorted(feature_probs, key=lambda x: x[1], reverse=True)[:top_k]
        
        self.actor.train()
        return top_features
         
    def get_operation_probabilities(self, state: torch.Tensor, current_selected_features: Optional[Set[int]] = None) -> Dict[str, float]:
        """
        Returns the probabilities for each operation, compatible with HierarchicalActor.
        """
        self.actor.eval()
        with torch.no_grad():
            features_to_use = current_selected_features if current_selected_features is not None else self.current_features
            
            op_dist, _ = self.actor(state)
            op_logits = op_dist.logits.clone()

            current_num_features = len(features_to_use)
            if current_num_features <= 1:
                op_logits[:, Operation.REMOVE.value] = -float('inf')
            if current_num_features == self.feature_dim:
                op_logits[:, Operation.ADD.value] = -float('inf')

            op_probs = F.softmax(op_logits, dim=1).squeeze(0)
            
        self.actor.train()
        return {op.name: op_probs[op.value].item() for op in Operation}

        
    def get_topk_features_for_operation(self, state: torch.Tensor, operation: Operation, k: int = 5, current_selected_features: Optional[Set[int]] = None) -> List[Tuple[int, float]]:
        self.critic1.eval()
        self.critic2.eval()
        with torch.no_grad():
            if state.dim() == 1: state = state.unsqueeze(0)
                
            q1_all = self.critic1(state).squeeze(0)
            q2_all = self.critic2(state).squeeze(0)
                
            q_min = torch.min(q1_all, q2_all)
            op_q_values = q_min[:, operation.value]
            
            
            num_features = op_q_values.shape[0]
            all_features = set(range(num_features))
            if current_selected_features is None: valid_features = all_features
            elif operation == Operation.REMOVE: valid_features = set(current_selected_features)
            elif operation == Operation.ADD: valid_features = all_features - set(current_selected_features)
            else: valid_features = set(current_selected_features)
            if not valid_features: return []
            valid_q_vals = [(idx, op_q_values[idx].item()) for idx in valid_features]
            valid_q_vals.sort(key=lambda x: x[1], reverse=True)
        self.critic1.train()
        self.critic2.train()
        return valid_q_vals[:k]

    def get_mean_q_values(self, state: torch.Tensor) -> Dict[str, float]:
        self.critic1.eval()
        self.critic2.eval()
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            q1_all = self.critic1(state)
            q2_all = self.critic2(state)
            
            q_min_all = torch.min(q1_all, q2_all).squeeze(0)
            mean_q_values = {}
            for op in Operation:
                mean_q = q_min_all[:, op.value].mean().item()
                mean_q_values[f"mean_q_{op.name.lower()}"] = mean_q
        self.critic1.train()
        self.critic2.train()
        return mean_q_values

    def _hard_update(self, target: nn.Module, source: nn.Module):
        """Performs a hard update, copying weights directly."""
        target.load_state_dict(source.state_dict())

    def _soft_update(self, target: nn.Module, source: nn.Module):
        """Performs a soft update of the target network weights."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path: str):
        """Saves the SAC agent state, including separate alphas."""
        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, "sac_agent.pth")
        
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'actor_optim_state_dict': self.actor_optim.state_dict(),
            'critic1_optim_state_dict': self.critic1_optim.state_dict(),
            'critic2_optim_state_dict': self.critic2_optim.state_dict(),
            'log_alpha_op': self.log_alpha_op.data,
            'log_alpha_feat': self.log_alpha_feat.data,
        }
        if not self.manual_alpha_control:
            save_dict['alpha_op_optim_state_dict'] = self.alpha_op_optim.state_dict()
            save_dict['alpha_feat_optim_state_dict'] = self.alpha_feat_optim.state_dict()
            
        torch.save(save_dict, checkpoint_path)
        logger.info(f"Saved agent state with separate alphas to {checkpoint_path}")

    def load(self, path: str):
        """Loads the SAC agent state, including separate alphas."""
        checkpoint_path = os.path.join(path, "sac_agent.pth")
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found at {checkpoint_path}. Starting fresh.")
            return
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic1_optim.load_state_dict(checkpoint['critic1_optim_state_dict'])
        self.critic2_optim.load_state_dict(checkpoint['critic2_optim_state_dict'])

        if 'log_alpha_op' in checkpoint and 'log_alpha_feat' in checkpoint:
            self.log_alpha_op.data.copy_(checkpoint['log_alpha_op'])
            self.log_alpha_feat.data.copy_(checkpoint['log_alpha_feat'])
        elif 'log_alpha' in checkpoint: 
            logger.warning("Loading from an old checkpoint with a single alpha. Initializing both alphas to the old value.")
            self.log_alpha_op.data.copy_(checkpoint['log_alpha'])
            self.log_alpha_feat.data.copy_(checkpoint['log_alpha'])

        if not self.manual_alpha_control:
            if 'alpha_op_optim_state_dict' in checkpoint:
                self.alpha_op_optim.load_state_dict(checkpoint['alpha_op_optim_state_dict'])
            if 'alpha_feat_optim_state_dict' in checkpoint:
                self.alpha_feat_optim.load_state_dict(checkpoint['alpha_feat_optim_state_dict'])
                
        self._hard_update(self.critic1_target, self.critic1)
        self._hard_update(self.critic2_target, self.critic2)
        logger.info(f"Loaded agent state with separate alphas from {checkpoint_path}")
