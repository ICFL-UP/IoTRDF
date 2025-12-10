"""
Orchestrates SAC-based feature-selection training.

The PolicyLearner coordinates the FeatureAwareSAC agent, reward function,
classifier evaluation, replay buffer, and SSE logging for per-epoch diagnostics.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import logging
from typing import Dict, Tuple, Generator, Optional, List, Set, Any
import json
import random
import os
import uuid

from .sac_agent import FeatureAwareSAC, Operation
from .reward_function import RewardFunction
from .sac_reply_buffer import PrioritizedReplayBuffer
from .classifier_evaluator import ClassifierEvaluator, ClassifierSpec

logger = logging.getLogger(__name__)

class PolicyLearner:
   
    def __init__(self, max_features: int, config: Dict = None, save_path: str = None):
        self.save_path = save_path
        self.max_features = max_features
        
        self.config = {
    
            'warmup_steps': 64,
            'sac_batch_size': 32,
            'sac_buffer_size': 10000,
            'per_alpha': 0.6,
            'per_beta': 0.4,
            'use_bsf_guider': True,
            'stuck_threshold_ratio': 0.05,
            'bloated_feature_threshold_for_bsf': 0.3,
            'optimal_f1_threshold': 0.99,
            'sac_hidden_dim': 512,
            'op_embed_dim': 32,
            'learning_rate': 3e-5,
            'actor_learning_rate': 3e-5,
            'alpha_lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.001,
            'manual_alpha_control': False,
            'initial_alpha': 0.3,
            'alpha_decay_rate': 0.999,
            'min_alpha': 0.05,
            'target_entropy_op_ratio': 0.2,
            'target_entropy_feat_ratio': 0.2,
            'classifier_spec': {'name': 'rf', 'params': {}},
            # Add any other hyperparameters here
            'max_epochs': 200,
        }
        
        self.config.update(config if config is not None else {})

        logger.info("-" * 60)
        logger.info(" PolicyLearner initialized with the following configuration:")
        for key, value in self.config.items():
            logger.info(f"  - {key}: {value}")
        logger.info("-" * 60)

        self.classifier_spec = ClassifierSpec(**self.config.get('classifier_spec'))

        self.steps_done = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.config.get('manual_alpha_control', False):
            self.current_alpha_value = self.config['initial_alpha']
        
        calculated_sac_state_dim = 3

        self.agent = FeatureAwareSAC(
            state_dim=calculated_sac_state_dim,
            feature_dim=self.max_features,
            hidden_dim=self.config['sac_hidden_dim'],
            op_embed_dim=self.config['op_embed_dim'],
            gamma=self.config['gamma'],
            tau=self.config['tau'],
            lr=self.config['learning_rate'],
            actor_lr=self.config['actor_learning_rate'],
            alpha_lr=self.config['alpha_lr'],
            target_entropy_op_ratio=self.config['target_entropy_op_ratio'],
            target_entropy_feat_ratio=self.config['target_entropy_feat_ratio'],
            manual_alpha_control=self.config['manual_alpha_control']
        ).to(self.device)

        if self.config.get('manual_alpha_control', False):
            self.agent.current_alpha_value = self.current_alpha_value

        self.reward_fn = RewardFunction()
        self.sac_replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config['sac_buffer_size'],
            alpha=self.config['per_alpha']
        )
        self.sac_replay_buffer.clear()
        
        self.classifier = ClassifierEvaluator(
            state_dim=self.max_features,
            classifier_spec=self.classifier_spec
        )
        
        initial_k = 3
        self.agent.initialize_features(k=initial_k)
        initial_metrics = {
            'accuracy': 0.0,
            'f1': 0.0,
            'num_features': len(self.agent.current_features)
        }
        self.best_f1_so_far = initial_metrics['f1']
        self.best_features_so_far = self.agent.current_features.copy()
        self.buffer_id = str(uuid.uuid4())[:8]
        self.current_sac_state_tensor = self._get_sac_state(initial_metrics, self.agent.current_features)

        logger.info(f"PolicyLearner initialized with Replay Buffer ID: {self.buffer_id}")
        logger.info(
            f"PolicyLearner initialized with MANUAL alpha control (Initial α: {self.config['initial_alpha']})."
        )

    def _get_sac_state(self, classifier_metrics: Dict[str, float], current_features: Set[int]) -> torch.Tensor:
        
        accuracy = classifier_metrics.get('accuracy', 0.0)
        f1_score = classifier_metrics.get('f1', 0.0)
        num_features_normalized = len(current_features) / self.max_features if self.max_features > 0 else 0.0

        # The state is ONLY the 3 high-level performance metrics.
        # This gives the critic a simple, low-dimensional input to learn from.
        performance_state = torch.tensor([accuracy, f1_score, num_features_normalized], dtype=torch.float32)

        return performance_state.to(self.device)

    def _get_guided_action(self) -> Optional[Tuple[int, int]]:
       
        missing_good_features = self.best_features_so_far - self.agent.current_features
        extra_bad_features = self.agent.current_features - self.best_features_so_far

        if missing_good_features:
            feature_to_add = random.choice(list(missing_good_features))
            logger.warning(f"[BSF Guider]: Agent is missing a good feature. Forcing ADD on feature {feature_to_add}.")
            return feature_to_add, Operation.ADD.value
        
        if extra_bad_features:
            feature_to_remove = random.choice(list(extra_bad_features))
            logger.warning(f"[BSF Guider]: Agent has an extra feature. Forcing REMOVE on feature {feature_to_remove}.")
            return feature_to_remove, Operation.REMOVE.value
            
        return None
    
    def get_diagnostics(self, state_tensor_unsqueezed: torch.Tensor, operation_action_value: int) -> Dict[str, Any]:
    
        with torch.no_grad():
        
            op_dist, shared_embedding = self.agent.actor(state_tensor_unsqueezed)
            
            expected_feat_entropy = 0
            op_probs = op_dist.probs
            for op_val in range(self.agent.num_operations):
                op_tensor = torch.full((1,), op_val, device=self.device, dtype=torch.long)
                feat_dist = self.agent.actor.get_feature_dist(shared_embedding, op_tensor)
                expected_feat_entropy += op_probs[:, op_val] * feat_dist.entropy()

            entropy_op = op_dist.entropy().item()
            entropy_feat = expected_feat_entropy.item()
            
            max_q_potentials = {}
            for op in Operation:
                top_features = self.agent.get_topk_features_for_operation(
                    state_tensor_unsqueezed, op, k=1, current_selected_features=self.agent.current_features
                )
                if top_features:
                    max_q_potentials[op.name] = top_features[0][1]
                else:
                    max_q_potentials[op.name] = float('-inf')

            return {
                'learning_rate': self.config['learning_rate'],
                'alpha_op': self.agent.alpha_op.item(),
                'alpha_feat': self.agent.alpha_feat.item(),
                'entropy_op': entropy_op,
                'entropy_feat': entropy_feat,
                'total_entropy': entropy_op + entropy_feat,
                'target_entropy_op': self.agent.target_entropy_op,
                'target_entropy_feat': self.agent.target_entropy_feat,
                'mean_q_values': self.agent.get_mean_q_values(state_tensor_unsqueezed),
                'operation_probs': self.agent.get_operation_probabilities(state_tensor_unsqueezed, self.agent.current_features),
                'max_q_potentials': max_q_potentials,
                'top_k_q_by_op': {
                    op.name: self.agent.get_topk_features_for_operation(
                        state_tensor_unsqueezed, op, k=5, current_selected_features=self.agent.current_features
                    ) for op in Operation
                },
                'top_feature_probs': self.agent.get_feature_action_probs(
                    state_tensor_unsqueezed, Operation(operation_action_value), top_k=5
                )
            }
        
    def run_training_step(self, real_data_tuple, val_data, epoch: int):
       
        X_train_np, y_train_np = real_data_tuple

        if len(np.unique(y_train_np)) < 2:
            yield self._make_json_serializable({
                'epoch': epoch, 'stage': 'skipped', 'message': 'Training data has only one class'
            })
            return

        self.steps_done += 1

        if self.config.get('manual_alpha_control', False) and self.steps_done > self.config['warmup_steps']:
            prev_alpha = self.current_alpha_value
            self.current_alpha_value = max(
                self.config['min_alpha'],
                self.current_alpha_value * self.config['alpha_decay_rate']
            )
            self.agent.log_alpha_op.data.fill_(np.log(self.current_alpha_value))
            self.agent.log_alpha_feat.data.fill_(np.log(self.current_alpha_value))
            if self.current_alpha_value != prev_alpha:
                logger.info(f"[Manual Alpha Decay] α: {prev_alpha:.4f} → {self.current_alpha_value:.4f}")

        state_tensor_unsqueezed = self.current_sac_state_tensor.unsqueeze(0)
        current_f1 = self._get_sac_state_metrics(self.current_sac_state_tensor)['f1']

        current_num_features = len(self.agent.current_features)
        features_for_experience = self.agent.current_features.copy()

        previous_metrics = {
            'f1': current_f1,
            'num_features': current_num_features,
            'features': features_for_experience  
        }
        state_for_experience = self.current_sac_state_tensor.cpu().numpy()

        if self.steps_done <= self.config['warmup_steps']:
            action_source = "Random Warmup"

            current_num_features = len(self.agent.current_features)
            valid_ops = list(Operation)  # [ADD, REMOVE, NO_OP]
            if current_num_features <= 1 and Operation.REMOVE in valid_ops:
                valid_ops.remove(Operation.REMOVE)
            if current_num_features >= self.max_features and Operation.ADD in valid_ops:
                valid_ops.remove(Operation.ADD)
            if not valid_ops:
                valid_ops = [Operation.NO_OP]

            chosen_op = random.choice(valid_ops)
            operation_action_value = chosen_op.value

            if chosen_op == Operation.ADD:
                possible_features = list(set(range(self.max_features)) - self.agent.current_features)
            else:  # REMOVE or NO_OP
                possible_features = list(self.agent.current_features)

            if not possible_features:  # safety fallback
                operation_action_value = Operation.NO_OP.value
                possible_features = [0]
                logger.warning("Warmup edge case: forcing NO_OP on feature 0.")

            feature_idx_action = random.choice(possible_features)

        else:
         
            if self.steps_done == self.config['warmup_steps'] + 1:
    
                self.best_f1_so_far = current_f1
                self.best_features_so_far = self.agent.current_features.copy()

            is_f1_optimal = current_f1 >= self.config['optimal_f1_threshold']
            is_feature_set_bloated = current_num_features > self.config['bloated_feature_threshold_for_bsf'] * self.max_features

            # Should we guide with Best-So-Far?
            should_activate_bsf_guider = (
                self.config.get('use_bsf_guider', False) and
                (
                    (not is_f1_optimal and current_f1 < (self.best_f1_so_far - self.config['stuck_threshold_ratio'])) or
                    (is_f1_optimal and is_feature_set_bloated and self.best_f1_so_far >= self.config['optimal_f1_threshold'])
                )
            )

            if should_activate_bsf_guider:
                guided_action = self._get_guided_action()
                if guided_action:
                    feature_idx_action, operation_action_value = guided_action
                    action_source = "BSF Guider"
                else:
                    feature_idx_action, operation_action_value = self.agent.select_action(
                        state_tensor_unsqueezed, eval_mode=False
                    )
                    action_source = "Agent (BSF no clear path)"
            else:
                feature_idx_action, operation_action_value = self.agent.select_action(
                    state_tensor_unsqueezed, eval_mode=False
                )
                action_source = "Agent"

        operation_enum = Operation(operation_action_value)

        with torch.no_grad():
            diagnostics = self.get_diagnostics(state_tensor_unsqueezed, operation_action_value)
            yield self._make_json_serializable({
                'stage': 'operation_decision',
                'epoch': epoch,
                'action_source': action_source,
                'chosen_operation': operation_enum.name,
                'chosen_feature': feature_idx_action,
                'diagnostics': diagnostics,
                **diagnostics
            })

        self.agent.apply_action(feature_idx_action, operation_action_value)
        new_feature_set = sorted(list(self.agent.current_features))

        if not new_feature_set:
            initial_k = self.config.get('ultra_lean_feature_count', 3)
            self.agent.initialize_features(k=initial_k)
            new_feature_set = sorted(list(self.agent.current_features))
            logger.warning(f" Feature set empty → Reinitialized to {initial_k} features")

        yield self._make_json_serializable({
            'stage': 'feature_selection',
            'epoch': epoch,
            'selected_features': new_feature_set,
            'action_taken': {'feature': feature_idx_action, 'operation': operation_enum.name}
        })

        binary_mask = np.zeros(self.classifier.state_dim, dtype=np.int32)
        if new_feature_set:
            binary_mask[new_feature_set] = 1

        val_metrics = self.classifier.train(X_train_np, y_train_np, binary_mask)
        val_metrics['num_features'] = len(new_feature_set)
        val_metrics['features'] = new_feature_set  # always available for the buffer
        new_f1 = val_metrics.get('f1', 0.0)

        if self.steps_done > self.config['warmup_steps']:
            is_new_bsf = (
                new_f1 > self.best_f1_so_far or
                (abs(new_f1 - self.best_f1_so_far) < 1e-4 and len(new_feature_set) < len(self.best_features_so_far))
            )
            if is_new_bsf:
                self.best_f1_so_far = new_f1
                self.best_features_so_far = self.agent.current_features.copy()
                logger.info(f" New best F1: {self.best_f1_so_far:.4f} with {len(new_feature_set)} features")
                if self.save_path:
                    self.agent.save(self.save_path)
                    try:
                        best_policy_data = {
                            'best_f1_score': self.best_f1_so_far,
                            'best_features': sorted(list(self.best_features_so_far))
                        }
                        best_policy_file_path = os.path.join(self.save_path, 'best_policy.json')
                        with open(best_policy_file_path, 'w') as f:
                            json.dump(best_policy_data, f, indent=4)
                        logger.info(f" Best policy saved to {best_policy_file_path}")
                    except Exception as e:
                        logger.error(f"Error saving best policy to JSON: {e}")

        yield self._make_json_serializable({
            'stage': 'accuracy',
            'epoch': epoch,
            'accuracy': val_metrics.get('accuracy', 0.0),
            'f1_score': new_f1,
            'selected_features': new_feature_set
        })

        unclipped_reward, reward_components = self.reward_fn.calculate(
            previous_metrics=previous_metrics,
            new_metrics=val_metrics,
            action=operation_enum
        )
        sac_reward = unclipped_reward
        logger.info(
            f"[Interaction] Epoch {epoch}: Action={action_source}/{operation_enum.name} F{feature_idx_action} -> "
            f"New F1={val_metrics.get('f1', 0):.3f}, Num Feats={val_metrics['num_features']}, "
            f"Reward={unclipped_reward:.3f} (Clipped to: {sac_reward:.3f})"
        )

        next_sac_state = self._get_sac_state(val_metrics, set(new_feature_set))
        experience = {
            "state": state_for_experience,
            "features": previous_metrics['features'],
            "action": (feature_idx_action, operation_action_value),
            "reward": sac_reward,
            "next_state": next_sac_state.cpu().numpy(),
            "next_features": self.agent.current_features,
            "done": False
        }
        self.sac_replay_buffer.add(experience)
        self.current_sac_state_tensor = next_sac_state  # advance to next state

        if self.steps_done > self.config['warmup_steps'] and len(self.sac_replay_buffer) >= self.config['sac_batch_size']:
            logger.info(
                f"[Learning] Sampling batch of {self.config['sac_batch_size']} experiences "
                f"from buffer (ID: {self.buffer_id}, current size: {len(self.sac_replay_buffer)})."
            )

            transitions, indices, is_weights = self.sac_replay_buffer.sample(self.config['sac_batch_size'])
            if transitions:
                batch = self._process_sac_batch_for_update(transitions)
                is_weights_tensor = torch.tensor(is_weights, dtype=torch.float32, device=self.device).unsqueeze(1)

                log_n = 10
                logger.info("------ [ReplayBuffer] Sampled Experiences for Update ------")
                for i in range(min(log_n, len(transitions))):
                    state = batch['states'][i].cpu().numpy()
                    feat_idx = batch['feature_idx'][i].item()
                    op_idx = batch['op_idx'][i].item()
                    reward = batch['rewards'][i].item()
                    done = batch['dones'][i].item()
                    logger.info(
                        f"[ReplayBuffer][Sampled] idx={i} | state={state} | action=(feature {feat_idx}, {Operation(op_idx).name}) | "
                        f"reward={reward:.3f} | done={done}"
                    )

                new_priorities, returned_losses = self.agent.update(batch=batch, is_weights=is_weights_tensor)
                self.sac_replay_buffer.update_priorities(indices, new_priorities.cpu().numpy())

                returned_losses = dict(returned_losses)  # make it mutable/plain

                returned_losses.setdefault("alpha_op",   float(self.agent.alpha_op.item()))
                returned_losses.setdefault("alpha_feat", float(self.agent.alpha_feat.item()))

                with torch.no_grad():
                    st = self.current_sac_state_tensor.unsqueeze(0)
                    op_dist, shared_emb = self.agent.actor(st)
                    op_entropy = op_dist.entropy().item()

                    expected_feat_entropy = 0.0
                    op_probs = op_dist.probs
                    for op_val in range(self.agent.num_operations):
                        op_tensor = torch.full((1,), op_val, device=self.device, dtype=torch.long)
                        feat_dist = self.agent.actor.get_feature_dist(shared_emb, op_tensor)
                        expected_feat_entropy += op_probs[:, op_val] * feat_dist.entropy()
                    feat_entropy = float(expected_feat_entropy.item())

                returned_losses.setdefault("op_entropy",   float(op_entropy))
                returned_losses.setdefault("feat_entropy", float(feat_entropy))

                returned_losses.setdefault("target_entropy_op",   float(getattr(self.agent, "target_entropy_op", 0.0)))
                returned_losses.setdefault("target_entropy_feat", float(getattr(self.agent, "target_entropy_feat", 0.0)))

                yield self._make_json_serializable({
                    'stage': 'sac_update',
                    'epoch': epoch,
                    'reward_total': sac_reward,
                    'reward_components': reward_components,
                    'losses': returned_losses,             
                    'selected_features': new_feature_set
                })

    def _process_sac_batch_for_update(self, transitions: List[Dict]) -> Dict[str, Any]:
  
        states = np.array([t['state'] for t in transitions], dtype=np.float32)
        features = [t['features'] for t in transitions] # This will be a list of sets
        actions = [t['action'] for t in transitions]
        rewards = np.array([t['reward'] for t in transitions], dtype=np.float32).reshape(-1, 1)
        next_states = np.array([t['next_state'] for t in transitions], dtype=np.float32)
        next_features = [t['next_features'] for t in transitions] # This will be a list of sets
        dones = np.array([t['done'] for t in transitions], dtype=np.bool_).reshape(-1, 1)

        feature_indices_np = np.array([a[0] for a in actions], dtype=np.int64)
        op_indices_np = np.array([a[1] for a in actions], dtype=np.int64)

        return {
            'states': torch.from_numpy(states).to(self.device),
            'features': features, # Pass the list of sets directly
            'feature_idx': torch.from_numpy(feature_indices_np).to(self.device),
            'op_idx': torch.from_numpy(op_indices_np).to(self.device),
            'rewards': torch.from_numpy(rewards).to(self.device),
            'next_states': torch.from_numpy(next_states).to(self.device),
            'next_features': next_features, # Pass the list of sets directly
            'dones': torch.from_numpy(dones).to(self.device),
        }

    def _get_sac_state_metrics(self, state_tensor: torch.Tensor) -> Dict[str, float]:

        state_np = state_tensor.cpu().numpy()
        return {'f1': state_np[1]}

    def _make_json_serializable(self, data: Any) -> Any:
        
        if isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        if isinstance(data, (list, tuple, set)):
            return [self._make_json_serializable(i) for i in data]
        if isinstance(data, (np.intc, np.intp, np.int8,
                             np.int16, np.int32, np.int64, np.uint8,
                             np.uint16, np.uint32, np.uint64)):
            return int(data)
        if isinstance(data, (np.float16, np.float32, np.float64)):
            return float(data)
        if isinstance(data, np.bool_):
            return bool(data)

        return data
