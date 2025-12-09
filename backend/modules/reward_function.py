"""
reward_function.py

Defines the RewardFunction used by the IRDF policy-training pipeline to score
state transitions during feature selection. The reward combines F1-score,
sparsity (number of features), and action-dependent bonuses to encourage
compact, high-performing feature subsets and stable policies.
"""
from enum import IntEnum
from typing import Dict, Tuple

class Operation(IntEnum):
   
    ADD = 0
    REMOVE = 1
    NO_OP = 2

class RewardFunction:
    def __init__(self, target_num_features: int = 2):
        self.target_num_features = target_num_features
        self.optimal_f1 = 0.999

        self.R_noop = 4.0

        self.R_remove = 1.0

        self.R_add = -3.0
      
        self.feature_penalty_weight = -0.2  

        self.bloat_additional_penalty = -4.0
        self.bloat_threshold = 2.0  

    def calculate(self, previous_metrics: Dict, new_metrics: Dict, action: Operation) -> Tuple[float, Dict]:
        new_f1 = float(new_metrics.get('f1', 0.0))
        num_features = max(int(new_metrics.get('num_features', 1)), 1)

        sparsity_ratio = self.target_num_features / num_features
        optimal_state = float(new_f1 >= self.optimal_f1)
        bloated = float(num_features > self.bloat_threshold * self.target_num_features)

        is_remove = float(action == Operation.REMOVE)
        is_noop = float(action == Operation.NO_OP)
        is_add = float(action == Operation.ADD)

        feature_penalty = num_features * self.feature_penalty_weight

        base_reward = new_f1 * sparsity_ratio

        optimal_bonus = 0.0
        if optimal_state and not bloated:

            optimal_bonus = (
                is_remove * self.R_remove +
                is_noop * self.R_noop +
                is_add * self.R_add
            )
        elif optimal_state and bloated:
            
            optimal_bonus = (
                is_remove * (self.R_remove + 1.0)   
                + is_noop * (-5.0)                 
                + is_add * (self.R_add - 2.0)     
            )

        final_reward = base_reward + optimal_bonus + feature_penalty

        reward_components = {
            'f1_score': new_f1,
            'num_features': num_features,
            'base_reward': base_reward,
            'optimal_bonus': optimal_bonus,
            'feature_penalty': feature_penalty,
            'final_reward': final_reward
        }

        return final_reward, reward_components