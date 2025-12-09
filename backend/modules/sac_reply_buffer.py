"""
Prioritized Experience Replay buffer used by the SAC agent.
Supports priority-based sampling using a SumTree structure and
computes importance-sampling weights for stable training.
"""
import numpy as np
import random
import logging
import uuid

logger = logging.getLogger(__name__)

class SumTree:
   
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
       
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, s):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if s <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    s -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]

class PrioritizedReplayBuffer:
   
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, epsilon=0.01, small_e=1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = epsilon
        self.small_e = small_e
        self.max_priority = 1.0
        
        self.buffer_id = str(uuid.uuid4())[:8]
        logger.info(f"Initialized new PrioritizedReplayBuffer with ID: {self.buffer_id}")

    def clear(self):
       
        self.tree = SumTree(self.tree.capacity)
        self.max_priority = 1.0
        logger.info(f"PrioritizedReplayBuffer (ID: {self.buffer_id}) has been cleared.")

    def add(self, experience):
 
        self.tree.add(self.max_priority, experience)
        if self.tree.n_entries > 0 and self.tree.n_entries % 1000 == 0:
            logger.debug(f"Buffer (ID: {self.buffer_id}) now contains {self.tree.n_entries} experiences.")

    def sample(self, batch_size):
        if self.tree.n_entries == 0:
            return [], [], [] 

        batch = []
        idxs = []
        segment = self.tree.total_priority / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
  
            if b > a:
                s = random.uniform(a, b)
                (idx, p, data) = self.tree.get_leaf(s)
            
                if data is not None:
                    priorities.append(p)
                    batch.append(data)
                    idxs.append(idx)
        
        if not priorities:
            return [], [], []
            
        sampling_probabilities = np.array(priorities) / (self.tree.total_priority + self.small_e)
        
        is_weight = np.power((self.tree.n_entries * sampling_probabilities) + self.small_e, -self.beta)
        is_weight /= (is_weight.max() + self.small_e)

        return batch, idxs, is_weight

    def update_priorities(self, tree_idxs, abs_td_errors):
        if not isinstance(abs_td_errors, np.ndarray):
            abs_td_errors = np.array(abs_td_errors)
            
        priorities = np.power(abs_td_errors + self.epsilon, self.alpha)
        
        if priorities.size > 0:
            self.max_priority = max(self.max_priority, np.max(priorities))
        
        for i, idx in enumerate(tree_idxs):
            self.tree.update(idx, priorities[i])


    def __len__(self):
        return self.tree.n_entries