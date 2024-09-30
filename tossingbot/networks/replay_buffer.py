import random

from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=10000):
        """
        Simple replay buffer to store transitions for experience replay.

        Args:
            capacity (int): Maximum size of the replay buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        """
        Add a new transition to the replay buffer.
        
        Args:
            transition (tuple): A tuple (state, action, reward, next_state, done).
        """
        self.buffer.append(transition)

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.
        
        Args:
            batch_size (int): Number of samples to draw from the buffer.
            
        Returns:
            list: A list of sampled transitions.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """
        Return the current size of the buffer.
        
        Returns:
            int: Current size of the buffer.
        """
        return len(self.buffer)

    def unpack_batch(self, batch):
        """Unpacks a list of tuples into separate lists, handling an unknown number of items in each tuple."""
        # Use zip(*) to unpack the batch into multiple lists
        unpacked = list(zip(*batch))
        
        # Convert tuples to lists
        return [list(items) for items in unpacked]