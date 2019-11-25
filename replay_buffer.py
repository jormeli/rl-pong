import numpy as np
import random


class ReplayBuffer(object):
    """Experience replay buffer with priority sampling."""

    def __init__(self, capacity, state_shape, action_shape, prioritized=True, stack_size=None):
        """Creates a new buffer.

        capacity (int): The maximum number of items in the new buffer.
        state_shape (tuple): Shape of states stored in the buffer (assumes CHW).
        action_shape (tuple): Shape of actions stored in the buffer.
        prioritized (bool): Whether to use prioritized sampling.
        stack_size (int): Stack n concurrent states when sampling."""

        self._capacity = capacity
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._stack_size = stack_size

        self._buffer = np.empty((capacity,),
                                dtype=[('state', np.uint8, state_shape),
                                       ('action', np.uint8, action_shape),
                                       ('reward', np.float32),
                                       ('next_state', np.uint8, state_shape),
                                       ('done', np.bool),
                                       ('priority', np.float32)])


        self._ptr = 0
        self._count = 0
        self._prioritized = prioritized

    def store_transition(self, state, action, reward, next_state, done):
        """Stores a new transition in the buffer.
        In case the buffer is already at maximum capacity, starts overwriting
        from position 0."""

        item = (state, action, reward, next_state, done, 1e10)
        self._buffer[self._ptr] = item
        self._ptr = (self._ptr + 1) % self._capacity
        self._count = min(self._count + 1, self._capacity)

    @property
    def count(self):
        """Number of items currently stored in this buffer."""

        return self._count

    @property
    def capacity(self):
        """The maximum capacity of this buffer."""

        return self._capacity

    def sample_batch(self, batch_size, beta=1.0):
        """Sample a batch of items from this buffer.

        batch_size (int): Size of the batch to be sampled.
        beta (float): Shape parameter for importance sampling.
        """

        def _softmax(x):
            x = np.exp(x)
            return x / (x + 1e-6).sum()

        end_idx = self.count - 1
        batch_size = min(end_idx, batch_size)

        if self._prioritized:
            idxs = np.arange(end_idx, dtype=np.uint32)
            probs = self._buffer[:end_idx]['priority']
            probs = _softmax(probs / probs.max()) # Sampling probabilities must sum to 1
            assert all(probs > 0.)
            batch_idxs = np.random.choice(idxs, (batch_size,), False, probs)
        else:
            batch_idxs = np.random.randint(0, end_idx, size=batch_size)

        batch = self._buffer[batch_idxs].copy()
        states = batch['state']
        actions = batch['action']
        rewards = batch['reward']
        next_states = batch['next_state']
        dones = batch['done']

        if self._stack_size is not None:
            stack_range = np.arange(-self._stack_size + 1, 1)
            stack_idxs = batch_idxs[:, None] + stack_range

            is_neg = stack_idxs < 0
            stacked_batch = self._buffer[stack_idxs].copy()

            # If the buffer is not full, copy the first element of the buffer
            # to the place of negative indiced that attempt to read the buffer circularly.
            if self.count < self.capacity:
                stacked_batch[is_neg] = self._buffer[0].copy()

            for item in stacked_batch:
                # If any but the last item is done it means that those
                # items are from a different episode. In that case, set the items
                # from the different episode as the initial observation of the last item's episode.
                if item['done'][:self._stack_size-1].any():
                    # Determine starting index of the episode.
                    ep_start_idx = self._stack_size - np.argmax(item['done'][::-1])
                    initial_state = item['state'][ep_start_idx]
                    initial_next_state = item['next_state'][ep_start_idx]

                    # Repeat initial state and next_state until the episode start
                    # to overwrite observations from last episode.
                    item['state'][:ep_start_idx] = initial_state
                    item['next_state'][:ep_start_idx] = initial_next_state

            # Stack frames so that newest frame is at the last channel.
            states = stacked_batch['state'].reshape((batch_size, -1) + self._state_shape[1:])
            next_states = stacked_batch['next_state'].reshape((batch_size, -1) + self._state_shape[1:])

        # Determine sample weights.
        weights  = (self.count * probs[batch_idxs])**-beta
        weights /= weights.max()

        return (states, actions, rewards, next_states, dones), batch_idxs, weights

    def update_priorities(self, idxs, errs):
        """Update priorities for items when they are used in training.

        idxs (numpy.NDArray): Corresponding indices for the new error values.
        errs (numpy.NDArray): New error values."""

        errs = errs.reshape((-1,))
        assert len(idxs) == len(errs)

        self._buffer['priority'][idxs] = errs

    def reset(self):
        """Reset this buffer."""

        self._count = 0
        self._ptr = 0
