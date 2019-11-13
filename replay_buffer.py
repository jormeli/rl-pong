import numpy as np
import random


class ReplayBuffer(object):
    """Experience replay buffer with priority sampling."""

    def __init__(self, capacity, state_shape, action_shape, prioritized=True, stack_size=None):
        """Creates a new buffer.

        capacity (int): The maximum number of items in the new buffer.
        state_shape (tuple): Shape of states stored in the buffer (assumes channels last).
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
                                       ('td_err', np.float32)])


        self._ptr = 0
        self._count = 0
        self._prioritized = prioritized


    def store_transition(self, state, action, reward, next_state, done):
        """Stores a new transition in the buffer.
        In case the buffer is already at maximum capacity, starts overwriting
        from position 0."""

        item = (state, action, reward, next_state, done, 1e10)
        self._buffer[self._ptr] = item
        self._ptr = (self._ptr + 1) % (self._capacity - 1)
        self._count = min(self._count + 1, self._capacity)


    @property
    def count(self):
        """Number of items currently stored in this buffer."""

        return self._count

    @property
    def capacity(self):
        """The maximum capacity of this buffer."""

        return self._capacity

    def sample_batch(self, batch_size):
        """Sample a batch of items from this buffer.

        batch_size (int): Size of the batch to be sampled."""

        def _softmax(x):
            x = np.exp(x)
            return x / (x + 1e-6).sum()

        end_idx = self.count - 1
        batch_size = min(end_idx, batch_size)

        if self._prioritized:
            idxs = np.arange(end_idx, dtype=np.uint32)
            probs = self._buffer[:end_idx]['td_err']
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
            stack_range = np.arange(-self._stack_size, 0)
            stack_idxs = batch_idxs[:, None] + stack_range

            is_neg = stack_idxs < 0
            stacked_batch = self._buffer[stack_idxs].copy()
            stacked_batch[is_neg] = 0.

            for item in stacked_batch:
                # If any but the last item is done, set it and preceding items to zero.
                # This means that those items are from a different episode.
                if item['done'][:self._stack_size-1].any():
                    done_idx = np.argmax(item['done'][::-1])
                    zero_until = self._stack_size - done_idx
                    item[:zero_until] = 0

            states = stacked_batch['state'].transpose([0, 2, 3, 4, 1]) \
                    .reshape((batch_size,) + self._state_shape[:-1] + (-1,))
            next_states = stacked_batch['next_state'].transpose([0, 2, 3, 4, 1]) \
                    .reshape((batch_size,) + self._state_shape[:-1] + (-1,))

        return (states, actions, rewards, next_states, dones), batch_idxs

    def set_td_errors(self, idxs, errs):
        """Update TD errors for items when they are used in training.

        idxs (numpy.NDArray): Corresponding indices for the new error values.
        errs (numpy.NDArray): New error values."""

        errs = errs.reshape((-1,))
        assert len(idxs) == len(errs)

        self._buffer[idxs]['td_err'] = errs

    def reset(self):
        """Reset this buffer."""

        self._count = 0
        self._ptr = 0