import numpy as np
from replay_buffer import ReplayBuffer


def test_replay_buffer():
    buf = ReplayBuffer(100, (16, 16, 1), (1,), True, 4)
    buf._count = 99
    buf._ptr = 0
    import pdb;pdb.set_trace()

    # TODO
