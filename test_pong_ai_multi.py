"""
This is an example on how to use the two player Wimblepong environment
with two SimpleAIs playing against each other
"""
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import numpy as np
import argparse
import wimblepong
from PIL import Image

from agent import Agent
from networks import *

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=4)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
# Number of episodes/games to play
episodes = 100000

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = wimblepong.SimpleAi(env, opponent_id)
player = Agent(input_shape=(1, 84, 84),
               num_actions=3,
               network_fn=DuelingDQN,
               network_fn_kwargs=None,
               minibatch_size=128,
               replay_memory_size=500000,
               stack_size=4,
               gamma=0.98,
               beta0=0.9,
               beta1=0.999,
               learning_rate=1e-4,
               device='cuda',
               normalize=False,
               noisy=False,
               prioritized=True)

player.load_model('./models/agent_ep27000.mdl')

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

win1 = 0
ep_lengths = []
for i in range(0,episodes):
    ob1, ob2 = env.reset()
    done = False
    step = 0
    while not done:
        # Get the actions from both SimpleAIs
        action1 = player.get_action(ob1,epsilon=0.0)
        action2 = opponent.get_action()
        # Step the environment and get the rewards and new observations
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        step += 1
        #img = Image.fromarray(ob1)
        #img.save("ob1.png")
        #img = Image.fromarray(ob2)
        #img.save("ob2.png")
        # Count the wins
        if rew1 == 10:
            win1 += 1
        if not args.headless:
            env.render()

    ep_lengths.append(step)
    print("episode {} over. WR: {:.3f}. Average episode length {:.2f} frames.".format(i, win1/(i+1), np.mean(ep_lengths)))
