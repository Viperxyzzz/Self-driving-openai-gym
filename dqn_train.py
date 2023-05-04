import gymnasium as gym
import numpy as np
import random
import math

import torch
import torch.optim as optim
import torchvision.transforms as trans
import torch.nn.functional as F

import matplotlib.pyplot as plt

from dqn_model import CNN
from replay_memory import ReplayMemory, Transition

env = gym.make("CarRacing-v2")
test_env = gym.make("CarRacing-v2", render_mode="human")

# actions that agent can take
# manually chosen action from continuous action space to discrete action space
# 0: STEER LEFT
# 1: DO NOTHING
# 2: STEER RIGHT
# 3: FULL GAS
# 4: FULL BREAK
ACTIONSPACE = [[-1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
n_actions = len(ACTIONSPACE)

BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
LEARNING_RATE = 0.001

# count randomly chosen action for each episode
RANDOM_COUNTER = 0
# count actions chosen from neural network for each episode
NN_COUNTER = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NET = CNN(n_actions).to(DEVICE)

OPTIMIZER = optim.RMSprop(NET.parameters(), lr=LEARNING_RATE)
MEMORY = ReplayMemory(2000)

# count steps taken in each episode
STEPS_DONE = 0

# how many episodes to play
NUM_EPISODES = 10
# list of scores from each episode
TOTAL_SCORES = []
# list of average calculated scores in each run
SCORE_AVG = []
# list of losses for each episode
TOTAL_AVG_LOSSES = []


# wait until env zooms to car
def wait_for_zoom():
    for x in range(0, 50):
        env.step([0, 0, 0])


# transform each env frame for program use
def get_screen(screen):
    transformation = trans.Compose([trans.ToTensor(), trans.ToPILImage(), trans.Grayscale(), trans.ToTensor()])

    screen = np.flip(screen, axis=0).copy()
    screen = np.flip(screen, axis=0).copy()
    screen = transformation(screen).to(DEVICE)

    track = screen[:, :66, 15:81]
    track = track.view(1, 1, 66, 66)

    return track


def select_action(state):
    global STEPS_DONE

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * STEPS_DONE / EPS_DECAY)

    # sustain threshold from going below boundary
    if eps_threshold < EPS_END:
        eps_threshold = EPS_END

    STEPS_DONE += 1

    if sample > eps_threshold:
        global NN_COUNTER
        NN_COUNTER += 1

        with torch.no_grad():
            # obtain q values from network based on state
            action_q_vals = NET(state)
            # choose highest q value from network
            selected_action_index = action_q_vals.max(1)[1].view(1, 1)

            return selected_action_index

    else:
        global RANDOM_COUNTER
        RANDOM_COUNTER += 1
        # select random action
        random_select = torch.tensor([[random.randrange(n_actions)]], device=DEVICE, dtype=torch.long)

        return random_select


# perform selected action 4 times
# accumulate states and rewards
def get_res_state(action):
    accumulated_reward = 0
    accumulated_screen = torch.Tensor().to(DEVICE)

    for i in range(4):
        s_observation, s_reward, terminated, truncated, _ = env.step(action)
        s_done = truncated or terminated
        
        accumulated_reward += s_reward

        screen = get_screen(s_observation)
        accumulated_screen = torch.cat((accumulated_screen, screen), dim=1)

    return accumulated_screen, accumulated_reward, s_done


def optimize_model():
    # wait for memory to fill up
    if len(MEMORY) < BATCH_SIZE:
        return

    transitions = MEMORY.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)

    # get q values from states matching current actions in batch
    state_action_values = NET(state_batch).gather(1, action_batch).squeeze(1)
    # get highest q values from next states in batch
    next_state_max_values = NET(next_state_batch).max(1)[0].detach()
    # calculate target
    expected_state_action_values = (next_state_max_values * GAMMA) + reward_batch

    # calculate loss
    loss = F.mse_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    OPTIMIZER.zero_grad()
    loss.backward()
    OPTIMIZER.step()

    return float(loss)


for i_episode in range(NUM_EPISODES):
    episode_losses = []
    total_reward = 0
    RANDOM_COUNTER = 0
    NN_COUNTER = 0
    score_sliding = 0

    # Initialize the environment and state
    print("Starting episode", i_episode)
    env.reset()
    wait_for_zoom()
    state, _, _ = get_res_state([0, 0, 0])

    for t in range(1000):
        # select current action
        action_index = select_action(state)
        action = ACTIONSPACE[action_index]

        # get normalized state from current action
        next_state, reward, done = get_res_state(action)

        total_reward += reward

        reward = float(reward)
        reward = torch.tensor([reward], device=DEVICE)

        if done:
            break

        # Store the transition in memory
        MEMORY.push(state, action_index, reward, next_state, done)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        episode_loss = optimize_model()

        if episode_loss is not None:
            episode_losses.append(episode_loss)

    if len(episode_losses) > 0:
        avgloss = sum(episode_losses)/len(episode_losses)

    else:
        avgloss = 0

    TOTAL_AVG_LOSSES.append(avgloss)

    print(f"Random steps this episode {RANDOM_COUNTER}")
    print(f"NN steps this episode {NN_COUNTER}")
    print(f"Total reward in episode {i_episode} is {total_reward} and awg loss is {avgloss}")
    TOTAL_SCORES.append(total_reward)
    score_sliding = sum(TOTAL_SCORES) / len(TOTAL_SCORES)
    SCORE_AVG.append(score_sliding)

    test_env.reset()
    for _ in range(1000):
        test_env.render()
        action_index = select_action(state)
        action = ACTIONSPACE[action_index]
        env.step(action)

# show graphs
plt.plot(TOTAL_SCORES)
plt.plot(SCORE_AVG)
plt.title(f"Scores per episode, batch size {BATCH_SIZE}")
plt.show()
plt.plot(TOTAL_AVG_LOSSES)
plt.title(f"Avg losses per episode")
plt.show()

print('Complete')

env.close()
