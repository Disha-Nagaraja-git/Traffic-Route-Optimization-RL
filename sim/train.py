import numpy as np
import pickle
import csv
import matplotlib.pyplot as plt

from maze_env import *

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.3
EPISODES = 2000
MAX_STEPS = 100

Q_table = np.zeros((ROWS, COLS, 4))

rewards_per_episode = []
steps_per_episode = []

for episode in range(EPISODES):

    state = START_STATE

    total_reward = 0

    for step in range(MAX_STEPS):

        row, col = state

        if np.random.uniform(0,1) < EPSILON:
            action = np.random.randint(4)

        else:
            action = np.argmax(Q_table[row,col])

        next_state = get_next_state(state, action)

        reward = get_reward(next_state)

        next_row, next_col = next_state

        old_q = Q_table[row,col,action]

        next_max = np.max(Q_table[next_row,next_col])

        new_q = old_q + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * next_max - old_q
        )

        Q_table[row,col,action] = new_q

        state = next_state

        total_reward += reward

        if state == GOAL_STATE:
            break

    rewards_per_episode.append(total_reward)
    steps_per_episode.append(step + 1)

# Save trained policy
with open("../policies/policy_v1.pkl", "wb") as f:
    pickle.dump(Q_table, f)

print("Training Complete")

# Save experiment results CSV
with open("../experiments/experiment_1.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(["episode", "reward", "steps"])

    for i in range(len(rewards_per_episode)):

        writer.writerow([
            i + 1,
            rewards_per_episode[i],
            steps_per_episode[i]
        ])

print("Experiment CSV saved successfully")

# Reward Plot
plt.figure()

plt.plot(rewards_per_episode)

plt.title("Training Reward Progress")
plt.xlabel("Episodes")
plt.ylabel("Reward")

plt.savefig("../plots/reward_plot.png")

print("Reward plot saved successfully")

# Steps Plot
plt.figure()

plt.plot(steps_per_episode)

plt.title("Steps Taken per Episode")
plt.xlabel("Episodes")
plt.ylabel("Steps")

plt.savefig("../plots/steps_plot.png")

print("Steps plot saved successfully")