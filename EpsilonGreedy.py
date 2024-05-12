import numpy as np
import gymnasium as gym
from simpleTabEnv import GridWorldEnv
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt

class EpsilonGreedy():

    def __init__(self, env_name, size=10, epsilon=0.05, alpha=0.1, lamb=0.99, IM=False):
        
        register(
            id='GridWorld-v0',
            entry_point=GridWorldEnv,
            max_episode_steps=size*2,
        )

        self.env = gym.make(env_name, size=size)
        self.epsilon = epsilon
        self.alpha = alpha
        self.lamb = lamb
        self.stepPenalty = 0.01
        self.IM = IM
        self.counts = {(m,n): 1 for m in range(size) for n in range(size)}

        # Init q_values
        size = self.env.observation_space["agent"].high[0] + 1
        self.q_values = {(m,n):[-self.stepPenalty]*4 for m in range(size) for n in range(size)}


    def learn(self, episodes, batchsize=100):
        
        batch_returns = []
        self.history = []
        self.batchsize = batchsize
        self.episodes = episodes

        for i in range(episodes):
            obs, info = self.env.reset()
            obs = tuple(obs["agent"])
            terminated, truncated = False, False
            returns = 0

            while not (terminated or truncated):
                current_q = self.q_values[obs]
                
                # Exploration
                if np.random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                
                # Exploitation
                else:
                    max_q = max(current_q)
                    max_actions = [i for i in range(len(current_q)) if current_q[i] == max_q]
                    action = np.random.choice(max_actions)

                prev_obs = obs
                obs, reward, terminated, truncated, info = self.env.step(action)
                obs = tuple(obs["agent"])

                # Negative rewards for each step + curiosity reward
                reward -= self.stepPenalty 
                reward += self.curiosityCounts(obs)
                returns += reward

                # Update Q-Values
                new_max_q = max(self.q_values[obs])
                self.q_values[prev_obs][action] = (1 -  self.alpha) * self.q_values[prev_obs][action] + self.alpha * (reward + self.lamb * new_max_q)
            
            batch_returns.append(returns)

            if (i+1) % batchsize == 0:
                self.history.append(sum(batch_returns) / len(batch_returns))
                batch_returns = []        


    def predict(self, obs_raw):
        obs = tuple(obs_raw["agent"])
        
        current_q = self.q_values[obs]
        max_q = max(current_q)
        max_actions = [i for i in range(len(current_q)) if current_q[i] == max_q]
        action = np.random.choice(max_actions)
        return action
    
    def test(self, episodes):
        returns = []
        for _ in range(episodes):
            obs, info = self.env.reset()
            terminated, truncated = False, False
            rewards = []
            
            while not (terminated or truncated):
                obs = tuple(obs["agent"])
                current_q = self.q_values[obs]

                max_q = max(current_q)
                max_actions = [i for i in range(len(current_q)) if current_q[i] == max_q]
                action = np.random.choice(max_actions)

                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Negative rewards for each step
                reward -= self.stepPenalty 
                rewards.append(reward)
            
            returns.append(sum(rewards))

        print("Average return: ", sum(returns) / len(returns))


    def learningGraph(self):
        plt.plot([0] + [i for i in np.arange(0, self.episodes, self.batchsize)], [0] + self.history)
        plt.xlabel("Episodes (in batches)")
        plt.ylabel("Mean return per episode")
        plt.ylim(0,1)

    
    def curiosityCounts(self, obs):
        if not self.IM:
            return 0
        
        self.counts[obs] += 1
        return 1/self.counts[obs]