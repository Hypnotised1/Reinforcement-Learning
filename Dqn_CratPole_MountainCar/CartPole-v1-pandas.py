import gym
import numpy as np
import pandas as pd
import math
from collections import deque


def choose_action(observation,epsilon):

    ind=int(observation[2])*12+int(observation[3])
    return env.action_space.sample() if (np.random.random() <= epsilon) else Q.loc[ind].idxmax()

def discretize(observation):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(observation[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(observation))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(observation))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(observation))]
    return new_obs

def get_epsilon(t):
    return max(min_epsilon, min(1, 1.0 - math.log10((t + 1) / ADA_DIVISOR)))
def get_alpha(t):
    return max(min_alpha, min(1, 1.0 - math.log10((t + 1) / ADA_DIVISOR)))

def update_q(state_old, action, reward, state_new, alpha):
    Q.at[state_old,action] +=alpha * (reward + GAMMA * np.max(Q.loc[state_new]) - Q.at[state_old,action])


MAX_EPISODE=1000
ADA_DIVISOR=25
GAMMA=1
env = gym.make('CartPole-v1')


data = np.zeros((1*1*6*12,2))
Q = pd.DataFrame(data)

buckets=(1, 1, 6, 12,)

min_epsilon=0.1
min_alpha=0.1




scores = deque(maxlen=100)



for i_episode in range(MAX_EPISODE):
    observation = env.reset()
    
    epsilon=get_epsilon(i_episode)
    alpha=get_alpha(i_episode)
    current_state=discretize(observation)

    for t in range(501):
        # env.render()
        action=choose_action(current_state,epsilon)
        obs, reward, done, info = env.step(action)
        new_state=discretize(obs)


        ind=int(current_state[2])*12+int(current_state[3])
        indn=int(new_state[2])*12+int(new_state[3])


        update_q(ind,action,reward,indn,alpha)


        current_state=new_state

        if done:
            scores.append(t+1)
            mean_score = np.mean(scores)
            print(f"{i_episode} Episode finished after {t+1} timesteps. Mean score is {mean_score}")

                
            break
    
env.close()

print(Q)