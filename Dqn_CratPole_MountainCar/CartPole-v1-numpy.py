import gym
import numpy as np
import math
from collections import deque



def choose_action(observation,epsilon):
    return env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(Q[observation])

def discretize(observation):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(observation[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(observation))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(observation))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(observation))]
    return tuple(new_obs)

def get_epsilon(t):
    return max(MIN_EPSILON, min(1, 1.0 - math.log10((t + 1) / ADA_DIVISOR)))

def get_alpha(t):
    return max(MIN_ALPHA, min(1, 1.0 - math.log10((t + 1) / ADA_DIVISOR)))



MAX_EPISODE=1000
ACTION=['left','right']
ADA_DIVISOR=25

MIN_EPSILON=0.1
MIN_ALPHA=0.1
GAMMA=1

env = gym.make('CartPole-v1')


buckets=(1, 1, 6, 12,)
Q = np.zeros(buckets + (env.action_space.n,))




def update_q(state_old, action, reward, state_new, alpha):
    Q[state_old][action] += alpha * (reward + GAMMA * np.max(Q[state_new]) - Q[state_old][action])


scores = deque(maxlen=100)
for i_episode in range(MAX_EPISODE):
    current_state = discretize(env.reset())
    
    epsilon=get_epsilon(i_episode)
    alpha=get_alpha(i_episode)
    
    for t in range(501):
        # env.render()
        action=choose_action(current_state,epsilon)
        obs, reward, done, info = env.step(action)
        new_state=discretize(obs)
        update_q(current_state, action, reward, new_state, alpha)
        current_state=new_state
        if done:
            scores.append(t+1)
            mean_score = np.mean(scores)
            print(f"{i_episode} Episode finished after {t+1} timesteps. Mean score is {mean_score}")
            break
    

    

env.close()

print(Q)