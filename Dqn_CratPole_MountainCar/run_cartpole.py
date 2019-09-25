import gym
import numpy as np
from RL_barin import DQN
from collections import deque

def run():
    SCORES=deque(maxlen=100)
    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    RL = DQN(n_actions=env.action_space.n,
                    n_features=env.observation_space.shape[0],
                    learning_rate=0.01, e_greedy=0.9,reward_decay=0.9,
                    replace_target_iter=100, memory_size=2000,
                    e_greedy_increment=0.001,
                    model_name='./cartpole.h5')

    total_steps = 0

    max_episode=200
    for i_episode in range(max_episode):

        observation = env.reset()
        ep_r = 0
        ttt=0
        epsilon=RL.get_epsilon(i_episode,max_episode)
        while True:
            #env.render()

            action = RL.choose_action(observation,epsilon)

            observation_, reward, done, info = env.step(action)

            # the smaller theta and closer to center the better
            x, x_dot, theta, theta_dot = observation_
            r1 = (env.unwrapped.x_threshold - abs(x))/env.unwrapped.x_threshold - 0.8
            r2 = (env.unwrapped.theta_threshold_radians - abs(theta))/env.unwrapped.theta_threshold_radians - 0.5
            reward = r1 + r2
            # reward = reward if not done else -reward
            RL.store_transition(observation, action, reward, observation_)
            ttt+=1
            ep_r += reward
            if total_steps > 1000:
                RL.learn()

            if done:
                SCORES.append(ttt)
                print('episode: ', i_episode,
                    'ep_r: ', round(ep_r, 2),
                    'ttt: ', round(ttt, 2),
                    'mean scores: ', round(np.mean(SCORES),2),
                    'epsilon: ', round(RL.epsilon, 2),
                    'epsilon: ', round(epsilon, 2))
                break

            observation = observation_
            total_steps += 1
    RL.save_model()
    RL.plot_loss()
env = gym.make('CartPole-v1')

# env = env.unwrapped

if __name__ == "__main__":
    run()