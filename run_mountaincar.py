import gym
import numpy as np
from RL_barin import DQN
from collections import deque
def run():

    # env = gym.make('MountainCarContinuous-v0')
    RL = DQN(n_actions=env.action_space.n,
                    n_features=env.observation_space.shape[0],
                    learning_rate=0.001, e_greedy=0.9,reward_decay=0.9,
                    replace_target_iter=300, memory_size=3000,
                    e_greedy_increment=0.00005,
                    model_name='./mountaincar.h5')

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    MAX_STEP=300
    total_step=0
    success_count=0
    for i in range(MAX_STEP):
        observation=env.reset()
        ep_r=0
        while True:
            #env.render()
            action=RL.choose_action(observation,0)
            observation_, reward, done, info = env.step(action)


            position, velocity = observation_
            # the higher the better
            reward = abs(position - (-0.5))     # r in [0, 1]

            # the higher the better, the faster the better
            # reward = abs(0.7*position +0.3*velocity)

            RL.store_transition(observation,action,reward,observation_)
            
            if total_step>1000:
                RL.learn()
            ep_r+=reward
            if done:
                if  observation_[0] >= env.unwrapped.goal_position:
                    success_count+=1
                    get='| Get'
                else:
                    get = '| ----'
                print('Epi: ', i,
                    get,
                    '| Ep_r: ', round(ep_r, 4),
                    '| Epsilon: ', round(RL.epsilon, 2),
                    'success count', success_count)
                break
            observation=observation_
            total_step+=1
    RL.save_model()
    RL.plot_loss()

env = gym.make('MountainCar-v0')

if __name__ == "__main__":
    run()