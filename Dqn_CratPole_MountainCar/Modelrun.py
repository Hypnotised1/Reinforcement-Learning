import gym
import numpy as np
from collections import deque
from tensorflow import keras


def choose_action(state):
    state=state[np.newaxis,:]
    result=eval_net.predict(state)
    return np.argmax(result)

model_name='./cartpole.h5'
# model_name='./mountaincar.h5'

eval_net = keras.models.load_model(model_name)
eval_net.summary()

env = gym.make('CartPole-v1')
# env = gym.make('MountainCar-v0')
# env = env.unwrapped

    
SCORES=deque(maxlen=100)
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)



for i_episode in range(10):

    observation = env.reset()
    ttt=0
    while True:
        env.render()

        action = choose_action(observation)
        observation_, reward, done, info = env.step(action)

        ttt+=1

        if done:
            SCORES.append(ttt)
            print('episode: ', i_episode,
                'ttt: ', round(ttt, 2),
                'mean scoreL: ', np.mean(SCORES))
            break

        observation = observation_
