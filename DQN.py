import gym
import math
import numpy as np

from collections import deque
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.compat.v2.keras.callbacks import TensorBoard
class DQN:
    def __init__(self,
        learning_rate=0.01,
        batch_size=32,
        epsilon=0.1,
        n_features=0,
        n_actions=0,
        memory_size=1000,
        replace_target_iter=300,
        e_greedy=0.9,
        reward_decay=0.9,
        e_greedy_increment=None,
        model_name=None,


        ):
        self.lr=learning_rate
        self.batch_size=batch_size
        
        self.n_features = n_features
        self.n_actions = n_actions
        self.memory_size = memory_size
        self.epsilon_max = e_greedy
        self.memory=np.zeros((self.memory_size,self.n_features*2+2))
        self.replace_target_iter=replace_target_iter
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.gamma = reward_decay
        self.learn_step_counter=0
        self.epsilon_increment = e_greedy_increment

        self._bulid_net()
        self.loss=[]
        self.model_name=model_name


    def _bulid_net(self):
        inputs = keras.Input(shape=(self.n_features,))
        x = keras.layers.Dense(64,activation='relu')(inputs)
        x = keras.layers.Dense(64,activation='relu')(x)
        outputs = keras.layers.Dense(self.n_actions)(x)
        self.eval_net = keras.Model(inputs=inputs, outputs=outputs)
        self.eval_net.compile(optimizer=RMSprop(lr=self.lr),
                    loss='mean_squared_error',
                    metrics=['accuracy'])

        config = self.eval_net.get_config()
        self.target_net = keras.Model.from_config(config)
        weights = self.eval_net.get_weights()  # Retrieves the state of the model.
        self.target_net.set_weights(weights)  # Sets the state of the model.
        self.target_net.compile(optimizer=RMSprop(lr=self.lr),
                    loss='mean_squared_error',
                    metrics=['accuracy'])

        self.eval_net.summary()
        self.target_net.summary()

        
        self.tbCallBack = TensorBoard(log_dir='./tmp',  # log 目录
                histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                #batch_size=32,     # 用多大量的数据计算直方图
                write_graph=True,  # 是否存储网络结构图
                # write_grads=True, # 是否可视化梯度直方图
                write_images=True,# 是否可视化参数
                embeddings_freq=0, 
                embeddings_layer_names=None, 
                embeddings_metadata=None)

    def get_epsilon(self, i_episode, max_episode):
        #使用 sigmoid 对epsilon进行配置, 先将训练轮数进行缩放, 然后再还原到[-5,5]的区间内, 最后是使用sigmoid进行输出

        #缩放 , 并还原至 -5,5
        i_episode= i_episode/max_episode * 10 -5
        sigmoid_e=1 / (1 + math.exp(-i_episode))

        epsilion= min(self.epsilon_max, sigmoid_e)
        # epsilion= self.epsilon_max if self.epsilon_max < sigmoid_e else sigmoid_e
        
        return epsilion




    

    def store_transition(self,state, action, reward, newstate):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        index = self.memory_counter % self.memory_size
        self.memory[index]=np.hstack((state,action,reward,newstate))
        self.memory_counter += 1



    def choose_action(self, state, epsilon):
        if np.random.random() < self.epsilon:
            state=state[np.newaxis,:]
            result=self.eval_net.predict(state)
            return np.argmax(result)
        else:
            return env.action_space.sample()
    
    def update_weight(self):
        weights = self.eval_net.get_weights()  # Retrieves the state of the model.
        self.target_net.set_weights(weights)  # Sets the state of the model.

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter ==0:
            self.update_weight()
            print('target params replaced')
        
        if self.memory_counter > self.memory_size:
            sample_index=np.random.choice(self.memory_size,size=self.batch_size)
        else:
            sample_index=np.random.choice(self.memory_counter,size=self.batch_size)
        
        batch_memory=self.memory[sample_index,:]

        
        q_tar, q_eval=self.target_net.predict(batch_memory[:,-self.n_features:]),self.eval_net.predict(batch_memory[:,:self.n_features])

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward=batch_memory[:,self.n_features+1]

        
        q_target = q_eval.copy()
        q_target[batch_index,eval_act_index]= reward + self.gamma*np.max(q_tar,axis=1)

        hist=self.eval_net.fit(batch_memory[:,:self.n_features],q_target,epochs=10,verbose=0,callbacks=[self.tbCallBack])
        self.loss.append(hist.history['loss'][-1])
        # print(hist.history['loss'])

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
    
    def save_model(self):
        print('Model is saving!')
        self.eval_net.save(self.model_name)
        print('Model saves success!')

    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.loss)), self.loss)
        plt.ylabel('Loss')
        plt.xlabel('training steps')
        plt.savefig('./Normal1.png')
        plt.show()

def cartpole():

    
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
            r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
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
    

def mountaincar():

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
# env = gym.make('CartPole-v1')
env = gym.make('MountainCar-v0')

# env = env.unwrapped

if __name__ == "__main__":
    mountaincar()
    # cartpole()