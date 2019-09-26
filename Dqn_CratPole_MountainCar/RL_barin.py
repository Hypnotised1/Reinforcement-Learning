import math
import numpy as np

from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard
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
            return np.random.choice(self.n_actions)
    
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
        self.target_net.save(self.model_name)
        print('Model saves success!')

    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.loss)), self.loss)
        plt.ylabel('Loss')
        plt.xlabel('training steps')
        plt.savefig('./Normal1.png')
        plt.show()

