import numpy as np
import matplotlib.pyplot as plt
import random

STATES=np.arange(1,100,dtype=int)

v=np.zeros(101)
v[100]=1

ph=0.4
    
while True:
    newv=np.copy(v)
    
    for state in STATES:
        rewards=[]
        
        for action in np.arange(min(state,100-state)+1):
            reward=ph*v[state+action]+(1-ph)*v[state-action]
            rewards.append(reward)

        
        newv[state]=np.max(rewards)
        
    value_change=abs(newv-v).sum()
    print('value change:'+str(value_change))

    if value_change<1e-9:
        print(np.transpose(v))
        
        break
    v=newv

policy=np.zeros(101)
policy[100]=0
for state in STATES:
    rewards=[]
    
    for action in np.arange(min(state,100-state)+1):
        reward=ph*v[state+action]+(1-ph)*v[state-action]
        rewards.append(reward)
    
    ## https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/issues/83
    action=np.arange(min(state,100-state)+1)[np.argmax(np.round(rewards[1:],5))+1]
    

    policy[state]=action
print(np.transpose(policy))


fig,ax=plt.subplots(2,1)
ax[0].plot(v)
ax[1].scatter(np.arange(101),np.transpose(policy))



plt.show()

        
