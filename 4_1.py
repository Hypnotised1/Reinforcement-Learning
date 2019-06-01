import numpy as np
import random

    
def getStateAndReward(state,action):
    state=np.asarray(state)
    if (state[0]==0 and state[1]==0) or (state[0]==3 and state[1]==3):
        return state, 0
    action=np.asarray(action)
    newstate=state+action
    if newstate[0]<0 or newstate[1]<0 or newstate[0]>3 or newstate[1]>3:
        return state,-1
    else:
        return newstate, -1



#上下左右
ACTIONS=[(-1,0),(1,0),(0,-1),(0,1)]


Grid=np.zeros((4,4))
# while True:
for k in range(10):
    newGrid=np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            state=(i,j)
            result=0
            for action in ACTIONS:
                newstate,reward=getStateAndReward(state,action)
                result+=reward+Grid[tuple(newstate)]
            newGrid[state]+=result/4
    print(newGrid)

    if(np.sum(np.abs(newGrid-Grid))<1e-4):
        print(newGrid)
        break
    Grid=newGrid
