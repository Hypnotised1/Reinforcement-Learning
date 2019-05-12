import numpy as np
import random

def getStateAndReward(state, action):
    i,j=state
    ai,aj=action
    if i==0 and j ==1:
        return (4,1),10
    if i==0 and j ==3:
        return (2,3),5
    i=i+ai
    j=j+aj
    if i<0 or j<0 or i>4 or j>4:
        return state,-1
    return (i,j),0


#上下左右
ACTIONS=[(-1,0),(1,0),(0,-1),(0,1)]
gamma=0.9




STATES=[(x,y) for x in range(5) for y in range(5)]
newGrid=np.zeros((5,5))
Grid=np.zeros((5,5))
while(True):
    newGrid=np.zeros((5,5))
    for k in range(len(STATES)):
        state=STATES[k]
        
        for j in range(4):
            newstate,reward=getStateAndReward(state,ACTIONS[j])
            newGrid[state]+=0.25*(gamma*Grid[newstate]+reward)
    if(np.sum(np.abs(newGrid-Grid))<1e-4):
        print(newGrid)
        break
    Grid=newGrid




