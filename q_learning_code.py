import numpy as np
gamma=0.8
Q=np.zeros((6,6))
R=np.array(
    [
        [-1,-1,-1,-1,0,-1],
        [- 1,-1,- 1,0,- 1,100],
        [- 1,- 1,- 1,0,- 1,- 1],
        [- 1,0,0,- 1,0,- 1],
        [0,- 1,- 1,0,- 1,100],
        [- 1,0,- 1,- 1,0,100]
    ]
)
def getMaxQ(state):
    return max(Q[state,:])

def QLearning(state):
    curAction=None
    for action in range(6):
        if(R[state][action]==-1):
            Q[state,action]=0
        else:
            curAction=action
            Q[state,action]=R[state][action]+gamma*getMaxQ(curAction)

for i in range(1000):
    for j in range(6):
        QLearning(j)
print(Q)