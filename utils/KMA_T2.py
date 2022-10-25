"""
KM algorithm
Soar Xiao 10/7/22
"""
import random
import numpy as np
import time
import matplotlib.pyplot as plt

class MakeCostMatrix():
    
    def __init__(self):
        self.rs = np.random.randint(0,1000,(N,2))
        self.rd = np.random.randint(300,600,(N,2))
        self.dm = np.zeros((np.shape(self.rs)[0],np.shape(self.rd)[0]))
        
    def calculate(self):
        for i in range(np.shape(self.rs)[0]):
            for j in range(M):
                #self.dm[i][j]=1000-(abs(self.rs[i][0]-self.rd[j][0])+abs(self.rs[i][1]-self.rd[j][1]))  
                #Calculate distance
                self.dm[i][j]=1000-round(np.sqrt((self.rs[i][0]-self.rd[j][0])**2+(self.rs[i][1]-self.rd[j][1])**2))
            for j in range(M,np.shape(self.rd)[0]):
                self.dm[i][j]=0
                
                
def find_path(i, visited_left, visited_right, slack_right):
    visited_left[i] = True
    for j, match_weight in enumerate(graph[i]):
        if visited_right[j]:
            continue
        gap = label_left[i] + label_right[j] - match_weight
        if gap == 0:
            visited_right[j] = True
            if j not in T or find_path(T[j], visited_left, visited_right, slack_right):
                T[j] = i
                S[i] = j
                return True

        else:
            slack_right[j] = min(slack_right[j], gap)
    return False

def KM():
    m = len(graph)
    for i in range(m):
        slack_right = [float('inf') for _ in range(m)]
        while True:
            visited_left = [False for _ in graph]
            visited_right = [False for _ in graph]
            if find_path(i,visited_left,visited_right, slack_right):
                break
            d = float('inf')
            for j, slack in enumerate(slack_right):
                if not visited_right[j] and slack < d:
                    d = slack
            for k in range(m):
                if visited_left[k]:
                    label_left[k] -= d
                if visited_right[k]:
                    label_right[k] += d
    return S, T


if __name__ == '__main__':   
    t0 = time.time()
    N=100
    M=30
    costm=MakeCostMatrix()
    print("rs=\n",costm.rs)
    print("rd=\n",costm.rd)
    costm.calculate()
    #print(costm.dm)
    print("elapsed time: {:.2f}s".format(time.time() - t0))
    graph = (costm.dm).tolist()
    #print(graph)
    label_left, label_right = [max(g) for g in graph], [0 for _ in graph]
    S, T = {}, {}
    visited_left = [False for _ in graph]
    visited_right = [False for _ in graph]
    slack_right = [float('inf') for _ in graph]
    KM()
    print(S)
    print("elapsed time: {:.2f}s".format(time.time() - t0))

    TotDis=0
    for i in range(len(S)):
        if S[i]<M:
            plt.plot([costm.rs[i][0], costm.rd[S[i]][0]],[costm.rs[i][1], costm.rd[S[i]][1]])
            plt.plot([costm.rd[S[i]][0]], [costm.rd[S[i]][1]],'ro')
            TotDis += (1000-costm.dm[i][S[i]])
        plt.plot([costm.rs[i][0]], [costm.rs[i][1]],'bo')
    print("total distance:",TotDis)
    
    plt.show()
    
    
