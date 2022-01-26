import numpy as np
import random
import matplotlib.pyplot as plt 
from tqdm import tqdm
import sys

grid = [['.' for i in range(5)] for j in range(5)]
grid[0][0] = 'R'
grid[0][4] = 'G'
grid[4][0] = 'Y'
grid[4][3] = 'B'

################################################################################ (A)

# state - {taxi position - (x_t,y_t), passenger position - (x_p,y_p), Inside taxi? flag = Y(1)/N(0)/Exit(-1) } - Tuple
# actions - { N,S,E,W,PU,PD }

actions = {0:'N',1:'S',2:'E',3:'W',4:'PU',5:'PD'}

'''
  Feed in the directional deterministic action in this function to obtain the final state.
'''
def nextState(state: tuple, action: int):
  dir = {0:[-1,0],1:[1,0],2:[0,1],3:[0,-1]}
  move = dir[action]
  final_state = [state[0]+move[0],state[1]+move[1]] #of taxi
  if final_state[0] < 0 or final_state[1] <0 or final_state[0]>4 or final_state[1]>4:
    return state
  walls = [
           [0,1,2],[1,1,2],[0,2,3],[1,2,3],
           [3,0,2],[4,0,2],[3,1,3],[4,1,3],
           [3,2,2],[4,2,2],[3,3,3],[4,3,3]
           ]
  if [state[0],state[1],action] in walls:
    return state
  if state[4] == 1:
    return tuple(final_state + final_state + [1])
  else:
    return tuple(final_state) + state[2:]

'''
  Given a Q-state, this functions makes a sample of the values possible of next state and reward with respective probabilities.
'''
################ A(1.b)
def simulator(grid,state:tuple,action:int,destDep:list):
  if action < 4:
    z = np.random.random()
    if z>=0.85 and z<0.9:
      action = (action + 1)%4
    elif z>=0.9 and z<0.95:
      action = (action + 2)%4
    elif z >=0.95 :
      action = (action + 3)%4

  ## If the actions are one of pickup or drop
  if state[4] == 1 and (state[0],state[1]) == (destDep[0],destDep[1]) and action==5:
    state = state[0:4] + (-1,)
    reward = 20
    return state,reward

  if (state[0],state[1]) != (state[2],state[3]) and action in [4,5]:
    return state,-10

  if (state[0],state[1]) == (state[2],state[3]) and action in [4,5]:
    if (action == 4 and state[4] == 0) or (action == 5 and state[4] == 1):
      state = state[0:4]+(1 - state[4],)
    return state,-1
  ## If the action is one of the directions
  return nextState(state,action),-1

def valueIteration(startDep,destDep,startPos,eps,discount = 0.9):
  d = {}
  for i in range(25):
    for j in range(25):
      d[(i//5,i%5,j//5,j%5,0)] = [0,0]
      d[(i//5,i%5,i//5,i%5,1)] = [0,0]
  
  policy = {}
  for state in d:
    policy[state] = -1

  maxNorms = []
  maxnorm = 10*9
  iter = 0

  while (maxnorm>eps*(1-discount)/discount):
    iter += 1
    maxnorm = 0 
    for state in d:
      q_states = []
      for action in actions:
        if action in [4,5]:
          out_state,reward = simulator(grid,state,action,destDep)
          if out_state[4] == -1:
            q_states.append(reward)
          else:
            q_states.append(reward+ discount*d[out_state][0])
        else:
          s = 0
          for i in range(4):
            prob = 0.85 if action == i else 0.05
            next_state = nextState(state,i)
            next_state = tuple(next_state)
            s += prob*(-1+ discount * d[next_state][0])
          q_states.append(s)
      d[state][1] = max(q_states)
      policy[state] = q_states.index(d[state][1])

    for state in d:
      maxnorm = max(maxnorm,abs(d[state][0]-d[state][1]))
      d[state][0] = d[state][1]
      d[state][1] = 0
    #print(iter, maxnorm)
    maxNorms.append(maxnorm)

  return d,len(maxNorms),maxNorms,policy


def policyIteration(startDep,destDep,startPos,eps,discount = 0.9):
  d = {}
  for i in range(25):
    for j in range(25):
      d[(i//5,i%5,j//5,j%5,0)] = [0,0]
      d[(i//5,i%5,i//5,i%5,1)] = [0,0]

  iter = 0
  past_loss = {}
  for state in d:
    past_loss[state] = []

  ## Choose a random initial policy 
  policy = {}
  for state in d:
    policy[state] = np.random.randint(0,6)
  new_policy = {}

  while(new_policy!=policy):
    for state in d :
      d[state] = [0,0]
    iter += 1
    # print("###### Iteration Number "+str(iter))
    if new_policy != {}:
      c = 0
      changes = []
      for x in new_policy:
        if policy[x] != new_policy[x]:
          changes.append([x,policy[x],new_policy[x]])
          c += 1
      # print(c)
      # if c <= 5:
      #   for x in changes:
      #     print(x)
      policy = new_policy
    maxnorm = 10*9
    # Policy Evaluation
    while (maxnorm>eps*(1-discount)/discount):
      maxnorm = 0 
      for state in d:
        r = 0
        action = policy[state]
        if action in [4,5]:
          out_state,reward = simulator(grid,state,action,destDep)
          if out_state[4] == -1:
            r = reward
          else:
            r = reward+ discount*d[out_state][0]
        else:
          s = 0
          for i in range(4):
            prob = 0.85 if action == i else 0.05
            next_state = nextState(state,i)
            s += prob*(-1+ discount * d[next_state][0])
          r = s
        d[state][1] = r
      for state in d:
        maxnorm = max(maxnorm,abs(d[state][0]-d[state][1]))
        d[state][0] = d[state][1]
        d[state][1] = 0
      #print(maxnorm)
    for state in d:
      past_loss[state].append(d[state][0])

  # Policy Improvement
    new_policy = {}
    for state in d:
      new_policy[state] = -1
    #print(actions)
    for state in d:
      q_states = []
      for action in actions:
        if action in [4,5]:
          out_state,reward = simulator(grid,state,action,destDep)
          if out_state[4] == -1:
            q_states.append(reward)
          else:
            q_states.append(reward+ discount*d[out_state][0])
        else:
          s = 0
          for i in range(4):
            prob = 0.85 if action == i else 0.05
            next_state = nextState(state,i)
            s += prob*(-1+ discount * d[next_state][0])
          q_states.append(s)
      # if state == (3, 4, 1, 1, 0):
      #   print(q_states)
      #new_policy[state] = q_states.index(max(q_states))
      max_val = max(q_states)
      for i in range(6):
        if abs(q_states[i]-max_val) < 0.00000001:
          new_policy[state] = i
          break
  return d,past_loss,policy, iter

def policy_loss_per_iteration(policy_loss):
  n = len(list(policy_loss.values())[0])
  l = [0 for i in range(n)]
  b = 0
  for state in policy_loss:
    a = policy_loss[state][-1]
    for i in range(n):
      l[i] = max(l[i],abs(policy_loss[state][i]-a))
  return l


'''
Linear Algebra method for policy evaluation
'''

def linalg(startDep,destDep,startPos,eps,policy,discount = 0.9):
  d = {}
  states = []
  statemap = {}
  for i in range(25):
    for j in range(25):
      d[(i//5,i%5,j//5,j%5,0)] = 0
      states.append((i//5,i%5,j//5,j%5,0))
      statemap[states[-1]] = len(states) - 1


  for i in range(25):
    d[(i//5,i%5,i//5,i%5,1)] = 0
    states.append((i//5,i%5,i//5,i%5,1))
    statemap[states[-1]] = len(states) - 1

  states.append((-1,-1,-1,-1,-1)) # Terminal State
  statemap[states[-1]] = len(states) - 1

  R = []
  for s in states[:-1]:
    action = policy[s]
    _,reward = simulator(grid,s,action,destDep)
    R.append(reward)
  R.append(0)

  R = np.array(R,dtype='float')

  T = np.zeros((651,651),dtype='float')
  for i in range(650):
      action = policy[states[i]]
      if action in [4,5]:
        next_state,reward = simulator(grid,states[i],action,destDep)
        if next_state[4]==-1:
          T[i][650] = 1
        else:
          T[i][statemap[next_state]] = 1
        continue
      for j in range(4):
        if action==j:
          T[i][statemap[nextState(states[i],j)]] += 0.85
        else:
          T[i][statemap[nextState(states[i],j)]] += 0.05

  for i in range(650):
    if np.sum(T[i])!=1:
      print("Error")

  # V = T*(R + d*V)
  # (I - d*T)V = R
  # AV = B where A = I - d*T and B = T*R

  A = np.identity(651,dtype='float')-discount*T
  # B = (T.dot(R)).reshape((651,1))

  # v = np.linalg.solve(A,B)
  v = np.linalg.inv(A).dot(R)

  # print(v[statemap[(4,3,4,3,1)]])
  # print(statemap[(4,3,4,3,1)])
  # for i in range(651):
  #   if T[648][i]>0:
  #     print(states[i],R[i])


  i = 0
  for s in states[:-1]:
    d[s] = v[i]
    i += 1

  return d

def policyIteration_linalg(startDep,destDep,startPos,eps,discount = 0.9):
  d = {}
  for i in range(25):
    for j in range(25):
      d[(i//5,i%5,j//5,j%5,0)] = 0
  for i in range(25):
    d[(i//5,i%5,i//5,i%5,1)] = 0

  iter = 0

  past_loss = {}
  for state in d:
    past_loss[state] = []

  ## Choose a random initial policy 
  policy = {}
  for state in d:
    policy[state] = random.randint(0,5)

  new_policy = {}

  while(new_policy!=policy):
    
    for state in d :
      d[state] = 0

    # print("Iteration ",iter+1)
    iter += 1

    if new_policy != {}:
      policy = new_policy

    # Policy Evaluation
    d = linalg(startDep,destDep,startPos,eps,policy,discount)

    for state in d:
      past_loss[state].append(d[state])

    # Policy Improvement
    new_policy = {}
    for state in d:
      new_policy[state] = -1
    #print(actions)
    for state in d:
      q_states = []
      for action in actions:
        if action in [4,5]:
          out_state,reward = simulator(grid,state,action,destDep)
          if out_state[4] == -1:
            q_states.append(reward)
          else:
            q_states.append(reward+ discount*d[out_state])
        else:
          s = 0
          for i in range(4):
            prob = 0.85 if action == i else 0.05
            next_state = nextState(state,i)
            s += prob*(-1 + discount * d[next_state])
          q_states.append(s)

      mx = max(q_states)
      # if state==(4,4,4,4,1):
      #   print(q_states)
      #print(q_states)
      for i in range(6):
        if abs(q_states[i]-mx) < 0.00000001:
          new_policy[state] = i
          break

      # new_policy[state] = np.argmax(np.array(q_states))



    # count = 0
    # for state in d:
    #   if policy[state]!=new_policy[state]:
    #     count += 1
    #     #print(state)
    # print(policy[(4, 4, 4, 4, 1)],new_policy[(4, 4, 4, 4, 1)])
    # print("Count ",count)

  return d,past_loss,policy, iter

def plot_policy(intervals,num_test,algo):
  l = []
  for i in tqdm(range(0,2001,intervals)):
    Q = algo([0,4],[4,0],[0,0],i,0.25,discount = 0.99)
    policy = policy_from_utility(Q)
    reward = 0
    for j in range(num_test):
      state = random_state_generator([4,0])
      numStep = 0
      td = 1 
      while(numStep<500):
        numStep += 1
        state, in_reward = simulator(grid,state,policy[state],[4,0])
        reward += td*in_reward
        td *= 0.99
        if state[4] == -1:
          break
    l.append(reward/num_test)
  return l

def plot_policy_b5(intervals,num_test):
  l = []
  for i in tqdm(range(0,10000,intervals)):
    Q = qLearning_b5([0,5],[4,6],[2,8],i,0.25,0.99,0.2)
    policy = policy_from_utility(Q)
    reward = 0
    for j in range(num_test):
      state = random_state_generator_b5([4,6])
      numStep = 0
      td = 1 
      while(numStep<500):
        numStep += 1
        state, in_reward = simulator_b5(grid,state,policy[state],[4,6])
        reward += td*in_reward
        td *= 0.99
        if state[4] == -1:
          break
    l.append(reward/num_test)
  plt.plot([i for i in range(0,10000,intervals)],l)
  plt.xlabel("Number of Training Episodes")
  plt.ylabel("Average Discounted Reward")
  plt.savefig('PartB5.png')
  plt.show()
  return l

def policy_from_utility(Q):
  policy = {}
  for i in Q:
    state = i[0]
    if state in policy:
      pass
    q_states = []
    for action in actions :
      q_states.append(Q[(state,action)])
    policy[state] = q_states.index(max(q_states))
  return policy

def simulate_policy(initial_state,num_steps,policy,des_dept):
  state_ = initial_state
  r = 0 
  dis = 1
  for i in range(num_steps):
    action = policy.get(state_)
    print(i+1,' ' if i <9 else '',state_,actions[action] if action != None else None)
    if state_[4] == -1:
      break
    state_, reward = simulator(grid,state_,action,des_dept)
    r += reward*dis
    dis*= 0.99
  return r

def random_state_generator(destpos):
  person_pos = [[0,0],[0,4],[4,0],[4,3]]
  person_pos.remove(destpos)
  n = random.randint(0,2)
  p_pos = person_pos[n]
  position =  [np.random.randint(0,5) for _ in range(2)]
  return tuple(position) + tuple(p_pos) + (0,)

def qLearning_a(startDep,destDep,startPos,numeps,alpha,discount = 0.9,epsilon = 0.1):
  q = {}
  for i in range(25):
    for j in range(25):
      for action in actions:
        q[((i//5,i%5,i//5,i%5,1),action)] = 0
        q[((i//5,i%5,j//5,j%5,0),action)] = 0
  for ii in range(numeps):
    state = random_state_generator(destDep)
    numSteps = 0
    #print(state)
    while(numSteps<500):
      numSteps += 1
      z = np.random.random()
      if z < epsilon:
        action = np.random.randint(0,6)
      else:
        q_states = []
        for i in actions:
          q_states.append(q[(state,i)])
        action = q_states.index(max(q_states))

      next_state,imm_reward = simulator(grid,state,action,destDep)

      if next_state[4]==-1:
        q[(state,action)] = (1-alpha)*q[(state,action)] + alpha*(imm_reward)
        break

      q[(state,action)] = (1-alpha)*q[(state,action)] + alpha*(imm_reward + discount*max([q[(next_state,_)] for _ in actions]))

      state = next_state

  return q


def qLearning_b(startDep,destDep,startPos,numeps,alpha,discount = 0.9):
  q = {}
  for i in range(25):
    for j in range(25):
      for action in actions:
        q[((i//5,i%5,i//5,i%5,1),action)] = 0
        q[((i//5,i%5,j//5,j%5,0),action)] = 0
  #print(len(q))
  totalSteps = 0 
  for ii in range(numeps):
    state = random_state_generator(destDep)
    numSteps = 0
    epsilon = 0.1
    #print(state)
    while(numSteps<500):
      numSteps += 1
      totalSteps += 1
      epsilon = epsilon/totalSteps
      z = np.random.random()
      if z < epsilon:
        action = np.random.randint(0,6)
      else:
        q_states = []
        for i in actions:
          q_states.append(q[(state,i)])
        action = q_states.index(max(q_states))

      next_state,imm_reward = simulator(grid,state,action,destDep)

      if next_state[4]==-1:
        q[(state,action)] = (1-alpha)*q[(state,action)] + alpha*(imm_reward)
        break

      q[(state,action)] = (1-alpha)*q[(state,action)] + alpha*(imm_reward + discount*max([q[(next_state,i)] for i in actions]))

      state = next_state

  return q

def SARSA_c(startDep,destDep,startPos,numeps,alpha,discount = 0.9):
  q = {}
  for i in range(25):
    for j in range(25):
      for action in actions:
        q[((i//5,i%5,i//5,i%5,1),action)] = 0
        q[((i//5,i%5,j//5,j%5,0),action)] = 0
  epsilon = 0.1

  for ii in range(numeps):
    numSteps = 0
    ## Random state
    state = random_state_generator(destDep)
    ## Random action
    z = np.random.random()
    if z < epsilon:
        action = np.random.randint(0,6)
    else:
      q_states = []
      for i in actions:
        q_states.append(q[(state,i)])
      action = q_states.index(max(q_states))

    while(numSteps<500):
      numSteps += 1
      z = np.random.random()
      next_state,imm_reward = simulator(grid,state,action,destDep)
    
      if next_state[4]==-1:
        q[(state,action)] = (1-alpha)*q[(state,action)] + alpha*(imm_reward)
        break
      
      if z < epsilon:
        next_action = np.random.randint(0,6)
      else:
        q_states = []
        for i in actions:
          q_states.append(q[(next_state,i)])
        next_action = q_states.index(max(q_states))

      q[(state,action)] = (1-alpha)*q[(state,action)] + alpha*(imm_reward + discount*q[(next_state,next_action)])
      state = next_state
      action = next_action
  return q

def SARSA_d(startDep,destDep,startPos,numeps,alpha,discount = 0.9):
  q = {}
  for i in range(25):
    for j in range(25):
      for action in actions:
        q[((i//5,i%5,i//5,i%5,1),action)] = 0
        q[((i//5,i%5,j//5,j%5,0),action)] = 0
  totalSteps = 0 
  for ii in range(numeps):
    numSteps = 0
    epsilon = 0.1
    ## Random state
    state = random_state_generator(destDep)

    ## Random action
    z = np.random.random()
    if z < epsilon:
        action = np.random.randint(0,6)
    else:
      q_states = []
      for i in actions:
        q_states.append(q[(state,i)])
      action = q_states.index(max(q_states))

    while(numSteps<500):
      numSteps += 1
      totalSteps += 1
      epsilon = epsilon/totalSteps
      z = np.random.random()
      next_state,imm_reward = simulator(grid,state,action,destDep)
    
      if next_state[4]==-1:
        q[(state,action)] = (1-alpha)*q[(state,action)] + alpha*(imm_reward)
        break
      
      if z < epsilon:
        next_action = np.random.randint(0,6)
      else:
        q_states = []
        for i in actions:
          q_states.append(q[(next_state,i)])
        next_action = q_states.index(max(q_states))

      q[(state,action)] = (1-alpha)*q[(state,action)] + alpha*(imm_reward + discount*q[(next_state,next_action)])
      state = next_state
      action = next_action
  return q

def plot_policy_b4(intervals,num_test,epsilon,alpha):
  l = []
  for i in tqdm(range(0,2001,intervals)):
    Q = qLearning_a([0,4],[4,0],[0,0],i,alpha,0.99,epsilon)
    policy = policy_from_utility(Q)
    reward = 0
    for j in range(num_test):
      state = random_state_generator([4,0])
      numStep = 0
      td = 1 
      while(numStep<500):
        numStep += 1
        state, in_reward = simulator(grid,state,policy[state],[4,0])
        reward += td*in_reward
        td *= 0.99
        if state[4] == -1:
          break
    l.append(reward/num_test)
  return l

"""PartB5"""

'''
  Feed in the directional deterministic action in this function to obtain the final state.
'''
def nextState_b5(state: tuple, action: int):
  dir = {0:[-1,0],1:[1,0],2:[0,1],3:[0,-1]}
  move = dir[action]
  final_state = [state[0]+move[0],state[1]+move[1]] #of taxi
  if final_state[0] < 0 or final_state[1] <0 or final_state[0]>9 or final_state[1]>9:
    return state
  walls = [
           [6,0,2],[6,1,3],[7,0,2],[7,1,3],[8,0,2],[8,1,3],[9,0,2],[9,1,3],
           [0,2,2],[0,3,3],[1,2,2],[1,3,3],[2,2,2],[2,3,3],[3,2,2],[3,3,3],
           [6,3,2],[6,4,3],[7,3,2],[7,4,3],[8,3,2],[8,4,3],[9,3,2],[9,4,3],
           [2,5,2],[2,6,3],[3,5,2],[3,6,3],[4,5,2],[4,6,3],[5,5,2],[5,6,3],
           [6,7,2],[6,8,3],[7,7,2],[7,8,3],[8,7,2],[8,8,3],[9,7,2],[9,8,3],
           [0,7,2],[0,8,3],[1,7,2],[1,8,3],[2,7,2],[2,8,3],[3,7,2],[3,8,3],
           ]
  if [state[0],state[1],action] in walls:
    return state
  if state[4] == 1:
    return tuple(final_state + final_state + [1])
  else:
    return tuple(final_state) + state[2:]

def random_state_generator_b5(destDep):
  person_pos = [[0,0],[0,5],[3,3],[9,4],[4,6],[9,9],[0,8],[8,0]]
  person_pos.remove(destDep)
  n = random.randint(0,2)
  p_pos = person_pos[n]
  position =  [np.random.randint(0,10) for _ in range(2)]
  return tuple(position) + tuple(p_pos) + (0,)

'''
  Given a Q-state, this functions makes a sample of the values possible of next state and reward with respective probabilities.
'''
def simulator_b5(grid,state:tuple,action:int,destDep:list):
  if action < 4:
    z = np.random.random()
    if z>=0.85 and z<0.9:
      action = (action + 1)%4
    elif z>=0.9 and z<0.95:
      action = (action + 2)%4
    elif z >=0.95 :
      action = (action + 3)%4

  ## If the actions are one of pickup or drop
  if state[4] == 1 and (state[0],state[1]) == (destDep[0],destDep[1]) and action==5:
    state = state[0:4] + (-1,)
    reward = 20
    return state,reward

  if (state[0],state[1]) != (state[2],state[3]) and action in [4,5]:
    return state,-10

  if (state[0],state[1]) == (state[2],state[3]) and action in [4,5]:
    if (action == 4 and state[4] == 0) or (action == 5 and state[4] == 1):
      state = state[0:4]+(1 - state[4],)
    return state,-1
  ## If the action is one of the directions
  return nextState_b5(state,action),-1

#################################################################################(B.1.a)
def qLearning_b5(startDep,destDep,startPos,numeps,alpha,discount = 0.9,epsilon = 0.1):
  q = {}
  for i in range(100):
    for j in range(100):
      for action in actions:
        q[((i//10,i%10,i//10,i%10,1),action)] = 0
        q[((i//10,i%10,j//10,j%10,0),action)] = 0
  total_steps = 0 
  for ii in range(numeps):
    state = random_state_generator_b5(destDep)
    numSteps = 0
    #print(state)
    while(numSteps<500):
      numSteps += 1
      total_steps += 1
      epsilon = epsilon/total_steps
      z = np.random.random()
      if z < epsilon:
        action = np.random.randint(0,6)
      else:
        q_states = []
        for i in actions:
          q_states.append(q[(state,i)])
        action = q_states.index(max(q_states))
      next_state,imm_reward = simulator_b5(grid,state,action,destDep)
      if next_state[4]==-1:
        q[(state,action)] = (1-alpha)*q[(state,action)] + alpha*(imm_reward)
        break
      q[(state,action)] = (1-alpha)*q[(state,action)] + alpha*(imm_reward + discount*max([q[(next_state,i)] for i in actions]))
      state = next_state
  return q


def show(policy):
  print("Case when the passenger is inside the taxi")
  out = []
  for i in range(25):
    out.append((i//5,i%5,actions[policy[(i//5,i%5,i//5,i%5,1)]]))
  l = []
  for i in range(30):
    if i%5 == 0:
      l.append("|")
    else:
      l.append(" ")

  j = []
  for i in range(26):
    j.append("-")
  l.append("\n")
  j.append("\n")
  s = []
  for k in range(10):
    if k % 2 == 0:
      s = s + j
    else:
      s = s + l
  wall = [37,95,206,264,216,274]
  for _ in wall:
    s[_] = "┃"
  s = s + j
  #show_dir = {"N":"⬆", "S":"⬇", "E":"➡ ","W":" ⬅","PD":""}
  show_dir = {"N":"↑", "S":"↓", "E":"→","W":"←","PD":" ","PU":" ","PD":" "}
  for j in out:
    s[29+58*j[0]+5*j[1]] = show_dir[j[2]]
  print("".join(s)) 
  print("\n")
  startDepts = [(0,0),(4,0),(0,4),(4,3)]
  for ini_Dept in startDepts:
    print("Case when the passenger is waiting at ", ini_Dept)
    out = []
    for i in range(25):
      out.append((i//5,i%5,actions[policy[(i//5,i%5,ini_Dept[0],ini_Dept[1],0)]]))
    l = []
    for i in range(30):
      if i%5 == 0:
        l.append("|")
      else:
        l.append(" ")

    j = []
    for i in range(26):
      j.append("-")
    l.append("\n")
    j.append("\n")
    s = []
    for k in range(10):
      if k % 2 == 0:
        s = s + j
      else:
        s = s + l
    wall = [37,95,206,264,216,274]
    for _ in wall:
      s[_] = "┃"
    s = s + j
    #show_dir = {"N":"⬆", "S":"⬇", "E":"➡ ","W":" ⬅","PD":""}
    show_dir = {"N":"↑", "S":"↓", "E":"→","W":"←","PD":" ","PU":" ","PD":" "}
    for j in out:
      s[29+58*j[0]+5*j[1]] = show_dir[j[2]]
    print("".join(s)) 

########################################################### EXECUTION CODE

if sys.argv[1]=='A':
  if sys.argv[2]=='2':
    if sys.argv[3]=='a':
      d,numIter,maxNorms,policy = valueIteration([0,0],[4,3],[2,2],0.001,0.9)
      print("Number of iterations taken to converge =", numIter)
      show(policy)
    
    elif sys.argv[3]=='b':
      for discount in [0.01,0.1,0.5,0.8,0.99]:
        d,numIter,maxNorms,policy = valueIteration([0,0],[4,3],[2,2],0.001,discount)
        print("For discount = ",discount,": ",numIter," iterations taken to converge")
        plt.plot(maxNorms,label = str(discount))
        plt.legend(loc="upper right")
        plt.xlabel("Iteration Index")
        plt.ylabel("max-norm distance")
      plt.savefig("PartA2b.png")
      plt.show()

    elif sys.argv[3]=='c':
      state_ = (2,2,0,0,0)
      policy1 = valueIteration([0,0],[4,3],[2,2],0.001,0.1)[3]
      policy2 = valueIteration([0,0],[4,3],[2,2],0.001,0.99)[3]
      print("For discount factor gamma = 0.1")
      for i in range(20):
        action = policy1.get(state_)
        print(i+1,' ' if i <9 else '',state_,actions[action] if action != None else None)
        if state_[4] == -1:
          break
        state_, reward = simulator(grid,state_,action,[4,3])
      print('\n')
      state_ = (2,2,0,0,0)
      print("For discount factor gamma = 0.99")
      for i in range(20):
        action = policy2.get(state_)
        print(i+1,' ' if i <9 else '',state_,actions[action] if action != None else None)
        if state_[4] == -1:
          break
        state_, reward = simulator(grid,state_,action,[4,3])  

  elif sys.argv[2]=='3':
    if sys.argv[3]=='i':
      #Iterative
      d1,policy_loss1,_,iter1 = policyIteration([0,0],[4,3],[2,2],0.01,0.01)
      print("For discount = ",0.01,": ",iter1," iterations taken to converge")
      d2,policy_loss2,_,iter2 = policyIteration([0,0],[4,3],[2,2],0.01,0.1)
      print("For discount = ",0.1,": ",iter2," iterations taken to converge")
      d3,policy_loss3,_,iter3 = policyIteration([0,0],[4,3],[2,2],0.01,0.5)
      print("For discount = ",0.5,": ",iter3," iterations taken to converge")
      d4,policy_loss4,_,iter4 = policyIteration([0,0],[4,3],[2,2],0.01,0.8)
      print("For discount = ",0.8,": ",iter4," iterations taken to converge")
      d5,policy_loss5,_,iter5 = policyIteration([0,0],[4,3],[2,2],0.01,0.99)
      print("For discount = ",0.99,": ",iter5," iterations taken to converge")

      #d,policy_loss,_,_= policyIteration([0,4],[4,3],[2,2],0.01,0.9)

      plt.plot(policy_loss_per_iteration(policy_loss1),label = str(0.01))
      plt.plot(policy_loss_per_iteration(policy_loss2),label = str(0.1))
      plt.plot(policy_loss_per_iteration(policy_loss3),label = str(0.5))
      plt.plot(policy_loss_per_iteration(policy_loss4),label = str(0.8))
      plt.plot(policy_loss_per_iteration(policy_loss5),label = str(0.99))
      plt.legend(loc="upper right")
      plt.xlabel("Iteration Number")
      plt.ylabel("Policy Loss")
      plt.savefig("PartA3i.png")
      plt.show()
    else:
      #Linear Algebra
      d1,policy_loss1,_,iter1= policyIteration_linalg([0,0],[4,3],[2,2],0.01,0.01)
      print("For discount = ",0.01,": ",iter1," iterations taken to converge")
      d2,policy_loss2,_,iter2= policyIteration_linalg([0,0],[4,3],[2,2],0.01,0.1)
      print("For discount = ",0.1,": ",iter2," iterations taken to converge")
      d3,policy_loss3,_,iter3= policyIteration_linalg([0,0],[4,3],[2,2],0.01,0.5)
      print("For discount = ",0.5,": ",iter3," iterations taken to converge")
      d4,policy_loss4,_,iter4= policyIteration_linalg([0,0],[4,3],[2,2],0.01,0.8)
      print("For discount = ",0.8,": ",iter4," iterations taken to converge")
      d5,policy_loss5,_,iter5= policyIteration_linalg([0,0],[4,3],[2,2],0.01,0.99)
      print("For discount = ",0.99,": ",iter5," iterations taken to converge")

      d,policy_loss,_,_= policyIteration([0,0],[4,3],[2,2],0.01,0.9)

      plt.plot(policy_loss_per_iteration(policy_loss1),label = str(0.01))
      plt.plot(policy_loss_per_iteration(policy_loss2),label = str(0.1))
      plt.plot(policy_loss_per_iteration(policy_loss3),label = str(0.5))
      plt.plot(policy_loss_per_iteration(policy_loss4),label = str(0.8))
      plt.plot(policy_loss_per_iteration(policy_loss5),label = str(0.99))
      plt.legend(loc="upper right")
      plt.xlabel("Iteration Number")
      plt.ylabel("Policy Loss")
      plt.savefig("PartA3l.png")
      plt.show()


elif sys.argv[1]=='B':
  if sys.argv[2]=='1':
    if sys.argv[3]=='a':
      Q = qLearning_a([0,4],[4,0],[0,0],2000,0.25,discount = 0.99)
      policy1 = policy_from_utility(Q)
      show(policy1)
      simulate_policy((0,0,0,4,0),2000,policy1,[4,0])
      l = plot_policy(20,10,qLearning_a)

      plt.plot([i for i in range(0,2001,20)][6:],l[6:])
      plt.xlabel("Number of Training Episodes")
      plt.ylabel("Average Discounted Reward")
      plt.savefig("PartB1a.png")
      plt.show()
    elif sys.argv[3]=='b':
      Q = qLearning_b([0,4],[4,0],[0,0],2000,0.25,discount = 0.99)
      policy2 = policy_from_utility(Q)
      show(policy2)
      simulate_policy((0,0,0,4,0),100,policy2,[4,0])
      l_b = plot_policy(20,10,qLearning_b)

      plt.plot([i for i in range(0,2001,20)][6:],l_b[6:])
      plt.xlabel("Number of Training Episodes")
      plt.ylabel("Average Discounted Reward")
      plt.savefig("PartB1b.png")
      plt.show()

    elif sys.argv[3]=='c':
      Q = SARSA_c([0,4],[4,0],[0,0],2000,0.25,discount = 0.99)
      policy3 = policy_from_utility(Q)
      show(policy3)
      simulate_policy((0,0,0,4,0),100,policy3,[4,0])
      l_c = plot_policy(20,10, SARSA_c)

      plt.plot([i for i in range(0,2001,20)][6:],l_c[6:])
      plt.xlabel("Number of Training Episodes")
      plt.ylabel("Average Discounted Reward")
      plt.savefig("PartB1c.png")
      plt.show()

    elif sys.argv[3]=='d':
      Q = SARSA_d([0,4],[4,0],[0,0],2000,0.25,discount = 0.99)
      policy4 = policy_from_utility(Q)
      show(policy4)
      simulate_policy((0,0,0,4,0),100,policy4,[4,0])
      l_d = plot_policy(20,10, SARSA_d)

      plt.plot([i for i in range(0,2001,20)][6:],l_d[6:])
      plt.xlabel("Number of Training Episodes")
      plt.ylabel("Average Discounted Reward")
      plt.savefig("PartB1d.png")
      plt.show()




  elif sys.argv[2]=='2':
    l5 = plot_policy(100,100,qLearning_a)
    l6 = plot_policy(100,100,qLearning_b)
    l7 = plot_policy(100,100,SARSA_c)
    l8 = plot_policy(100,100,SARSA_d)
    plt.figure(figsize=(10,8))
    b = [i for i in range(0,2001,100)][2:]
    plt.plot(b,l5[2:],label = "Q Learning")
    plt.plot(b,l6[2:],label = "Q Learning Adaptive")
    plt.plot(b,l7[2:],label = "SARSA")
    plt.plot(b,l8[2:],label = "SARSA Adaptive")
    plt.legend(loc="lower right")
    plt.savefig("PartB2.png")
    plt.show()
    print("Q Learning",sum(l5[-3:-1])/3)
    print("Q Learning Adaptive",sum(l6[-3:-1])/3)
    print("SARSA",sum(l7[-3:-1])/3)
    print("SARSA Adaptive",sum(l8[-3:-1])/3)

  
  elif sys.argv[2]=='3':
    Q = qLearning_b([0,4],[4,0],[0,0],10000,0.25,discount = 0.99)
    dest = [4,0]
    policy_b3 = policy_from_utility(Q)
    possible_pos = [[0,0],[0,4],[4,3]]
    print("Problem 1")
    print("Initial Taxi Depot ",(0,0))
    print("Initial Person Depot ",(0,4))
    print("Reward ", simulate_policy((0,0,0,4,0),2000,policy_b3,[4,0]))
    print("Problem 2")
    print("Initial Taxi Depot ",(0,4))
    print("Initial Person Depot ",(4,3))
    print("Reward ", simulate_policy((0,4,4,3,0),2000,policy_b3,[4,0]))
    print("Problem 3")
    print("Initial Taxi Depot ",(4,3))
    print("Initial Person Depot ",(0,4))
    print("Reward ", simulate_policy((4,3,0,4,0),2000,policy_b3,[4,0]))
    print("Problem 4")
    print("Initial Taxi Depot ",(4,3))
    print("Initial Person Depot ",(0,0))
    print("Reward ", simulate_policy((4,3,0,0,0),2000,policy_b3,[4,0]))
    print("Problem 5")
    print("Initial Taxi Depot ",(0,4))
    print("Initial Person Depot ",(0,0))
    print("Reward ", simulate_policy((0,4,0,0,0),2000,policy_b3,[4,0]))
    print("\n")
    show(policy_b3)

  elif sys.argv[2]=='4a':
    l_e_0 = plot_policy_b4(100,10,0,0.1)
    l_e_005 = plot_policy_b4(100,10,0.05,0.1)
    l_e_01 = plot_policy_b4(100,10,0.1,0.1)
    l_e_05 = plot_policy_b4(100,10,0.5,0.1)
    l_e_09 = plot_policy_b4(100,10,0.9,0.1)

    plt.figure(figsize=(10,8))
    b = [i for i in range(0,2001,100)][3:]
    plt.plot(b,l_e_0[3:],label = "epsilon = 0")
    plt.plot(b,l_e_005[3:],label = "epsilon = 0.05")
    plt.plot(b,l_e_01[3:],label = "epsilon = 0.1")
    plt.plot(b,l_e_05[3:],label = "epsilon = 0.5")
    plt.plot(b,l_e_09[3:],label = "epsilon = 0.9")
    plt.legend(loc="lower right")
    plt.xlabel("Number of Training Episodes")
    plt.ylabel("Average Discounted Reward")
    plt.savefig("PartB4_epsilon.png")
  
  elif sys.argv[2]=='4b':
    l_a_01 = plot_policy_b4(100,10,0.1,0.1)
    l_a_02 = plot_policy_b4(100,10,0.1,0.2)
    l_a_03 = plot_policy_b4(100,10,0.1,0.3)
    l_a_04 = plot_policy_b4(100,10,0.1,0.4)
    l_a_05 = plot_policy_b4(100,10,0.1,0.5)
    b = [i for i in range(0,2001,100)][3:]
    plt.figure(figsize=(10,8))
    plt.plot(b,l_a_01[3:],label = "alpha = 0.1")
    plt.plot(b,l_a_02[3:],label = "alpha = 0.2")
    plt.plot(b,l_a_03[3:],label = "alpha = 0.3")
    plt.plot(b,l_a_04[3:],label = "alpha = 0.4")
    plt.plot(b,l_a_05[3:],label = "alpha = 0.5")
    plt.legend(loc="lower right")
    plt.xlabel("Number of Training Episodes")
    plt.ylabel("Average Discounted Reward")
    plt.savefig("PartB4_alpha.png")

  elif sys.argv[2]=='5':
    dst_depots = [[0,0],[0,5],[3,3],[9,4],[4,6],[9,9],[0,8],[8,0]]
    avg_reward = []
    for depot in dst_depots:
      Q = qLearning_b5([8,0],depot,[5,2],10000,0.5,0.99,0.1)
      policy = policy_from_utility(Q)
      state = random_state_generator_b5(depot)
      print("Destination Depot",depot,"Start Depot",state[2:4], "Initial Taxi Location",state[0:2])
      numStep = 0
      reward = 0
      td = 1
      while(numStep<500):
        numStep += 1
        state, in_reward = simulator_b5(grid,state,policy[state],depot)
        reward += td*in_reward
        td *= 0.99
        if state[4] == -1:
          break
      print("Reward",reward)  
      avg_reward.append(reward)
    print("Average Reward Over 8 Instances is ", sum(avg_reward)/10)
    plot_policy_b5(250,200)
    # Q = qLearning_b5([8,0],depot,[5,2],100000,0.25,0.99,0.1)
    # policy = policy_from_utility(Q)
    # simulate_policy(random_state_generator_b5(depot),200000,policy,depot)






