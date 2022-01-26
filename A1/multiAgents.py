# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import math
from game import Agent
from collections import deque as queue

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #More score diff better move
        prevScore = currentGameState.getScore()
        newScore = successorGameState.getScore()
        scoreDiff = newScore - prevScore

        newNumFood = successorGameState.getNumFood()
        foodList = newFood.asList()
        if foodList:
            closeFoodDist = min([manhattanDistance(newPos, food) for food in foodList])
        else:
            closeFoodDist = 0
        capsulePos = successorGameState.getCapsules()
        if capsulePos:
            capsuledist = manhattanDistance(newPos,capsulePos[0])
        else:
            capsuledist = 0 
        
        ghostDist = manhattanDistance(newPos, newGhostStates[0].getPosition()) 

        smallScareTime = newScaredTimes[0]
        if smallScareTime != 0:
            ## Means that ghosts are scared and we should chase them to get 200 points
            ghostDist = -ghostDist*2
        correction = 0 
        if ghostDist < 2 and smallScareTime ==0:
            correction = -100
        if action == 'Stop':
            correction = -20
        return 15/(closeFoodDist+1) + ghostDist/3 + scoreDiff + correction + 30/(capsuledist+1)

## Output using the above mentioned code gives the following Output
'''
Pacman emerges victorious! Score: 1439
Pacman emerges victorious! Score: 1434
Pacman emerges victorious! Score: 1255
Pacman emerges victorious! Score: 1443
Pacman emerges victorious! Score: 1293
Pacman emerges victorious! Score: 1446
Pacman emerges victorious! Score: 1253
Pacman emerges victorious! Score: 1184
Pacman emerges victorious! Score: 1428
Pacman emerges victorious! Score: 1260
Average Score: 1343.5
Scores:        1439.0, 1434.0, 1255.0, 1443.0, 1293.0, 1446.0, 1253.0, 1184.0, 1428.0, 1260.0
Win Rate:      10/10 (1.00)
Record:        Win, Win, Win, Win, Win, Win, Win, Win, Win, Win
*** PASS: test_cases/test_reflex/grade-agent.test
'''

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        answer = self.layer(gameState,0,0)
        bestmove = answer[1]
        bestscore = answer[0]
        #print(bestscore)
        return bestmove
    ## This function returns the minmax value and bestaction at the gamestate.
    ## Agentindex is used to identify whether its a pacman or a ghost.
    ## depth represents the depth of that node of the tree.
    def layer(self,gamestate, agentindex, depth):
        numAgents = gamestate.getNumAgents()
        childIndex = (agentindex+1)%numAgents
        if agentindex == numAgents - 1:
            childDepth = depth + 1
        else:
            childDepth = depth
        if depth == self.depth or gamestate.isWin() or gamestate.isLose():
            return self.evaluationFunction(gamestate),None
        if agentindex == 0:
            act = None 
            opt = -math.inf
            for action in gamestate.getLegalActions(agentindex):
                value = self.layer(gamestate.generateSuccessor(agentindex,action),childIndex,childDepth)[0]
                if value > opt:
                    opt = value
                    act = action
            return opt,act
        else:
            act = None 
            opt = math.inf
            for action in gamestate.getLegalActions(agentindex):
                value = self.layer(gamestate.generateSuccessor(agentindex,action),childIndex,childDepth)[0]
                if value < opt:
                    opt = value
                    act = action
            return opt,act            

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        a = - math.inf
        b = math.inf
        answer = self.layer(gameState,0,0,a,b)
        bestmove = answer[1]
        bestscore = answer[0]
        #print("cost is", bestscore)
        return bestmove
    ## This function returns the minmax value and bestaction at the gamestate.
    ## Agentindex is used to identify whether its a pacman or a ghost.
    ## depth represents the depth of that node of the tree.
    ## a => Max nodes best path to the root
    ## b => Min nodes best path to the root
    def layer(self,gamestate, agentindex, depth,a,b):
        numAgents = gamestate.getNumAgents()
        childIndex = (agentindex+1)%numAgents
        if agentindex == numAgents - 1:
            childDepth = depth + 1
        else:
            childDepth = depth
        if depth == self.depth or gamestate.isWin() or gamestate.isLose():
            return self.evaluationFunction(gamestate),None
        if agentindex == 0:
            act = None 
            opt = -math.inf
            for action in gamestate.getLegalActions(agentindex):
                value = self.layer(gamestate.generateSuccessor(agentindex,action),childIndex,childDepth,a,b)[0]
                if value > opt:
                    opt = value
                    act = action
                if opt > b:
                    #print("pruning at",opt,a,b)
                    return opt,act
                a = max(a,opt)
            return opt,act
        else:
            act = None 
            opt = math.inf
            for action in gamestate.getLegalActions(agentindex):
                value = self.layer(gamestate.generateSuccessor(agentindex,action),childIndex,childDepth,a,b)[0]
                if value < opt:
                    opt = value
                    act = action
                if opt < a:
                    #print("pruning at ",opt,a,b)
                    return opt,act
                b = min(b,opt)
            return opt,act  

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        answer = self.layer(gameState,0,0)
        bestmove = answer[1]
        bestscore = answer[0]
        #print("cost is", bestscore)
        return bestmove
    ## This function returns the minmax value and bestaction at the gamestate.
    ## Agentindex is used to identify whether its a pacman or a ghost.
    ## depth represents the depth of that node of the tree.
    def layer(self,gamestate, agentindex, depth):
        numAgents = gamestate.getNumAgents()
        childIndex = (agentindex+1)%numAgents
        if agentindex == numAgents - 1:
            childDepth = depth + 1
        else:
            childDepth = depth
        if depth == self.depth or gamestate.isWin() or gamestate.isLose():
            return self.evaluationFunction(gamestate),None
        if agentindex == 0:
            act = None 
            opt = -math.inf
            for action in gamestate.getLegalActions(agentindex):
                value = self.layer(gamestate.generateSuccessor(agentindex,action),childIndex,childDepth)[0]
                if value > opt:
                    opt = value
                    act = action
            return opt,act
        else:
            actionsList = gamestate.getLegalActions(agentindex)
            numActions = len(actionsList)
            ans = 0 
            for action in actionsList:
                ans += self.layer(gamestate.generateSuccessor(agentindex,action),childIndex,childDepth)[0]
            ans = ans/numActions
            return ans,None

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: We perfomed BFS with pacman as the source and calculated the evaluation score of the state
    using distances from food, enemy and capsules. 
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    INF = 100000000000000000000

    pacman_pos = currentGameState.getPacmanPosition()
    ghost_pos = currentGameState.getGhostPositions()
    #print(ghost_pos)
    num_ghosts = len(ghost_pos)
    ghost_states = currentGameState.getGhostStates()
    scared_times = []
    for ghost_state in ghost_states:
        scared_times.append(ghost_state.scaredTimer)
    capsules = currentGameState.getCapsules()
    num_capsules = len(capsules)
    food_list = currentGameState.getFood().asList()
    num_food = currentGameState.getNumFood()
    score = currentGameState.getScore()

    value = 0
    value += 1000000*score

    dist = bfs(pacman_pos,currentGameState)
    #print(dist)

    for food in food_list:
        value += 90/(0.01 + dist[food[1]][food[0]])*num_food

    for i in range(num_ghosts):
        if scared_times[i]==0:
            value += dist[int(ghost_pos[i][1])][int(ghost_pos[i][0])]
        else :
            value += 100000000/((dist[int(ghost_pos[i][1])][int(ghost_pos[i][0])]*scared_times[i])+1)

    value += 1000000000000/(num_capsules+1)

    if currentGameState.isWin():
        value += INF
    elif currentGameState.isLose():
        value -= INF

    return value

def bfs(pos,currentGameState):
    walls = currentGameState.getWalls()
    n = walls.height
    m = walls.width
    # print(n,m)
    # print(walls)
    q = queue()
    dRow = [ -1, 0, 1, 0]
    dCol = [ 0, 1, 0, -1]
    q.append(pos)
    vis = [[ False for i in range(m)] for i in range(n)]
    dist = [[ 100000000 for i in range(m)] for i in range(n)]
    # print(pos)
    vis[pos[1]][pos[0]] = True
    dist[pos[1]][pos[0]] = 0
 
    while (len(q) > 0):
        cell = q.popleft()
        x = cell[0]
        y = cell[1]
        for i in range(4):
            adjx = x + dRow[i]
            adjy = y + dCol[i]
            # print(adjx,adjy)
            if (adjx<m and adjy<n and vis[adjy][adjx]==False and walls[adjx][adjy]==False):
                dist[adjy][adjx] = dist[y][x] + 1
                q.append((adjx, adjy))
                vis[adjy][adjx] = True
    return dist


# Abbreviation
better = betterEvaluationFunction
