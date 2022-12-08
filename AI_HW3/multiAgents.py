from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


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
    """Your minimax agent (Part 1)"""
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth and self.evaluationFunction.
        Here are some method calls that might be useful when implementing minimax.
        
        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        def performMinimax(depth, agentIndex, gameState):
            if (gameState.isWin() or gameState.isLose() or depth > self.depth): # Terminal condition (win or lose or exceed depth)
                return self.evaluationFunction(gameState) # Get evaluation

            value = []  # Store the values for this node
            todo = gameState.getLegalActions(agentIndex)  # Get all legal actions (a list) of an agent
            for action in todo:
                successor = gameState.getNextState(agentIndex, action) # The gameState after taking the legal action 
                if((agentIndex+1) >= gameState.getNumAgents()): # When all agents are done
                    value += [performMinimax(depth+1, 0, successor)] # Pacman go to the next level
                else:
                    value += [performMinimax(depth, agentIndex+1, successor)]
                    # Calculate ghosts's min value for this successor(next possible state)
                    # Ghost n will use ghost n-1's successors. After getting the last ghost's value, we go backtracking.

            if agentIndex == 0: # Pacman
                if(depth == 1): # Back to root : return action
                    for i in range( len(value) ):
                        if (value[i] == max(value)):
                            return todo[i]
                else:
                    return max(value) # Not a root : return max value
            elif agentIndex > 0: # Ghosts : return min value
                return min(value)

        return performMinimax(1, 0, gameState) # Go to the function with agent 0 (pacman),depth 1
        # End your code (Part 1)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """Your minimax agent with alpha-beta pruning (Part 2)"""
    def getAction(self, gameState):
        """Returns the minimax action using self.depth and self.evaluationFunction"""
        # Begin your code (Part 2)
        def performAlphaBeta(depth, agentIndex, gameState, alpha, beta): # alpha : best for max, beta : best for min
            if (gameState.isWin() or gameState.isLose() or depth > self.depth): # Terminal condition (win or lose or exceed depth)
                return self.evaluationFunction(gameState) # Get evaluation

            valueList = []  # Store the values for this node
            todo = gameState.getLegalActions(agentIndex) # Get all legal actions (a list) of an agent
            for action in todo:
                successor = gameState.getNextState(agentIndex, action) # The gameState after taking the legal action 
                if((agentIndex+1) >= gameState.getNumAgents()): # When all agents are done
                    value = performAlphaBeta(depth+1, 0, successor, alpha, beta) # Pacman go to the next level 
                else:
                    value = performAlphaBeta(depth, agentIndex+1, successor, alpha, beta)
                    # Calculate ghosts's min value for this successor(next possible state) (We carry the current alpha and beta)
                    # Ghost n will use ghost n-1's successors. After getting the last ghost's value, we go backtracking.
                if(agentIndex == 0 and value > beta): # Pacman : impossible to go this branch : cut
                    return value
                if (agentIndex > 0 and value < alpha): # Ghost : impossible to go this branch : cut
                    return value
                if (agentIndex == 0 and value > alpha): # Pacman : find value > alpha : replace it
                    alpha = value
                if (agentIndex > 0 and value < beta): # Ghost : find value < beta : replace it
                    beta = value
                valueList += [value]

            if agentIndex == 0: # Pacman
                if(depth == 1): # Back to root : return action
                    for i in range(len(valueList)):
                        if (valueList[i] == max(valueList)):
                            return todo[i]
                else:
                    return max(valueList) # Not a root : return max value
            elif agentIndex > 0: # Ghosts
                return min(valueList) # Ghosts : return min value
        return performAlphaBeta(1, 0, gameState, -99999, 99999) # Go to the function with very small alpha, very big beta
        # End your code (Part 2)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """Your expectimax agent (Part 3)"""
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their  legal moves.
        """
        # Begin your code (Part 3)
        def performExpectimax(depth, agentIndex, gameState):
            if (gameState.isWin() or gameState.isLose() or depth > self.depth): # Terminal condition (win or lose or exceed depth)
                return self.evaluationFunction(gameState) # Get evaluation

            value = []  # Store the values for this node
            todo = gameState.getLegalActions(agentIndex) # Get all legal actions (a list) of an agent

            for action in todo:
                successor = gameState.getNextState(agentIndex, action) # The gameState after taking the legal action 
                if((agentIndex+1) >= gameState.getNumAgents()): # When all agents are done
                    value += [performExpectimax(depth+1, 0, successor)] # Pacman go to the next level
                else:
                    value += [performExpectimax(depth, agentIndex+1, successor)]
                    # Calculate ghosts's expected utilities for this successor(next possible state)
                    # Ghost n will use ghost n-1's successors. After getting the last ghost's expected utilities, we go backtracking.
            if agentIndex == 0: # Pacman
                if(depth == 1): # Back to root : return action
                    for i in range(len(value)):
                        if (value[i] == max(value)):
                            return todo[i]
                else:
                    return max(value) # Not a root : return max value
            elif agentIndex > 0: # Ghosts
                s = sum(value)
                l = len(value)
                return float(s/l) # Ghosts : return expected utilities
        return performExpectimax(1, 0, gameState) # Go to the function with agent 0 (pacman), depth 1
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function (Part 4)."""
    # Begin your code (Part 4)
    newPos = currentGameState.getPacmanPosition() # Get pacman position data
    newFood = currentGameState.getFood() # Get food position data
    newCapsules = currentGameState.getCapsules() # Get capsule position data
    newGhostStates = currentGameState.getGhostStates() # Get ghosts position data
    #Set up the weight
    INF = 1000000000000.0
    WEIGHT_FOOD = 5.0
    WEIGHT_CAP = 10.0
    WEIGHT_GHOST = -10.0
    WEIGHT_SCARED_GHOST = 200.0
    adjust_score = currentGameState.getScore() # Adjust score is modified from base score

    distancesToFoodList = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()] # Calculate the distances to food
    if len(distancesToFoodList) > 0:
        adjust_score += ( WEIGHT_FOOD / min(distancesToFoodList) ) # Consider the closest food. The food is closer, the score is higher.

    distancesToCapList = []
    for ii in range(len(newCapsules)):
        distancesToCapList.append(abs(newPos[0] - newCapsules[ii][0]) + abs(newPos[1] - newCapsules[ii][1])) #Calculate the distances between pacman and capsule
        if len(distancesToCapList) > 0:
                adjust_score += WEIGHT_CAP / min(distancesToCapList) # Consider the closest capsule

    for ghost in newGhostStates:
        distance = manhattanDistance(newPos, ghost.getPosition()) # Calculate the distances to ghost
        if distance > 0:
            if ghost.scaredTimer > 1:
                adjust_score += WEIGHT_SCARED_GHOST / distance # In scared time : the ghost is closer, the score is higher.
            elif ghost.scaredTimer > 0:
                adjust_score += (WEIGHT_SCARED_GHOST - 50) / distance # Scared time is about to end : WEIGHT decreases
            else:
                adjust_score += WEIGHT_GHOST / distance # Not in scared time : the ghost is closer, the score is lower. (WEIGHT_GHOST is negative)
        else: # Distance <= 0 : pacman died
            return -INF
    return adjust_score
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction

