# pacmanAgents.py
# ---------------
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


from pacman import Directions
from game import Agent
from heuristics import scoreEvaluation
import random
import Queue # We need queue to do tree search
import heapq

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class GreedyAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(scoreEvaluation(state), action) for state, action in successors]
        # get best choice
        bestScore = max(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)

class BFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write BFS Algorithm instead of returning Directions.STOP
        Q = Queue.Queue() # FIFO queue
        initialNode = Node(state, None, None) # Node definition in format of "Node(state, action, parent)"
        highestScoredNode = initialNode
        explored = dict()

        Q.put(initialNode) # Put the root note in the queue

        while not Q.empty():
            tmpNode = Q.get()
            if explored.has_key(tmpNode.state): # Check whether it is explored or not
                continue
            explored[tmpNode.state] = True # Label it if the node wasn't explored

            if tmpNode.state == None:
                continue
            if tmpNode.state.isWin():
                continue
            if tmpNode.state.isLose():
                continue

            for actions in tmpNode.state.getLegalPacmanActions():
                child = Node(tmpNode.state.generatePacmanSuccessor(actions), actions,
                             tmpNode)  # Node(state, action, parent)

                if not child.state == None:
                    Q.put(child)
                else: # Keep the node with the highest score
                    if scoreEvaluation(tmpNode.state) > scoreEvaluation(highestScoredNode.state):
                        highestScoredNode = tmpNode

        findRoot = highestScoredNode
        bestAction = ''
        while findRoot.parent != None:
            bestAction = findRoot.action
            findRoot = findRoot.parent
        return bestAction

class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write DFS Algorithm instead of returning Directions.STOP
        Q = Queue.LifoQueue() # LIFO Queue
        initialNode = Node(state, None, None)  # Node definition in format of "Node(state, action, parent)"
        highestScoredNode = initialNode
        explored = dict()

        Q.put(initialNode)  # Put the root note in the queue

        while not Q.empty():
            tmpNode = Q.get()
            if explored.has_key(tmpNode.state):  # Check whether it is explored or not
                continue
            explored[tmpNode.state] = True  # Label it if the node wasn't explored

            if tmpNode.state == None:
                continue
            if tmpNode.state.isWin():
                continue
            if tmpNode.state.isLose():
                continue

            for actions in tmpNode.state.getLegalPacmanActions():
                child = Node(tmpNode.state.generatePacmanSuccessor(actions), actions,
                             tmpNode)  # Node(state, action, parent)

                if not child.state == None:
                    Q.put(child)
                else:  # Keep the node with the highest score
                    if scoreEvaluation(tmpNode.state) > scoreEvaluation(highestScoredNode.state):
                        highestScoredNode = tmpNode

        findRoot = highestScoredNode
        bestAction = ''
        while findRoot.parent != None:
            bestAction = findRoot.action
            findRoot = findRoot.parent
        return bestAction

##########################################
# Node definition for search algorithms
##########################################
class Node(object):
    def __init__(self,state,action=None,parent=None):
        self.state = state
        self.action = action  # Action executed before this state
        self.parent = parent