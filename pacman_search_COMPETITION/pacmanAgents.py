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


"""
Name: Heejong Kim / NetID: hk2451
"""

from pacman import Directions
from game import Agent
import random
import math
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

class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        tempState = state;
        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valide actions
        return self.actionList[0];

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


class CompetitionAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        def findWall(state):
            """
            Finding edible (reachable) position and wall position in initial state.
            """
            pelletPosition = state.getPellets()
            capsulePosition = state.getCapsules()
            initialPacmanPosition = tuple(state.getPacmanPosition())
            initialGhostPosition = [state.getGhostPosition(1), state.getGhostPosition(2)]
            ediblePosition = pelletPosition + capsulePosition + [
                initialPacmanPosition] + initialGhostPosition
            maxP = max(ediblePosition)
            minP = min(ediblePosition)
            coordinates = []
            for x in range(minP[0], maxP[0] + 1):
                for y in range(minP[1], maxP[1] + 1):
                    coordinates.append((x, y))
            wallPosition = list(set(coordinates) - set(ediblePosition))
            return wallPosition, ediblePosition

        def makeEdibleMaze(self):
            """
            Using the edible (reachable) position, ...
            construct a graph so we can calculate the shortest path.
            """
            ediblePosition = self.ediblePosition
            maxP = max(ediblePosition)
            minP = min(ediblePosition)
            sorted(ediblePosition, key=lambda k: (k[1], k[0]))
            edibleMaze = {pos: [] for pos in ediblePosition}
            for x, y in ediblePosition:
                if x < maxP[0] + 1 and edibleMaze.has_key((x + 1, y)):
                    edibleMaze[(x, y)].append(("EAST", (x + 1, y)))
                    edibleMaze[(x + 1, y)].append(("WEST", (x, y)))
                if y < maxP[1] + 1 and edibleMaze.has_key((x, y + 1)):
                    edibleMaze[(x, y)].append(("NORTH", (x, y + 1)))
                    edibleMaze[(x, y + 1)].append(("SOUTH", (x, y)))
            return edibleMaze

        [self.wallPosition, self.ediblePosition] = findWall(state)
        self.edibleMaze = makeEdibleMaze(self)
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        def mazeDistance(startPos, endPos, edibleMaze):
            """
            A* search algorithm on the edible (reachable) graph
            It finds the shortest path between two positions.
            evaluation function = [Manhattan distance from current neighbor position to end position]
            """
            Q = []
            mazeDist = 0
            heapq.heappush(Q, (mazeDist, startPos))
            explored = set()
            while Q:
                tmpPos = heapq.heappop(Q)
                mazeDist = tmpPos[0] + 1
                if tmpPos[1] == endPos:
                    return mazeDist
                if tmpPos[1] in explored:
                    continue
                explored.add(tmpPos[1])
                for nextPos in edibleMaze[tmpPos[1]]:
                    if nextPos[1] in explored:
                        continue
                    mazePriority = mazeDist + abs(nextPos[1][0] - endPos[0]) + abs(
                        nextPos[1][1] - endPos[1])
                    heapq.heappush(Q, (mazePriority, nextPos[1]))
            return ""

        def heuristic(state, edibleMaze):
            """
            heuristic function = [Reachable maze distance from Pacman to the closest pellets] + ...
            [Number of pellets left in the maze] + [Weight for losing state]
            """
            pelletPosition = state.getPellets()
            capsulePosition = state.getCapsules()
            ediblePosition = pelletPosition + capsulePosition
            pacmanPosition = state.getPacmanPosition()

            distFromPacman = [{abs(xy[1] - pacmanPosition[1]) + abs(xy[0] - pacmanPosition[0]): xy}
                              for xy in ediblePosition]

            closestPelletDist = 0
            if pelletPosition.__len__() is not 0:
                closestPelletDist = mazeDistance(pacmanPosition, distFromPacman[0].values()[0],
                                                 edibleMaze)

            edibleLeft = 0
            if ediblePosition.__len__() is not None:
                edibleLeft = ediblePosition.__len__()

            H = closestPelletDist + edibleLeft + [0, 10000][state.isLose()]
            return H

        """
        Current state's position information
        """
        pelletPosition = state.getPellets()
        capsulePosition = state.getCapsules()
        ediblePosition = pelletPosition + capsulePosition
        pacmanPosition = state.getPacmanPosition()
        distFromPacman = [{abs(xy[1] - pacmanPosition[1]) + abs(xy[0] - pacmanPosition[0]): xy} for
                          xy in
                          ediblePosition]
        distFromPacman.sort()

        """
        A* search algorithm on game state
        """
        edibleMaze = self.edibleMaze
        Q = []
        initialNode = Node(state, None, None)  # Node(state, action, parent)
        explored = dict()
        pathCost = -1  # Because depth starts from 0, pathCost initialized by -1.
        rootHeuristicCost = heuristic(initialNode.state, edibleMaze)
        initialPriorityScore = pathCost
        highestGameScoreNode = initialNode
        explored[initialNode.state] = True
        bestGameScore = 0
        heapq.heappush(Q, (initialPriorityScore, initialNode))

        while Q:
            T = heapq.heappop(Q)
            tmpNode = T[1]
            if [tmpNode.state.getPacmanPosition()] == distFromPacman[0].values():
                break
            else:
                pathCost = pathCost + 1  # pathCost = Depth of the node
                if tmpNode.state.isWin() or tmpNode.state.isLose() or tmpNode.state == None:
                    break
                else:
                    for actions in tmpNode.state.getLegalPacmanActions():
                        child = Node(tmpNode.state.generatePacmanSuccessor(actions), actions,
                                     tmpNode)
                        if child.state == None:
                            currentScore = tmpNode.state.getScore()
                            if currentScore > bestGameScore:
                                bestGameScore = currentScore
                                highestGameScoreNode = tmpNode
                        else:
                            if explored.has_key(
                                    child.state) or child.state.getPacmanPosition() not in self.ediblePosition:
                                continue
                            if not explored.has_key(child.state):
                                explored[child.state] = True
                            currentPriorityScore = pathCost + heuristic(child.state,
                                                                           edibleMaze) - rootHeuristicCost
                            heapq.heappush(Q, (currentPriorityScore, child))
                            currentScore = child.state.getScore()
                            if currentScore > bestGameScore:
                                bestGameScore = currentScore
                                highestGameScoreNode = child

        findRoot = highestGameScoreNode
        bestAction = ''
        while findRoot.parent != None:
            bestAction = findRoot.action
            findRoot = findRoot.parent
        return bestAction

class Node(object):
    """
    Node definition for the algorithm.
    """
    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action  # Action executed before this state
        self.parent = parent
