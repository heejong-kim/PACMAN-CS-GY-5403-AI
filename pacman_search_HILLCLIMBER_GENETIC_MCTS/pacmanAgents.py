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
from heuristics import *
import random
import math

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

class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = []
        for i in range(0, 5):
            self.actionList.append(Directions.STOP)
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Hill Climber Algorithm instead of returning Directions.STOP

        # GetAction Function: Called with every frame
        # Initialization
        possibleSeq = self.actionList  # List with Directions.STOP
        possible = state.getAllPossibleActions()  # Possible actions
        bestSeq = self.actionList
        bestScore = 0.0
        # Possible action sequence will keep updating if current state is not in losing or winning state.
        # And the loop will stop and return BEST ONE when we hit the limit calling "generatePacmanSuccessor"
        print("Current state is not winning or losing:", [state.isWin() + state.isLose() == 0])
        while state.isWin() + state.isLose() == 0:
            tempState = state
            tempScore = scoreEvaluation(state)
            for i in range(0, len(possibleSeq)):
                if random.random() > 0.5:
                    possibleSeq[i] = possible[random.randint(0, len(possible) - 1)]
            j = 0
            while tempState != None and tempState.isWin() + tempState.isLose() == 0 and j != len(possibleSeq):
                tempState = tempState.generatePacmanSuccessor(possibleSeq[j])
                j = j + 1
            if tempState == None or tempState.isWin() + tempState.isLose() != 0:
                print("Hit the limit calling generatePacmanSuccessor OR met winning or losing state")
                return bestSeq[0]
            else:
                tempScore = scoreEvaluation(tempState)
                if bestScore < tempScore:
                    print("New best sequence is updated")
                    bestScore = tempScore
                    bestSeq = possibleSeq
                    print("Current BEST SCORE:", bestScore, "with best move (", bestSeq[0], ")")

class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = []
        for i in range(0, 5):
            self.actionList.append(Directions.STOP)
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP

        def rankSelection(population, scoreRank):
            """
            rankSelection returns a pair of chromosomes from population according to given scoreRank
            :param population: 8 chromosomes to be selected
            :param scoreRank: rank of chromosomes from the population. (Larger number (high score) gives high chance to be selected)
            :return: a pair of chromosomes from population
            """
            print("Rank Selection")
            rankpool = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8,
                        8, 8, 8, 8]
            P1 = random.randint(0, 35)
            P2 = random.randint(0, 35)
            return (population[scoreRank[rankpool[P1] - 1]], population[scoreRank[rankpool[P2] - 1]])

        def mutation(newPopulation):
            """
            mutation mutates chromosomes from population if rank test result is less than 0.1
            One action from the selected chromosome will be replaced by random action.
            :param newPopulation: population to be mutated
            :return: mutated population
            """
            print("Mutation")
            thr = 0.1
            randomTest = [random.random() < thr for x in range(newPopulation.__len__())]
            idx = 0
            for bool in randomTest:
                if bool == True:
                    chromosome_mutation = newPopulation[idx]
                    chromosome_mutation[random.randint(0, len(possible) - 1)] = possible[
                        random.randint(0, len(possible) - 1)]
                    newPopulation[idx] = chromosome_mutation
                idx = idx + 1
                return newPopulation

        """
        Initialization
        """
        population_size = 8
        possible = state.getAllPossibleActions()
        sequenceInit = self.actionList
        scoreList = []
        population = []
        print("Initial population with random actions")
        for k in range(0, population_size):
            possibleSeq = []
            for i in range(0, len(sequenceInit)):
                possibleSeq.append(possible[random.randint(0, len(possible) - 1)])
            j = 0
            tempState = state
            while tempState != None and tempState.isWin() + tempState.isLose() == 0 and j != len(possibleSeq):
                tempState = tempState.generatePacmanSuccessor(possibleSeq[j])
                j = j + 1
            tempScore = scoreEvaluation(tempState)
            scoreList.append(tempScore)
            population.append(possibleSeq)
        scoreRank = [i[0] for i in sorted(enumerate(scoreList), key=lambda x: x[1])]

        # While not running out of calling function or have wining state or losing state, keep update population
        oldPopulation = population
        oldScoreRank = scoreRank
        while True:
            bestAction = oldPopulation[oldScoreRank.index(7)][0]
            newPopulation = []
            newScoreList = []
            number_of_picking_parent = 4
            for j in range(1, number_of_picking_parent + 1):
                parent = rankSelection(oldPopulation, oldScoreRank)
                child1 = []
                child2 = []
                if random.random() < 0.7:  # CROSSOVER
                    print("CROSSOVER")
                    for i in range(0, len(sequenceInit)):
                        if random.random() < 0.5:
                            child1.append(parent[0][i])
                            child2.append(parent[1][i])
                        else:
                            child1.append(parent[1][i])
                            child2.append(parent[0][i])
                else:  # KEEP PARENT
                    print("PARENT KEPT")
                    child1 = parent[0]
                    child2 = parent[1]
                newPopulation.append(child1)
                newPopulation.append(child2)
            newPopulationMutated = mutation(newPopulation)
            # checking score
            for k in range(0, population_size):
                possibleSeq = newPopulationMutated[k]
                j = 0
                tempState = state
                while tempState != None and j != len(possibleSeq):
                    tempState = tempState.generatePacmanSuccessor(possibleSeq[j])
                    if tempState == None or tempState.isWin() or tempState.isLose():
                        break
                    j = j + 1
                if j != len(possibleSeq) and tempState == None:
                    print("Hit the limit calling generatePacmanSuccessor OR met winning or losing state")
                    return bestAction
                tempScore = scoreEvaluation(tempState)
                newScoreList.append(tempScore)
            newScoreRank = [i[0] for i in sorted(enumerate(newScoreList), key=lambda x: x[1])]
            oldPopulation = newPopulationMutated
            oldScoreRank = newScoreRank
        return bestAction

class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write MCTS Algorithm instead of returning Directions.STOP
        def UCB(currentNode):
            """
            UCB calculates UCB score of the current node
            :param currentNode:
            :return: UCB score
            """
            C = 1.0
            visit_count = currentNode.visit_count
            if currentNode.visit_count == 0:
                visit_count = 0.000000000000001
            UCBvalue = float(sum(currentNode.total_reward) / float(visit_count)) + C * math.sqrt(
                float(2 * math.log(currentNode.parent.visit_count) / float(visit_count)))
            return float(UCBvalue)

        def EXPAND(currentNode,currentState,terminalFlag):
            """
            EXPAND expands MCTS tree if the node still has nodes to be expanded (action to be explored).
            :param currentNode:
            :param currentState:
            :param terminalFlag: helps to get out of function and return best action from root at that time.
            :return: expanded child
            """
            tmpActionList = currentState.getLegalPacmanActions()
            if tmpActionList.__len__() == 0:
                print("Reached the end, No more actions to generate!")
                terminalFlag = 1
                return None, terminalFlag
            if currentNode.next_action.__len__() != tmpActionList.__len__():
                for action in tmpActionList:
                    if not action in currentNode.next_action:
                        currentNode.next_action.append(action)
                        child = Node(action,currentNode)
                        currentNode.children[action] = child
                        return child, terminalFlag
            else: print("Warning: Already expanded all (This condition never happen)")

        def TREEPOLICY(currentNode, currentState, terminalFlag):
            """
            TREEPOLICY expands tree until we run out of computational limit. (generatePacmanSuccessor in this HW)
            :param currentNode: current node input
            :param currentState: current state input
            :param terminalFlag: helps to get out of function and return best action from root at that time.
            :return:
            """
            while currentState != None:
                child = []
                terminalFlag = 0
                if currentNode.next_action.__len__() == 0:
                    print("Initial expansion")
                    [child, terminalFlag] = EXPAND(currentNode,currentState,terminalFlag)
                    return child, terminalFlag
                else:
                    if currentNode.next_action.__len__() != currentState.getLegalPacmanActions().__len__():
                        print("Not fully expanded, expand more")
                        [child, terminalFlag] = EXPAND(currentNode, currentState, terminalFlag)
                        return child, terminalFlag
                    else:
                        print("Fully expanded, choose best")
                        currentNode = BESTCHILD(currentNode)
            return currentNode,terminalFlag

        def BESTCHILD(currentNode):
            """
            BESTCHILD compares USB score and choose the best child
            :param currentNode: current node input
            :return: best child according to UCB score
            """
            bestScore = 0.0
            bestChildren = []
            bestChild = currentNode
            for action in currentNode.next_action:
                child = currentNode.children[action]
                if bestScore < UCB(child):
                    bestScore = UCB(child)
                    bestChild = child
                    bestChildren = []
                elif bestScore == UCB(child):
                    bestChildren.append(child)
            if not bestChildren == None:
                bestChildren.append(bestChild)
            bestNode = random.choice(bestChildren)
            return bestNode

        def DEFAULTPOLICY(currentState, terminalFlag):
            """
            DEFAULTPOLICY calculate the reward (normalized score) from the current node state after 5 random steps.
            :param currentState: current state input (the starting point of random walk)
            :param terminalFlag: helps to get out of function and return best action from root at that time.
            :return: reward and terminal flag
            """
            tmpState = currentState
            i = 0
            while tmpState != None and i < 5:
                tmpActionList = tmpState.getLegalPacmanActions()
                if tmpActionList.__len__() == 0:
                    print("Meet the end, No more actions to generate!")
                    terminalFlag = 1
                    reward = None
                    return reward, terminalFlag
                tmpState = tmpState.generatePacmanSuccessor(random.choice(tmpActionList))
                i = i + 1
            if tmpState == None:
                print("Hit the maximum generatePacmanSuccessor() call STOPPED from action state")
                terminalFlag = 1
                reward = None
                return reward, terminalFlag
            reward = normalizedScoreEvaluation(currentState,tmpState)
            print(reward)
            return reward, terminalFlag

        def BACKUP(currentNode, reward):
            """
            BACKUP follow the path from the current node up to the root, update number of visits and reward.
            :param currentNode: current node input
            :param reward: reward input
            :return:
            """
            #print("BEFORE BACKUP")
            #print("visit", currentNode.visit_count, "reward", currentNode.total_reward, "children",
            #      currentNode.children.__len__())
            while currentNode.parent != None:
                #print("BACKUP WHILE LOOP HAPPENED")
                currentNode.visit_count = currentNode.visit_count + 1
                currentNode.total_reward.append(reward)
                #print("visit",currentNode.visit_count,"reward", currentNode.total_reward, "children", currentNode.children.__len__())
                currentNode = currentNode.parent
            #print("Update the root's")
            currentNode.visit_count = currentNode.visit_count + 1
            currentNode.total_reward.append(reward)
            #print("visit", currentNode.visit_count, "reward", currentNode.total_reward, "children",
                  #currentNode.children.__len__())

        def MCTS(root,state,terminalFlag):
            while True:
                print("================")
                print("MCTS loop")
                print("================")
                currentNode = []
                [currentNode, terminalFlag] = TREEPOLICY(root,state,terminalFlag)
                if terminalFlag == 1:
                    return None, terminalFlag
                # update the state according to previous actions made for the currentNode
                prevActions = []
                tmpNode = currentNode
                updatedState = state
                while tmpNode.parent != None:
                    prevActions.append(tmpNode.action)
                    tmpNode = tmpNode.parent
                print("Current depth before ROLLOUT",prevActions.__len__())
                for i in range(prevActions.__len__()):
                    updatedState = updatedState.generatePacmanSuccessor(prevActions[prevActions.__len__()-(i+1)])
                    if updatedState == None or (updatedState.isWin()+updatedState.isLose()!=0):
                        terminalFlag = 0
                        return None, terminalFlag
                [reward, terminalFlag] = DEFAULTPOLICY(updatedState, terminalFlag)
                if terminalFlag == 1:
                    return None, terminalFlag
                BACKUP(currentNode, reward)
            return BESTCHILD(root), terminalFlag


        """
        MCTS algorithm
        """
        root = Node(None, None)
        currentNode = root
        terminalFlag = 0
        while currentNode != None:
            tmpNode = currentNode
            [currentNode, terminalFlag] = MCTS(tmpNode, state,terminalFlag)
            if terminalFlag == 1:
                maxVisit = 0
                maxVisitChild = tmpNode
                maxVisitChildren = []
                for action in tmpNode.next_action:
                    child = tmpNode.children[action]
                    if maxVisit < child.visit_count:
                        maxVisitChildren = []
                        maxVisit = child.visit_count
                        maxVisitChild = child
                        print("MAX VISIT updated", maxVisit)
                    if maxVisit == child.visit_count:
                        maxVisitChildren.append(child)
                maxVisitChildren.append(maxVisitChild)
                nodeTobereturned = random.choice(maxVisitChildren)
                return nodeTobereturned.action

##########################################
# Node definition for MCTS algorithm
##########################################
class Node(object):
    def __init__(self, action=None, parent=None):
        self.action = action  # Action executed before this state
        self.parent = parent
        self.next_action = []
        self.children = dict()
        self.visit_count = 0
        self.total_reward = []
