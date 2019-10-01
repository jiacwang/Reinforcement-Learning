# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        # Jiachen Wang
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            current_state = self.mdp.getStates()
            # the value of next round is initialized as the copy of the current value.
            v_k_plus_one = self.values.copy()
            # calculate the value for each state
            for state in current_state:
                # terminal state has default value zero.
                if not self.mdp.isTerminal(state):
                    # if this state is not terminal state,
                    # get possible actions from this state
                    possible_actions = self.mdp.getPossibleActions(state)
                    # calculate Q values for theses actions
                    Q_values = [self.getQValue(state, action) for action in possible_actions]
                    # pick the max, which is the value for this state
                    value_of_this_state = max(Q_values)
                    # store the value for this state into dictionary
                    v_k_plus_one[state] = value_of_this_state
            # all values of the current states are calculated, update them.
            # print("updated self.values")
            self.values = v_k_plus_one


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Jiachen Wang
        # Calculate Q_value
        Q_value = sum(prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state]) for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action))
        #print("successfully calculated Q_value")
        return Q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Jiachen Wang
        possible_actions = self.mdp.getPossibleActions(state)
        if not possible_actions:
            return None
        else:
            # return the action that has the max Q_value.
            # use max() function, the sorting key is Q_value
            return max(possible_actions, key = lambda action: self.getQValue(state, action))

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
