import numpy as np
import pandas as pd


class rlalgorithm:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1, lambda_decay=0.9):
        self.actions = actions  
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.lambda_decay = lambda_decay
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.e_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name="Sarsa(λ)"
        print("Using Sarsa(λ) ...")

    '''Choose the next action to take given the observed state using an epsilon greedy policy'''
    def choose_action(self, observation):
        # Add non-existing state to our q_table
        self.check_state_exist(observation)
 
        # Select next action
        if np.random.uniform() >= self.epsilon:
            # Choose argmax action
            state_action_values = self.q_table.loc[observation, :]
            action = np.random.choice(state_action_values[state_action_values == np.max(state_action_values)].index) # handle multiple argmax with random
        else:
            # Choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)

        # determine q_target
        if s_ != 'terminal':
            a_ = self.choose_action(str(s_)) # argmax action
            q_target = r + self.gamma * self.q_table.loc[s_, a_] # max state-action value
        else:
            q_target = r  # next state is terminal

        # update q_table using eligibility trace
        error = q_target - self.q_table.loc[s, a]
        self.e_table.loc[s, a] += 1

        self.q_table += self.lr * error * self.e_table # update state-action value for all states and actions

        # update eligibility trace
        if s_ != 'terminal':
            self.e_table *= self.gamma * self.lambda_decay # decay the eligibility trace for all states and actions
        else:
            self.e_table = pd.DataFrame(columns=self.actions, dtype=np.float64) # clear the eligibility

        return s_, a_


    '''States are dynamically added to the Q(S,A) table as they are encountered'''
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

        if state not in self.e_table.index:
            # append new state to q table
            self.e_table = self.e_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.e_table.columns,
                    name=state,
                )
            )
