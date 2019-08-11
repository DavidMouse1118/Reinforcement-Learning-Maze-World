import numpy as np
import pandas as pd


class rlalgorithm:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        self.actions = actions  
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name="Expected Sarsa"
        print("Using Expected Sarsa ...")

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


    '''Update the Q(S,A) state-action value table using the latest experience
       This is a not a very good learning update 
    '''
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_current = self.q_table.loc[s, a]

        if s_ != 'terminal':
            # calculate expected value according to epsilon greedy policy
            state_action_values = self.q_table.loc[s_,:]
            value_sum = np.sum(state_action_values)
            max_value = np.max(state_action_values)
            max_count = len(state_action_values[state_action_values == max_value])
            k = len(self.actions) # total number of actions

            expected_value_for_max = max_value * ((1 - self.epsilon) / max_count + self.epsilon / k) * max_count
            expected_value_for_non_max = (value_sum - max_value * max_count) * (self.epsilon / k)

            expected_value = expected_value_for_max + expected_value_for_non_max

            q_target = r + self.gamma * expected_value # max state-action value
        else:
            q_target = r  # next state is terminal

        self.q_table.loc[s, a] += self.lr * (q_target - q_current)  # update current state-action value

        return s_, self.choose_action(str(s_))


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
