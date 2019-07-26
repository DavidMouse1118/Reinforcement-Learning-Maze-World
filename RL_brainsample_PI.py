import numpy as np
import pandas as pd


class rlalgorithm:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        self.actions = actions  
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name="Asynch PI Hack"

    '''Choose the next action to take given the observed state using an epsilon greedy policy'''
    def choose_action(self, observation):
        self.check_state_exist(observation)
 
        #BUG: Epsilon should be .1 and signify the small probability of NOT choosing max action
        if np.random.uniform() >= self.epsilon:
           
            state_action = self.q_table.loc[observation, :]
           
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
         
            action = np.random.choice(self.actions)
        return action


    '''Update the Q(S,A) state-action value table using the latest experience
       This is a not a very good learning update 
    '''
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        if s_ != 'terminal':
            a_ = self.choose_action(str(s_))
            q_target = r + self.gamma * self.q_table.loc[s_, a_] 
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] = q_target  # update
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
