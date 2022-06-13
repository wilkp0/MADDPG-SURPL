import numpy as np
import torch
import os
from maddpg.maddpg import MADDPG


class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)




    #providing a better select action method that can handle multiple different policies
    #as of right now, policies input and output methods will need their own methods.
    def select_action_2(self, o, noise_rate, epsilon):
        if np.random.uniform() > epsilon:
            pass
        else:
            #print(self.args.high_action)
            u = np.random.uniform(self.args.low_action, self.args.high_action, self.args.action_shape[self.agent_id])





    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            #print(self.args.high_action)
            u = np.random.uniform(self.args.low_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:

            #inputs the observation data into the policy in order to choose another action
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            #print(inputs)
            pi = self.policy.actor_network(inputs).squeeze(0)
            #print(pi)
            #print('{} : {}'.format(self.agent_id, pi))
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, self.args.low_action, self.args.high_action)
        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

