from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
from torch.utils.tensorboard import SummaryWriter
import torch
#import torchvision
#import torchvision.transforms as transforms
import sys
import os
import numpy as np
from datetime import datetime, date
import matplotlib
matplotlib.use('Agg')      #Resource exhaustion, speed up using (Agg>>TkAgg)
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
       # C:\Users\Patrick Wilk\Documents\RL\JJ_first_Look\MADDPG_DPR\SURPL_MARL\results\04\04\27\22\2022_11\smart_building
        #self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        now = datetime.now().strftime("%m/%D/%Y_%H")
        self.save_path = "/Users/Patrick Wilk/Documents/RL/MADDPG/results/" + self.args.scenario_name
        sourceDir = "/Users/Patrick Wilk/Documents/RL/MADDPG"
        self.fileOut = open(sourceDir + "/results/MADDPG2.txt", "w+")
        #figDir = sourceDir + '/results/plt_figs'
        #if not os.path.exists(self.save_path):
            #os.makedirs(self.save_path)
            
        self.writer = SummaryWriter(log_dir=self.save_path + "/logs1/agent")
        self.writerOpt = SummaryWriter(log_dir=self.save_path + "/logs1/opt")
        self.writerOrig = SummaryWriter(log_dir=self.save_path + "/logs1/orig")

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        self.episode_limit = 3
        self.args.time_steps = 3*32000

        self.args.evaluate_rate=3*30
        self.args.graphing_rate = 3*20
        
        returns = []

        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment  

            if time_step % self.episode_limit == 0:
                print('RESETING')
                s = self.env.reset()
            u = []

            #Generation new actions based on observations
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)
            for i in range(self.args.n_agents, self.args.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            s_next, r, done, info = self.env.step(actions)
            #if done:
                #s = self.env.reset()
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                agentsReward = self.evaluate()
                #plt.figure()
                #plt.plot(range(len(returns)), returns)
                #plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                #plt.ylabel('average returns')
                #plt.savefig(self.save_path + '/plt.png', format='png')
                tensorOpt = -16.5
                tensorOrig = -2
                print('LOOK HERE for reward',  sum(r)/ self.args.evaluate_episodes)
                self.writer.add_scalar("Cost per Episode", sum(r)/self.args.evaluate_episodes, time_step)
                self.writerOpt.add_scalar("Cost per Episode", tensorOpt, time_step)
                self.writerOrig.add_scalar("Cost per Episode", tensorOrig, time_step)

            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)
            np.save(self.save_path + '/returns.pkl', returns)

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                #self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
        print('Returns is', rewards)
        #print('eveluate episodes', self.args.evaluate_episodes)
        return sum (returns) / self.args.evaluate_episodes
