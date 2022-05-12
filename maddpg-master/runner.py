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
        self.save_path = "/Users/Patrick Wilk/Documents/RL/JJ_first_Look/MADDPG_DPR/SURPL_MARL/results/" + self.args.scenario_name
       
        #if not os.path.exists(self.save_path):
            #os.makedirs(self.save_path)
            
        self.writer = SummaryWriter(log_dir=self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        returns = []
        self.args.evaluate_rate=3*30
        self.args.time_steps = 3*30000
        for time_step in tqdm(range(self.args.time_steps)):
            #self.env.render()
#NEVER REACHES DONE BECAUSE OF THIS 
            # reset the environment
            self.episode_limit = 3
            if time_step % self.episode_limit == 0:
#if done:
                s = self.env.reset()
            u = []
            
            #Generation new actions based on observations
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    #goes to select action in maddpg agent class
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)
            for i in range(self.args.n_agents, self.args.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            #self.env.render()
            #Calls the multi-agent environment for the next steps
            s_next, r, done, info = self.env.step(actions)
            #self.env.render()
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])

            #the next observation replaces the previous observation
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')

                

                for count, i in enumerate(self.env.world.actions):
                    p = 0
                    q = 0
                    u = []
                    for j in range(1, len(i), 3):
                        u.append(sum(i[j-1:j+2]))


                    plt.figure()
                    plt.plot(u[-100:])
                    plt.ylabel("Action")
                    plt.xlabel("Time Step")
                    label = ""
                    if count == 0:
                        label = "Smart Building 1"
                    elif count == 1:
                        label ="Smart Building 2"
                    plt.title(label)
                    plt.savefig(self.save_path + "/" + label + ".png", format ='png')
                    plt.close('all')

            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.noise - 0.0000005)
            np.save(self.save_path + '/returns.pkl', returns)

            #img_grid = torchvision.utils.make_grid(returns)
            #self.writer.add_image('example',img_grid)
            #self.writer.close()
            #sys.exit()
    

    def evaluate(self):
        returns = []
        self.args.evaluate_episode_len = 3
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                #self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        #0 gaussian noise or epsilon
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
                #DISPLAY REWARDS
                #DISPLAY ACTIONS
                #DISPLAY STATES
                #DIFFERENT AGENTS??
            returns.append(rewards)
            self.writer.add_scalar("Loss/train", sum(returns), episode)
            self.writer.add_scalar("Loss/train", len(returns),sum(returns))
            # self.writer.add_scalar("Loss/test", sum(returns), episode)
            # self.writer.add_scalar("Accuracy/train", sum(returns), episode)
            # self.writer.add_scalar("Accuracy/test", sum(returns), episode)
        print('Returns is', rewards)
        self.env.reset()
        return sum(returns) / self.args.evaluate_episodes
