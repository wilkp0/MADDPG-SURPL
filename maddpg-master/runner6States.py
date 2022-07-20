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

        self.demand1 = [1,1,2,3,2,1]
        self.demand2 = [1,1,2,3,2,1]

        self.totalCost = []
        self.low1 = []
        self.high1 = []
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
        returns = []
        averew = []
        low = []
        high = []
        self.args.evaluate_rate=3*30
        self.args.graphing_rate = 3*20
        self.args.time_steps = 3*32000
        for time_step in tqdm(range(self.args.time_steps)):
            #self.env.render()
#NEVER REACHES DONE BECAUSE OF THIS 
            # reset the environment
            self.episode_limit = 1
            if time_step % self.episode_limit == 0:
#if done:
                s = self.env.reset()
            u = []
            graphing = []
            
            #Generation new actions based on observations
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    #goes to select action in maddpg agent class
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    graphing.append(action)
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
                x, y = self.evaluate()
                averew.append(x)
                returns.append(y)
                #averew, returns.append(self.evaluate())
                low.append(-50)
                high.append(-64)
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.plot(range(len(low)), low)
                plt.plot(range(len(high)), high)
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

            
            if time_step > 0 and time_step % self.args.graphing_rate == 0:
                #print('Graphing', graphing)
                energy = np.array(graphing)
             
                #Usedenergy = (energy* self.demand1) # working
                Usedenergy1 = (energy[0]* self.demand1)
                Usedenergy2 = (energy[1]* self.demand2)

                #dCharge = np.add(Usedenergy[0],Usedenergy[1])
                dCharge = np.add(Usedenergy1,Usedenergy2)
                dCharge = max(dCharge)

                print('Energy1', Usedenergy1) 
                print ('Energy2', Usedenergy2) #,  file=self.fileOut)
                #comfort = np.subtract([3,1,1],Usedenergy)
                comfort1 = np.subtract(self.demand1,Usedenergy1)
                comfort2 = np.subtract(self.demand2,Usedenergy2)

                #print('1comfort', comfort1)
                #print('2comfort', comfort2)
                comfort = np.concatenate([comfort1] + [comfort2])
                comfort = comfort **2
                #print('comf concat', comfort) #, file=self.fileOut)

                #print('concat', Usedenergy)
                Usedenergy = np.concatenate([Usedenergy1] + [Usedenergy2])
                Usedenergy = np.sum(Usedenergy)
                print('UseedEnegy', Usedenergy, file=self.fileOut)
                print('Comfort', comfort, file=self.fileOut)
                print('DemandCharge', dCharge, file=self.fileOut)
                
                Cost = Usedenergy +sum(comfort)+ 2*dCharge 
                print('FCost', Cost, file=self.fileOut)
                print('*'*25, file=self.fileOut)
                tensorOpt = -25
                tensorOrig = -32

                self.writer.add_scalar("Cost per Episode", -Cost, time_step)
                self.writerOpt.add_scalar("Cost per Episode", tensorOpt, time_step)
                self.writerOrig.add_scalar("Cost per Episode", tensorOrig, time_step)

                self.totalCost.append(-Cost)
                self.low1.append(-25)
                self.high1.append(-32)
                #plt.figure(10)
                #plt.plot(range(len(self.totalCost)), self.totalCost)
                #plt.plot(range(len(self.low1)), self.low1)
                #plt.plot(range(len(self.high1)), self.high1)
                #plt.ylim(-40,-20)
                #plt.xlabel('episodes * ' + str(self.args.evaluate_rate / self.episode_limit))
                #plt.ylabel('Costs')
                #plt.savefig(self.save_path + '/plt2.png', format='png') 
                

    def evaluate(self):
        returns = []
        self.args.evaluate_episode_len = 1
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
            returns.append(rewards)

        print('Returns is', rewards)
        self.env.reset()
        return (sum(returns) / self.args.evaluate_episodes, rewards)
