import numpy as np
import cvxpy as cp
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

sourceDir = "/Users/Patrick Wilk/Documents/RL/MADDPG"
fileOut = open(sourceDir + "/results/MADDPG.txt", "w+")
figDir = sourceDir + '/results/plt_figs'




class Scenario(BaseScenario):

    def __init__(self):
        #self.day_reward = True
        self.method = "main"

    def make_world(self):
        print('MAKE WORLD', file=fileOut)
        world = World()

        #used for plotting actions done by policy. 
        #Future work: implementing it so that it can work with any number of input dimensions and entities
        world.actions = [[],[]]
        world.time = 0
        num_agents = 2
        world.num_agents = num_agents
        world.dim_c = 1
        #self.done = True 
        world.dim_p = 1
        num_landmarks = 1
        num_adversaries = 0
        self.demandCharge = []
        world.collaborative = True

        # add agents
        world.agents = [Agent() for i in range(num_agents)]

        for i, agent in enumerate(world.agents):
            if i == 0:
                agent.name = 'SB1'
            elif i == 1:
                agent.name = 'SB2'
            agent.silent = False
            agent.movable = False
            agent.size = 0.1
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.name = 'Load'
                landmark.size = .2
            landmark.collide = False
            landmark.movable = False
            landmark.boundary = False

        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):

        print('AT RESET WORLD', file=fileOut)
        self.total_reward = 0
        self.load = 0
        self.HighPenalty = []
        world.time = 0
        world.energy_costs = []

        #filling out agent detail
        for agent in world.agents:
            if agent.name == "SB1":
                agent.demands = [3,1,1]
                agent.min = 1
                agent.energy = 0
                agent.agent_callback = None
            if agent.name == "SB2":
                agent.demands = [3,1,1]
                agent.min = 1
                agent.energy = 0
                agent.agent_callback = None
            #agent.state.c =np.random.uniform(0,1, world.dim_c)
            #agent.state.c = np.zeros(world.dim_c)
            #agent.action.c = np.zeros(world.dim_c)
            agent.state.c = np.random.uniform(0, 1, world.dim_c)
            agent.action.c = np.random.uniform(0, 1, world.dim_c)
            

        #filling out landmark detail
        for landmark in world.landmarks:
            if landmark.name == "Load":
                landmark.state.demand = [3,1,1]


    #reward method for calling. Deliberates to specific reward functions
    def reward(self,agent,world):
        print('AT REWARD', file=fileOut)
        print('WORLD TIME', world.time)
        if world.time == 3:
            #dCharge = np.array([(a.state.c*a.demands) for a in world.agents])
            #dCharge = max(np.add(dCharge[0], dCharge[1]))
            #self.done = True
            self.reset_world(world)

        #world.time +=1 
        if world.time == 3:
            dCharge = np.array([(a.state.c*a.demands) for a in world.agents])
            dCharge = max(np.add(dCharge[0], dCharge[1]))   
        else:
            #self.HighPenalty.append([agent.state.c*agent.demands[world.time-1] for agent in world.agents])
            dCharge = 0
        #agents are rewarded based on the minimal cost compared to demand
        #dCharge = self.demandCharge(agent,world)
        
        for agent in world.agents:
            if agent.name == 'SB1':
                print('Time Step', world.time)
                print('Energy SB1', agent.state.c* agent.demands[world.time-1])
                print('Demand SB1', agent.demands[world.time-1])
                print('Charge', dCharge)
                
            if agent.name == 'SB2':
                print('Time Step', world.time)
                print('Eneryg SB2', agent.state.c* agent.demands[world.time-1])
                print('Demand SB2', agent.demands[world.time-1])
                print('Charge', dCharge)
                #print('SB2', self.SB2_reward(agent,world))
        #print('States:', [a.state.c for a in world.agents])
        #print('Agent Demand', [a.demands for a in world.agents])
        #print("Energy Both: ", [a.state.c * a.demands for a in world.agents])
        #print('Demand Charge', dCharge) 
        #print('cost', self.SB1_reward(agent,world)+self.SB2_reward(agent,world)+ dCharge)
        print('*' *25)
        

        return self.SB1_reward(agent,world) + self.SB2_reward(agent,world) + dCharge
    
    def SB1_reward(self, agent, world):
        print('SB1 REWARD', file=fileOut)
        #print('SB1 time', world.time)
        cost = 0
        for a in world.agents:
            if a.name == 'SB1':
                if 0 in agent.state.c:
                    cost = -np.sum((a.state.c*a.demands[world.time-1]) + 10*(a.demands[world.time-1]-(a.state.c*a.demands[world.time-1]))**2)
                else: 
                    cost = -np.sum((a.state.c* a.demands[world.time-1]) + (a.demands[world.time-1]- (a.state.c* a.demands[world.time-1]))**2)
                #print('SB1 Energy', a.state.c* a.demands[world.time-1] )
        return cost

    def SB2_reward(self, agent, world):
        print('SB2 REWARD', file=fileOut)
        #print('SB2 time', world.time)
        cost = 0
        for a in world.agents:
            if a.name == 'SB2':
                if 0 in agent.state.c:
                    cost = -np.sum((a.state.c* a.demands[world.time-1]) + 10*(a.demands[world.time-1]- (a.state.c* a.demands[world.time-1]))**2 )
                else:
                    cost = -np.sum((a.state.c* a.demands[world.time-1]) + (a.demands[world.time-1]- (a.state.c* a.demands[world.time-1]))**2 )
                #print('SB2 Energy', a.state.c* a.demands[world.time-1] )
        return cost
    

    def observation(self,agent,world):
        print('OBSERVATION', file=fileOut)
        agent.prev_energy = agent.energy    

        agentObs = []
        print('OBS world time', world.time)
        for agent in world.agents:
            if agent.name == 'SB1':
                agentObs.append(agent.state.c * agent.demands[world.time-1])
            if agent.name == 'SB2':
                agentObs.append(agent.state.c * agent.demands[world.time-1])
        
        #High = []
        #High.append(self.demandCharge(agent,world))
  
        #print('1', High)
        print('2', agentObs)

        return np.concatenate(agentObs)

    def done(self, agent, world):
        for agents in world.agents:
            if world.time == 3:
                #self.reset(world)
                return True 
            else:
                return False

    