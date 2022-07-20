import numpy as np
import cvxpy as cp
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

sourceDir = "/Users/Patrick Wilk/Documents/RL/JJ_first_Look/MADDPG_DPR/SURPL_MARL"
fileOut = open(sourceDir + "/results/MADDPG2.txt", "w+")
figDir = sourceDir + '/results/plt_figs'




class Scenario(BaseScenario):

    def __init__(self):
        #self.day_reward = True
        self.method = "main"

    def make_world(self):
        world = World()

        #used for plotting actions done by policy. 
        #Future work: implementing it so that it can work with any number of input dimensions and entities
        world.actions = [[],[]]

        num_agents = 2
        world.num_agents = num_agents
        world.dim_c = 8 
        world.dim_p = 2
        num_landmarks = 1
        num_adversaries = 0
        self.demandCharge = []

        world.collaborative = True

        if self.method != "main":
            self.peak = 2
        self.done = False
        world.time = 0

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
        self.total_reward = 0
        world.time = 0
        self.load = 0
        self.demandCharge =[]
        self.done = False
        world.energy_costs = []
        self.total_reward = 0

        #filling out agent detail
        for agent in world.agents:
            if agent.name == "SB1":
                agent.demands = [0.5,1,2,4,2,3,1,0.5]
                agent.min = 1
                agent.energy = 0
                agent.color = np.array([0.5,0.5,0.5])
                agent.state.p_pos = np.array([-.2,0.2])
                agent.agent_callback = None
            if agent.name == "SB2":
                agent.demands = [1,1,2,3,4,2,0.5,1]
                agent.min = 1
                agent.energy = 0
                agent.color = np.array([0.5,0.5,0.5])
                agent.state.p_pos = np.array([-.2,0.2])
                agent.agent_callback = None
            #agent.state.c =np.random.uniform(0,1, world.dim_c)
            #agent.state.c = np.zeros(world.dim_c)
            #agent.action.c = np.zeros(world.dim_c)
            agent.state.c = np.random.uniform(0, 1, world.dim_c)
            agent.action.c = np.random.uniform(0, 1, world.dim_c)
            

        #filling out landmark detail
        for landmark in world.landmarks:
            if landmark.name == "Load":
                landmark.state.demand = [1,1,2,3,2,1]
                #landmark.color = np.array([0.1+self.load/self.peak/0.9, 1-self.load/self.peak,0])
                landmark.state.p_pos = np.array([-.3,-.5])

  #reward method for calling. Deliberates to specific reward functions

    def reward(self,agent,world):
        #agents are rewarded based on the minimal cost compared to demand
        return self.SB_reward(agent,world) 
    
    def SB_reward(self, agent, world):
        #energy= agent.state.c* agent.demand    NOT WORKING CORRECLTY 
        cost = 0
        #for a in world.agents:
        dCharge = np.array([(a.state.c*a.demands) for a in world.agents])
        dCharge = max(np.add(dCharge[0], dCharge[1]))
        
        if 0 in agent.state.c:
            #for zero in a.state.c:
                #if zero  == 0:
            #cost = sum([-40 + sum(a.state.c* a.demands) for a in world.agents])
            #cost = sum([-40  for a in world.agents])
            cost = -2*dCharge + -np.sum([(a.state.c* a.demands) + 10*(a.demands- (a.state.c* a.demands))**2  for a in world.agents])
            print('less than 0')
        else:
            cost = -2*dCharge + -np.sum([(a.state.c* a.demands) + (a.demands- (a.state.c* a.demands))**2  for a in world.agents])
            print('Demand charge', dCharge)
            
        #print('Energy SB1', [a.state.c* a.demands if a in world.agents: agent.name == 'SB1' ])
        for agent in world.agents:
            if agent.name == 'SB1':
                print('Energy SB1', agent.state.c* agent.demands)
                print('SB1', cost)
            if agent.name == 'SB2':
                print('Eneryg SB2', agent.state.c* agent.demands)
                print('SB2', cost)
        print('States:', [a.state.c for a in world.agents])
        print('Agent Demand', [a.demands for a in world.agents])
        print("Energy Both: ", [a.state.c * a.demands for a in world.agents]) 
        print('cost', cost)
        print('*' *25)
        #cost = np.sum((energy) + (agent.demands- energy)**2 )
        return cost

    def observation(self,agent,world):
        agent.prev_energy = agent.energy      


        #agentStateAc = []
        #for agent in world.agents:
            #agentStateAc.append(agent.state.c)
        agentObs = []
        for agent in world.agents:
            agentObs.append(agent.state.c * agent.demands)
        #entityDemand = []
        #for agent in world.agents:
            #entityDemand.append(np.array(agent.demands))
        
        #agentSum = []
        #for agent in world.agents:
        #energy = [(agent.state.c * agent.demands) for agent in world.agents]
        #agentSum.append(np.add(energy[0],energy[1]))

        High = []
        energy = [(agent.state.c * agent.demands) for agent in world.agents]
        High.append(np.array(max(np.add(energy[0],energy[1]))))


        #print('1.1', agentStateAc)
        print('1', High)
        print('2', agentObs)
        #print('3', entityDemand)
        #print('4', agentSum )
        #print(np.concatenate([High] + agentObs + entityDemand + agentSum))

        return np.concatenate(agentObs+ [High]  )

