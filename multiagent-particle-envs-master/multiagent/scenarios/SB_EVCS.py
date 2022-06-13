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
        world.dim_c = 3 
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
                agent.name = 'SB'
            elif i == 1:
                agent.name = 'EV'
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
            if agent.name == "SB":
                agent.demands = [3,1,1]
                agent.min = 1
                agent.energy = 0
                agent.color = np.array([0.5,0.5,0.5])
                agent.state.p_pos = np.array([-.2,0.2])
                agent.agent_callback = None
            if agent.name == "EV":
                agent.required = 2
                agent.maxcharge = 1
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
                landmark.state.demand = [3,1,1]
                #landmark.color = np.array([0.1+self.load/self.peak/0.9, 1-self.load/self.peak,0])
                landmark.state.p_pos = np.array([-.3,-.5])

  #reward method for calling. Deliberates to specific reward functions

    def reward(self,agent,world):
        #agents are rewarded based on the minimal cost compared to demand
        return self.SB_EV_reward(agent,world) 
    
    def SB_EV_reward(self, agent, world):
        cost = 0
        for a in world.agents:
            if a.name == "SB" and 0 in a.state.c:
                SBenergy = 25+(a.state.c*a.demands)
                SBpenalty = (a.demands- SBenergy)**2
                print('less than 0')
            elif a.name == "SB":
                SBenergy = (a.state.c* a.demands)
                SBpenalty = (a.demands- SBenergy)**2
            if a.name == "EV":
                EVenergy = a.state.c*a.maxcharge
                EVpenalty = 5*((a.required - (sum(EVenergy)))**2)

        totalEnergy = np.add(SBenergy,EVenergy)
        dCharge = max(totalEnergy)
        
        
        EVcost = sum(EVenergy)+ EVpenalty
        print('EV Cost', EVcost)

        
        SBcost = sum(SBenergy) + sum(SBpenalty)
        print('SB Cost', SBcost)

        print("SB Energy Var", SBenergy)
        print("EV Energy Var", EVenergy)
        print('*^'*25)


        cost = -(2*dCharge + SBcost + EVcost)
        print ('Cost: ', cost)
        return cost
            
        #print('Energy SB1', [a.state.c* a.demands if a in world.agents: agent.name == 'SB1' ])
        '''
        for agent in world.agents:
            if agent.name == 'SB':
                print('Energy SB', agent.state.c* agent.demands)
                print('SB', cost)
            if agent.name == 'EV':
                print('Eneryg EV', agent.state.c* agent.required)
                print('EV', cost)
        print('States:', [a.state.c for a in world.agents])
        print('Agent Demand', [a.demands for a in world.agents])
        print('Agent Required', [a.required for a in world.agents])
        #print("Energy Both: ", [a.state.c * a.demands if a == 'SB' and a.state.c * a.required if a == 'EV' for a in world.agents]) 
        print('cost', cost)
        print('*' *25)
        #cost = np.sum((energy) + (agent.demands- energy)**2 )
        return cost
        '''
    def observation(self,agent,world):
        agent.prev_energy = agent.energy                 #Not alll arrays have the same dimension since [[]] 

        #agentObs = []
        #for agent in world.agents:
            #agentObs.append(agent.state.c * agent.demands)
        #entityDemand = []
        #for agent in world.agents:
            #entityDemand.append(agent.demands)
        agentStateAc = []
        for agent in world.agents:
            if agent.name == 'SB':
                agentStateAc.append(agent.state.c*agent.demands)
            if agent.name == 'EV':
                agentStateAc.append(agent.state.c*agent.maxcharge)

        maxCharge = []
        for agent in world.agents:
            if agent.name == 'SB':
               energySB = (agent.state.c * agent.demands)
            if agent.name == 'EV':
                energyEV = (agent.state.c)
                maxCharge.append(max(np.add(energySB,energyEV)))


        #High = []
        #energy = [(agent.state.c * agent.demands) for agent in world.agents]
        #High.append(np.array(max(np.add(energy[0],energy[1]))))

        #agentSum = []
        #for agent in world.agents:
        #energy = [(agent.state.c * agent.demands) for agent in world.agents]
       # agentSum.append(np.add(energy[0],energy[1]))


        print('1', agentStateAc)
        print('2', maxCharge)
        #print('3', agentStateAc)
        #print('4', agentSum )

        return np.concatenate(agentStateAc + [maxCharge])#(agentObs + entityDemand+ agentStateAc + agentSum)

  