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
fileOut = open(sourceDir + "/results/PPO2.txt", "w+")
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

        num_agents = 1
        world.num_agents = num_agents
        world.dim_c = 3 
        world.dim_p = 2
        num_landmarks = 1
        num_adversaries = 0


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
        self.done = False
        world.energy_costs = []
        self.total_reward = 0

        #filling out agent detail
        for agent in world.agents:
            if agent.name == "SB1":
                agent.demands = [3,1,1]
                agent.min = 1
                agent.energy = 0
                agent.color = np.array([0.5,0.5,0.5])
                agent.state.p_pos = np.array([-.2,0.2])
                agent.agent_callback = None
            agent.state.c = np.zeros(world.dim_c)
            agent.action.c = np.zeros(world.dim_c)


        #filling out landmark detail
        for landmark in world.landmarks:
            if landmark.name == "Load":
                landmark.state.demand = [3,1,1]
                #landmark.color = np.array([0.1+self.load/self.peak/0.9, 1-self.load/self.peak,0])
                landmark.state.p_pos = np.array([-.3,-.5])

  #reward method for calling. Deliberates to specific reward functions
    def reward(self, agent, world):
        cost = 0
        x = agent.action.c
        y = agent.state.c 
        print('action',x)
        print('state', y)
        energy= agent.state.c* agent.demands 
        print('Energy', energy)
        dCharge = max(energy)
        cost = np.sum((energy) + (energy - agent.demands)**2 ) + 2*dCharge
        print("Demand Charge", dCharge )
        print("Cost", cost)
        return -cost




    def observation(self,agent,world):
        agent.prev_energy = agent.energy                                  

        agentObs = []
        for agent in world.agents:
            agentObs.append(agent.state.c * agent.demands)


        return np.concatenate(agentObs)

    

