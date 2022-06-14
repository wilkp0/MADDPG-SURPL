import numpy as np
from tqdm import tqdm
import cvxpy as cp
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")





class Scenario(BaseScenario):

    def __init__(self):
        #self.day_reward = True
        self.method = "main"

    def make_world(self):

        sourceDir = "/Users/Patrick Wilk/Documents/RL/JJ_first_Look/MADDPG_DPR/SURPL_MARL"
        fileOut = open(sourceDir + "/results/MADDPG2.txt", "w+")
        figDir = sourceDir + '/results/plt_figs'
        self.save_path = "/Users/Patrick Wilk/Documents/RL/MADDPG/results/" + 'smart_loadsRandom'

        world = World()
        self.time_steps = 3*15000
        self.graphing_rate  = 2*30
        self.totalCost = []
        self.low = []
        self.high = []
        self.evaluate_rate=3*30
        self.episode_limit = 1
        #used for plotting actions done by policy. 
        #Future work: implementing it so that it can work with any number of input dimensions and entities
        world.actions = [[],[]]
        self.timeReset = 1
        self.changeRate = 10
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
                agent.name = 'SB1'
                agent.demands = [3,1,1]
            elif i == 1:
                agent.name = 'SB2'
                agent.demands = [3,1,1]
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
                if self.timeReset % self.changeRate == 0:
                        agent.demands = np.random.uniform(0,3,3)
                agent.min = 1
                agent.energy = 0
                agent.color = np.array([0.5,0.5,0.5])
                agent.state.p_pos = np.array([-.2,0.2])
                agent.agent_callback = None
            if agent.name == "SB2":
                if self.timeReset % self.changeRate == 0:
                    agent.demands = np.random.uniform(0,3,3)
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

        self.timeReset += 1
        print('TIME RESET', self.timeReset)
            
        #filling out landmark detail
        for landmark in world.landmarks:
            if landmark.name == "Load":
                landmark.state.demand = [3,1,1]
                #landmark.color = np.array([0.1+self.load/self.peak/0.9, 1-self.load/self.peak,0])
                landmark.state.p_pos = np.array([-.3,-.5])
        


    def reward(self,agent,world):
        #agents are rewarded based on the minimal cost compared to demand

        energy = [a.state.c * a.demands for a in world.agents]
        demands = [a.demands for a in world.agents]
        #Usedenergy = (energy* self.demand1) # working
        Usedenergy1 = (energy[0])
        Usedenergy2 = (energy[1])

        #dCharge = np.add(Usedenergy[0],Usedenergy[1])
        dCharge = np.add(Usedenergy1,Usedenergy2)
        dCharge = max(dCharge)

        #print('Energy1', Usedenergy1) 
        #print ('Energy2', Usedenergy2) #,  file=self.fileOut)
        #comfort = np.subtract([3,1,1],Usedenergy)
        comfort1 = np.subtract(demands[0],Usedenergy1)
        comfort2 = np.subtract(demands[1],Usedenergy2)

        #print('1comfort', comfort1)
        #print('2comfort', comfort2)
        comfort = np.concatenate([comfort1] + [comfort2])
        comfort = comfort **2
        #print('comf concat', comfort) #, file=self.fileOut)

        #print('concat', Usedenergy)
        Usedenergy = np.concatenate([Usedenergy1] + [Usedenergy2])
        Usedenergy = np.sum(Usedenergy)
        #print('UseedEnegy', Usedenergy, file=self.fileOut)
        #print('Comfort', comfort, file=self.fileOut)
        #print('DemandCharge', dCharge, file=self.fileOut)
        
        Cost = Usedenergy + sum(comfort) + 2*dCharge 
        #print('FCost', Cost, file=self.fileOut)
        #print('*'*25, file=self.fileOut)

        OrigOpt = self.optimizer(agent,world)
        print( 'VALUES', OrigOpt)

        self.totalCost.append(-Cost)
        self.low.append(OrigOpt[0])
        self.high.append(OrigOpt[1])
        plt.figure(10)
        plt.plot(range(len(self.totalCost)), self.totalCost)
        plt.plot(range(len(self.low)), self.low)
        plt.plot(range(len(self.high)), self.high)
        #plt.ylim(-24,-12)
        plt.xlabel('episodes * ' + str(self.evaluate_rate / self.episode_limit))
        plt.ylabel('Costs')
        plt.savefig(self.save_path + '/plt2.png', format='png')

        return self.SB_reward(agent,world) 
    
    def SB_reward(self, agent, world):
        cost = 0

        dCharge = np.array([(a.state.c*a.demands) for a in world.agents])
        dCharge = max(np.add(dCharge[0], dCharge[1]))
        
        if 0 in agent.state.c:
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

        return np.concatenate(agentObs+ [High] )


    def optimizer(self,agent,world):
        for agent in world.agents:
            if agent.name == "SB1":
                demand1 = agent.demands
            if agent.name == "SB2":
                demand2 = agent.demands

        timeStep = len(demand1)

        load1 = cp.Variable(timeStep)
        load2 = cp.Variable(timeStep)

        constraints = [0 <= load1, 
                    0 <= load2 ]

            ############## CALCULATIONS ###############
        SB1deltaPenalty = demand1 - load1
        SB2deltaPenalty = demand2 - load2
            #demandCharge = np.maximum(load1,load2)
            ################ SOLVER ##################
        timeLoad = load1 + load2   
        Dcharge = cp.max(timeLoad)
        
            #function = load1**2 + SB1deltaPenalty + load2**2 + SB2deltaPenalty + demandCharge1 + demandCharge2
        function =  2*Dcharge + cp.sum(load1) + cp.sum(load2) + cp.sum(SB1deltaPenalty**2) + cp.sum(SB2deltaPenalty**2)

        objective = cp.Minimize(function)
            #objective = cp.Minimize(function)

        prob = cp.Problem(objective, constraints)

        prob.solve()

        OriginalPrice = 2*(np.max(np.add(demand1 , demand2))) + np.sum(demand1) + np.sum(demand2)

        print("Solution type: ", prob.status)
        print("-"*25)
        print("Minimized value: ", round(prob.value, 3))
        print("Optimal1 value: ", list(load1.value))
        print("Optimal2 value: ", list(load2.value))
        print("Original Price", OriginalPrice)
    
        return ([prob.value, OriginalPrice])
