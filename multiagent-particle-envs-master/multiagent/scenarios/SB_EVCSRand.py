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
save_path = "/Users/Patrick Wilk/Documents/RL/MADDPG/results/" + '2SBnEV'



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

        self.optimal = []
        self.nonoptimal = []
        self.MAddpg = []

        self.timeReset = 1
        self.changeRate = 50
        self.inc = 1
        self.sample = 0
        self.graphing_rate  = 2*30
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
                agent.demands =[3,1,1]
            elif i == 1: 
                agent.name = 'EV'
                agent.required = 2

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

        if self.sample % 100 == 0:
            self.sample = 0
            self.totalCost = []
            self.low = []
            self.high = []
        #filling out agent detail
        for agent in world.agents:
            if agent.name == "SB":
                if self.timeReset % self.changeRate == 0:
                    agent.demands = [3,1,1]
                    agent.demands = agent.demands + np.random.uniform(-0.3,0.3,3)
                agent.min = 1
                agent.energy = 0
                agent.color = np.array([0.5,0.5,0.5])
                agent.state.p_pos = np.array([-.2,0.2])
                agent.agent_callback = None
            elif agent.name == "EV":
                if self.timeReset % self.changeRate == 0:
                    agent.required = 2
                    agent.required = agent.required + np.random.uniform(-0.3,0.3,1)
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

        self.timeReset += 1
        #filling out landmark detail
        for landmark in world.landmarks:
            if landmark.name == "Load":
                landmark.state.demand = [3,1,1]
                #landmark.color = np.array([0.1+self.load/self.peak/0.9, 1-self.load/self.peak,0])
                landmark.state.p_pos = np.array([-.3,-.5])

  #reward method for calling. Deliberates to specific reward functions

    def reward(self,agent,world):

        if self.inc % self.graphing_rate == 0:
            self.sample += 1
            MADCost = self.SB_EV_reward(agent,world)

            OrigOpt = self.optimizer(agent,world)
            #print( 'VALUES', OrigOpt)

            self.totalCost.append(MADCost)
            self.low.append(-OrigOpt[0])
            self.high.append(-OrigOpt[1])
            print("Ckecking HERE", -OrigOpt[0], OrigOpt[1])
            print("MAD COSt", MADCost)

            self.optimal.append(sum(self.low)/(self.sample))   #this should be the length of the array to find the average 
            self.nonoptimal.append(sum(self.high)/self.sample)
            self.MAddpg.append(sum(self.totalCost)/self.sample)
      

            plt.figure(2)
            #plt.plot(range(len(self.totalCost)), self.totalCost)
            #plt.plot(range(len(self.low)), self.low)
            #plt.plot(range(len(self.high)), self.high)
            plt.plot(range(len(self.optimal)), self.optimal)
            plt.plot(range(len(self.nonoptimal)), self.nonoptimal)
            plt.plot(range(len(self.MAddpg)), self.MAddpg)
            
            plt.xlabel('episodes * ')
            plt.ylabel('Average Costs')
            plt.savefig(save_path + '/plt2.png', format='png')

        self.inc +=1        
        #agents are rewarded based on the minimal cost compared to demand
        return self.SB_EV_reward(agent,world) 
    
    def SB_EV_reward(self, agent, world):
        cost = 0
        for a in world.agents:
            #if a.name == "SB" and 0 in a.state.c:
                #SBenergy = 10+(a.state.c*a.demands)
                #SBpenalty = (a.demands- SBenergy)**2
                #print('less than 0')
            if a.name == "SB":
                SBenergy = (a.state.c* a.demands)
                SBpenalty = (a.demands- SBenergy)**2            
            if a.name == "EV":
                EVenergy = a.state.c*a.maxcharge
                EVpenalty = 2*((a.required - (sum(EVenergy)))**2)
                

        #totalEnergy = np.add(SB1energy,SB2energy)
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
            
  
    def observation(self,agent,world):
        agent.prev_energy = agent.energy                 #Not alll arrays have the same dimension since [[]] 

        #agentObs = []
        #for agent in world.agents:
            #agentObs.append(agent.state.c * agent.demands)
        entityDemand = []
        for agent in world.agents:
            if agent.name == 'SB':
                entityDemand.append(agent.demands)

        #Demand2 = []
        #for agent in world.agents:
            if agent.name == 'EV':
                entityDemand.append(agent.required)

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
                energyEV = (agent.state.c*agent.maxcharge)
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
        print('3', entityDemand)
        #print('4', Demand2)
        #print('3', agentStateAc)
        #print('4', agentSum )

        return np.concatenate(agentStateAc + entityDemand  + [maxCharge] )#(agentObs + entityDemand+ agentStateAc + agentSum)


    def optimizer(self,agent,world):
        for agent in world.agents:
            if agent.name == "SB":
                demand1 = agent.demands
            if agent.name == "EV":
                demand3 = agent.required
                charge = agent.maxcharge

        timeStep = len(demand1)

        load1 = cp.Variable(timeStep)
        load3 = cp.Variable(timeStep)

        constraints = [0 <= load1, 
                    demand3 == cp.sum(load3) ]

            ############## CALCULATIONS ###############
        SB1deltaPenalty = demand1 - load1
            #demandCharge = np.maximum(load1,load2)
            ################ SOLVER ##################
        timeLoad = load1 + load3   
        Dcharge = cp.max(timeLoad)
        
            #function = load1**2 + SB1deltaPenalty + load2**2 + SB2deltaPenalty + demandCharge1 + demandCharge2
        function =  2*Dcharge + cp.sum(load1)  + cp.sum(load3) + cp.sum(SB1deltaPenalty**2) 

        objective = cp.Minimize(function)
            #objective = cp.Minimize(function)

        prob = cp.Problem(objective, constraints)

        prob.solve()
        
        demands13 = []
        evSOE = demand3
        while evSOE >= charge and len(demands13)<=3:
            demands13.append(charge)
            evSOE = evSOE - charge
        else:
            if len(demands13) >= 3: 
                demands13 = demands13[0:3]
            while evSOE < charge and len(demands13) < 3: 
                demands13.append(evSOE)
                evSOE = evSOE - evSOE
        
        OriginalPrice = 2*(np.max(np.add(demand1 , demands13))) + np.sum(demand1) + + np.sum(demand3)

        #print("Solution type: ", prob.status)
        #print("-"*25)
        #print("Minimized value: ", round(prob.value, 3))
        #print("Optimal1 value: ", list(load1.value))
        #print("Optimal2 value: ", list(load2.value))
        #print("Original Price", OriginalPrice)
    
        return ([prob.value, OriginalPrice])