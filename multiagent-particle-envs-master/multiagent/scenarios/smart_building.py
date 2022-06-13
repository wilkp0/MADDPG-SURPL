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

        num_agents = 2
        world.num_agents = num_agents
        world.dim_c = 1 
        world.dim_p = 2
        num_landmarks = 3
        num_adversaries = 0

        self.SB1CostList = []
        self.SB2CostList = []
        self.totalSBsCostsList = []
        self.nonScheduled = []
        self.minOptimal = []
        self.demandChargeList = []

        world.collaborative = True
        self.deltaUtilization = 0 
        self.comfort = 0 
        self.load = 0
        self.peak = 2
        world.timeWindow = 2 
        if self.method != "main":
            self.peak = 2
        self.done = False
        world.time = 0
        #self.day_reward = True
        self.start_required = 2

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
            elif i == 1:
                landmark.name = 'energy cost'
                landmark.size = .1
            elif i == 2:
                landmark.name = 'comfort'            
                landmark.size = .1
            landmark.collide = False
            landmark.movable = False
            landmark.boundary = False

        # make initial conditions
        self.reset_world(world)
        return world

    #reward method for calling. Deliberates to specific reward functions
    def reward(self, agent, world):
        reward = 0

        if world.time-1 < world.timeWindow:
            if agent.name == "SB1":
                reward = self.smart_building_reward(agent, world)
                self.demandChargeList.append(agent.energyList[world.time-1])
                print("SB1 reward: ", reward)
                
  
            elif agent.name == "SB2":
                reward = self.smart_building_reward(agent, world)
                self.demandChargeList.append(agent.energyList[world.time-1])
                print("SB2 reward: ", reward)
                print("*"*25)
            print('Added Reward:', reward)
            return reward

        if world.time -1 == world.timeWindow:

            if agent.name == "SB1":
                reward = self.smart_building_reward(agent, world)
                self.demandChargeList.append(agent.energyList[world.time-1]) 
                print("SB1 reward: ", reward)
  
            elif agent.name == "SB2":
                reward = self.smart_building_reward(agent, world)
                self.demandChargeList.append(agent.energyList[world.time-1])
                print("SB2 reward: ", reward)
                print("*"*25, )
                demandCharge = max(self.demandChargeList)
                print('Demand charge' ,demandCharge)
                print('ENERGY LIST', self.demandChargeList)
                reward = reward - 2*demandCharge
                print('Demand Charge', demandCharge)
                print('Total reward', reward)

            return reward

        # reward -= ((self.load**2)/2)
    '''
        if (self.day_reward == False):
            return reward
        elif(self.day_reward == True and world.time-1 == 2):
            self.total_reward += reward
            tmp = self.total_reward
            return tmp
        else:
            self.total_reward += reward
            return 0
    '''

    #Reward functions used from 
    def smart_building_reward(self, agent, world):
        '''
        if self.occupation[world.time-1] == 1:
            reward -= self.cost(world)* max(agent.energy, 0) - agent.comfort_coef*((self.peak - agent.energy)**2)
        else:
            reward -= self.cost(world)* max(agent.energy, 0)
        '''
        reward = 0
        #reward -= (agent.demands[world.time-1] - agent.energy)**2
        # print(world.time)
        # print(world.time, file=fileOut)
        agent.energyList.append((agent.action.c[0] * agent.demands[world.time-1]))
        self.deltaUtilization = abs(agent.demands[world.time -1] - agent.energyList[world.time -1])
        self.comfort = (self.deltaUtilization) ** 2
        

        if world.time-1 == world.timeWindow:
            # print("1", file=fileOut)
            #agent.demandCharge = max(agent.energyList)
            #reward -= agent.energyList[world.time-1] + self.comfort + 2*(agent.demandCharge)
            reward -= agent.energyList[world.time-1] + self.comfort 
        
        # IF HAVEN'T REACHED END OF SIMULATION YET
        elif world.time-1 < world.timeWindow and agent.energyList[world.time-1] != 0:
            # print("2", file=fileOut)
            reward -= agent.energyList[world.time-1] + self.comfort

        else:
            # print("3", file=fileOut)
            reward -= agent.energyList[world.time-1] + 1 + self.comfort

        print("*"*40)
        print('*'*12 + "Time Step", world.time-1)
        print("Action Taken 1: ", agent.action.c[0], "\n") if agent.name == "SB1" else print("Action taken 2: ", agent.action.c[0], "\n")
        print("Demand list 1:", agent.demands[world.time-1], '\n') if agent.name == "SB1" else print ("Demand list 2:", agent.demands[world.time-1], "\n")
        print("Energy list 1: ", agent.energyList, "\n") if agent.name == "SB1" else print("Energy list 2: ", agent.energyList, "\n")
        print("Energy list 1: ", agent.energyList[world.time-1], "\n") if agent.name == "SB1" else print("Energy list 2: ", agent.energyList[world.time-1], "\n")
        print("Delta 1:", self.deltaUtilization, '\n') if agent.name == "SB1" else print ("Delta 2:", self.deltaUtilization, "\n")
        print("Comfort 1:", self.comfort, '\n') if agent.name == "SB1" else print ("Comfort 2:", self.comfort, "\n")
        print("Demand Charge 1:", agent.demandCharge, '\n') if agent.name == "SB1" else print ("Demand Charge 2:", agent.demandCharge, "\n")        
        print("*"*40)
        print("Reward", reward, "\n")


        if len(agent.energyList) >= 3 and agent.name == "SB1":
            comfortSB1 = np.subtract( agent.demands,agent.energyList)
            print('Demand-Energy1',comfortSB1)
            comfortPenaltySB1 = sum(i**2 for i in comfortSB1)
            #self.SB1CostList.append(sum(agent.energyList) + 2*max(agent.energyList) + comfortPenaltySB1)
            self.SB1TCost = (sum(agent.energyList)  + comfortPenaltySB1)
            print('sb1 cost', self.SB1TCost)


        if len(agent.energyList) >= 3 and agent.name == "SB2":
            comfortSB2 = np.subtract( agent.demands,agent.energyList)
            print('Demand2-Energy2',comfortSB2)
            comfortPenaltySB2 = sum(i**2 for i in comfortSB2)
            #self.SB2CostList.append(sum(i for i in agent.energyList) + 2*max(agent.energyList) + comfortPenaltySB2)
            self.SB2TCost= (sum(i for i in agent.energyList) + 2*max(self.demandChargeList) + comfortPenaltySB2)
            print('sb2 cost' ,self.SB2TCost)
            self.nonScheduled.append(self.noSchedule(agent))
            self.minOptimal.append(self.optimizer(agent))
            self.sumSBs = self.SB1TCost + self.SB2TCost
            #self.totalSBsCostsList = np.add(self.SB1CostList, self.SB2CostList)
            self.totalSBsCostsList.append(self.sumSBs) 
            #print('This is the total cost : ', self.totalSBsCostsList)
        '''
        fig5 = plt.figure(5, figsize = (15,7))
        #tc1 = [range(len(self.SB1CostList))]
        #tcc1 = np.reshape(tc1, (len(self.SB1CostList),))
        #tc2 = [range(len(self.SB2CostList))]
        #tcc2 = np.reshape(tc2, (len(self.SB2CostList),))
        tc5 = [range(len(self.totalSBsCostsList))]
        tcc5 = np.reshape(tc5, (len(self.totalSBsCostsList),))

        tc3 = [range(len(self.nonScheduled))]
        tcc3 = np.reshape(tc3, (len(self.nonScheduled),))
        tc4 = [range(len(self.minOptimal))]
        tcc4 = np.reshape(tc4, (len(self.minOptimal),))

        plt.plot(tcc3, self.nonScheduled, color="brown", label="SB1", linestyle="solid", zorder=0)
        plt.plot(tcc4, self.minOptimal, color="green", label="SB1", linestyle="solid", zorder=15)
        plt.plot(tcc5, self.totalSBsCostsList, color="red", label="SB1", linestyle="solid", zorder=15)         
        #plt.plot(tcc1, self.SB1CostList, color="red", label="SB1", linestyle="solid", zorder=5)
        #plt.plot(tcc2, self.SB2CostList, color="blue", label="SB2", linestyle="dashed", zorder=10)
  
        plt.ylabel("Cost")
        plt.xlabel("Time Step") 
        fig5.savefig(figDir + '/RewardPlot'+'.png')
        '''


        return reward


    def reset_world(self, world):
        self.total_reward = 0
        world.time = 0
        world.timeWindow = 2
        self.load = 0
        self.done = False
        world.energy_costs = []
        self.total_reward = 0

        #filling out agent detail
        for agent in world.agents:
            if agent.name == "SB1":
                self.comfort = 0
                agent.demandCharge = 0
                agent.demands = [3,1,1]
                agent.min = 1
                agent.energy = 0
                agent.energyList = []
                agent.color = np.array([0.5,0.5,0.5])
                agent.state.p_pos = np.array([-.2,0.2])
                agent.agent_callback = None
                pass
            elif agent.name == "SB2":
                self.comfort = 0
                self.demandChargeList = []
                agent.demandCharge = 0
                agent.demands = [1,1,3]
                agent.min = 1
                agent.energy = 0
                agent.energyList = []
                agent.state.p_pos = np.array([-.0,0.2])
                agent.color = np.array([0.8,0.5,0.8])
                agent.agent_callback = None
                pass
            agent.state.c = np.zeros(world.dim_c)
            agent.action.c = np.zeros(world.dim_c)
            pass

        #filling out landmark detail
        for landmark in world.landmarks:
            if landmark.name == "Load":
                landmark.color = np.array([0.1+self.load/self.peak/0.9, 1-self.load/self.peak,0])
                landmark.state.p_pos = np.array([-.3,-.5])
                pass
            elif landmark.name == "energy cost":
                landmark.state.p_pos = np.array([0,-0.5])
                landmark.color = np.array([0.8,0.5,0.8])
                pass
            elif landmark.name == "comfort":
                landmark.state.p_pos = np.array([.2,-.5])
                landmark.color = np.array([0.1,0.5,0.8])
                pass



    '''
    Main observation method. Same as OpenAI observation.
    Obs - should be returning multiple 
    '''
    '''
    def observation(self, agent, world):
        agent.prev_energy = agent.energy
        #observation = agent.demands[world.time]
        if (self.method != "main"):
            return self.rule(agent, world)

        if agent.name == "SB1":
            SB1energy =[]
            world.actions[0].append(agent.state.c)
            # agent.energy = agent.energyList.append(agent.action.c[0]) * agent.demands[world.time-1]
            agent.energy = agent.state.c[0] * agent.demands[world.time-1]
            SB1energy.append(agent.energy)
            print("SB1 state: ", SB1energy, file=fileOut)
            # agent.energyList.append(agent.energy)
            # agent.energyList = agent.energyList[:2] if len(agent.energyList) >= 4 else agent.energyList
        elif agent.name  == "SB2":
            SB2energy = []
            world.actions[1].append(agent.state.c)
            # agent.energy = agent.energyList.append(agent.action.c[0]) * agent.demands[world.time-1]
            agent.energy = agent.state.c[0] * agent.demands[world.time-1]
            SB2energy.append(agent.energy)
            print("SB2 state: ", SB2energy, file=fileOut)
            # agent.energyList.append(agent.action.c[0])
            # agent.energyList.append(agent.energy)
            # agent.energyList = agent.energyList[:2] if len(agent.energyList) >= 4 else agent.energyList
        
        
        return np.concatenate(SB1energy + SB2energy)

    '''

    def observation(self,agent,world):
        agent.prev_energy = agent.energy

        if (self.method != 'main'):
            return self.rule(agent.world)

        agentObs = []
        for agent in world.agents:
            agentObs.append(agent.state.c * agent.demands[world.time-1])


        return np.concatenate(agentObs)

    

    def done(self,agent, world):
        if world.time == 3:
            return True
        else:
            return False


    def smart_building1(self, agent, world):
        return [agent for agent in world.agents if (agent.name == "SB1")]

    def smart_building2(self, agent, world):
        return [agent for agent in world.agents if (agent.name == "SB2")]




    # NO NEEDED THIS IS COMPARING THE OTHER 2 SCENARIOS
    '''
    Rule-based methods for environment.
    For future work, should either add custom policy or 
    max - maximum amount of power is distributed evenly among agents
    half - half of the max is distributed evenly among agents
    min - the minimum amount of energy is distributed to agents
    '''
    def rule(self, agent, world):
        if self.method == "max":
            self.no_sched(agent, world)
        elif self.method =="individual":
            self.individual(agent,world)
        return [0,0]


    def no_sched(self,agent, world):
        for agent in world.agents:
            if agent.name == "SB":
                agent.energy = agent.demands[world.time-1]
            if agent.name == "SB":
                agent.energy = agent.demands[world.time-1]


        self.load = 0
        for agent in world.agents:
            self.load += agent.energy

        if world.time-1 == 0:
            assert(self.load ==4)
        elif world.time-1 == 1:
            assert(self.load == 2)
        elif world.time-1 == 2:
            assert(self.load== 1)
    pass

    def individual(self,agent, world):
        for agent in world.agents:
            if world.time-1 == 0:
                if agent.name =="SB1":
                    agent.energy = 1.5
                if agent.name =="SB2":
                    agent.energy = 1.5
            if world.time-1 == 1:
                if agent.name =="SB1":
                    agent.energy = 0.5
                if agent.name =="SB2":
                    agent.energy = .5
            if world.time-1 == 2:
                if agent.name =="SB1":
                    agent.energy = 0.5
                if agent.name =="SB2":
                    agent.energy = 0.5
        pass


    def cost(self, world):
        if self.load <= self.peak/3:
            cost = self.load
        elif self.load <= self.peak/3 * 2:
            cost = self.load**2
        else:
            cost = self.load**3
        return cost

    def new_load(self, world):
        load = 0
        for agent in world.agents:
            load += agent.energy
        return load 



    def noSchedule(self,agent):

        demandSB1 = [1,1,3]
        demandSB2 = [3,1,1]

        demandMax = np.maximum(demandSB1, demandSB2)
        demandCharge = max(demandMax)
        cost =  sum(demandSB1) + sum(demandSB2) + 2*demandCharge

        return cost



    def optimizer(self,agent):
    ################# GLOBAL #################

        demand1 =  [1, 1, 3]
        demand2 =  [3, 1, 1]

        timeStep = len(demand1)
        Charge = [0,0,1]

        #Maxs = np.maximum(load1, load2)
        #demandCharge = max(Maxs)
        #demandCharge = cp.Parameter(demandCharge)
        #demandCharge = cp.Variable(len(demand1))


        ############ SCALAR VARIABLES ############
        load1 = cp.Variable(timeStep)
        load2 = cp.Variable(timeStep)

        demandCharge = cp.Variable(1)
        #demandCharge1 = cp.Variable(timeStep)
        #demandCharge2 = cp.Variable(timeStep)
        ############## CONSTRAINTS ################
        constraints = [0 <= load1, 
                        0 <= load2, 
                        load1 <= demandCharge, load2 <= demandCharge]


        ################ SOLVER ##################


        ############## CALCULATIONS ###############
        SB1deltaPenalty = (demand1 - load1)**2
        SB2deltaPenalty = (demand2 - load2)**2
        #demandCharge = np.maximum(load1,load2)
        ################ SOLVER ##################
        
        #function = load1**2 + SB1deltaPenalty + load2**2 + SB2deltaPenalty + demandCharge1 + demandCharge2
        function = load1 + SB1deltaPenalty + load2 + SB2deltaPenalty+ 2*(demandCharge * Charge)

        objective = cp.Minimize(cp.sum(function))
        #objective = cp.Minimize(function)

        prob = cp.Problem(objective, constraints)

        prob.solve()


        #print("Solution type: ", prob.status)
        #print("-"*25)
        #print("Minimized value: ", round(prob.value, 3))
        #print("Optimal1 value: ", list(load1.value))
        #print("Optimal2 value: ", list(load2.value))
        #print("Demand Charge", demandCharge.value)
        #print ("Demand", demand1)
        #print("Demand Charge", demandCharge1.value)
        #print("Demand Charge", demandCharge2.value)
        loadF1= list(load1.value)
        loadF2= list(load2.value)
        penalty1 = list(np.subtract(demand1,load1.value))
        penalty2 = list(np.subtract(demand2,load2.value))
        #print('*' *25)
        #print( 'pen1', penalty1)
        #print( 'pen2', penalty2)
        demandC = demandCharge.value
        penalties = sum(p1**2 for p1 in penalty1) + sum(p2**2 for p2 in penalty2)
        cost = sum(i*2 for i in demandC) + sum( x for x in loadF1) + sum(y for y in loadF2) + penalties
        #print ('penalty', penalties)
        #print( 'Cost', cost)

        return cost

        