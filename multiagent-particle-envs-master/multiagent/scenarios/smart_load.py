import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random
import matplotlib.pyplot as plt
import pandas as pd
import os



class Scenario(BaseScenario):
    def __init__(self):
        self.day_reward = True
        self.method = "main"
        
    def make_world(self):
        world = World()

        #used for plotting actions done by policy. 
        #Future work: implementing it so that it can work with any number of input dimensions and entities
        world.actions = [[],[]]

        '''
        Scenario Properties
        '''

        num_agents = 2
        num_adversaries = 0
        world.dim_c = 1
        world.dim_p = 2

        world.collaborative = True

        self.load = 0
        self.peak = 2
        if self.method != "main":
            self.peak = 2
        self.done = False
        world.time = 0
        self.day_reward = True
        self.start_required = 2
        
        #Generate Agents
        world.agents = [Agent() for i in range(num_agents)]

        for i, agent in enumerate(world.agents):
            if i == 0:
                agent.name = 'Smart_Building'
            elif i == 1:
                agent.name = "Charging_Station"
            agent.silent = False
            agent.movable = False
            agent.size = .1
        


        #Generate Landmarks
        world.landmarks = [Landmark() for i in range(3)]
        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.name = "Load"
                landmark.size = .2
            elif i==1:
                landmark.name = "energy cost"
                landmark.size = .1
            elif i ==2:
                landmark.name = "comfort"
                landmark.size = .1
            landmark.collide = False
            landmark.movable = False
            landmark.boundary = False

                
        self.reset_world(world)
        return world


    #reward method for calling. Deliberates to specific reward functions
    def reward(self, agent, world):
        reward = 0


        if agent.name == "Smart_Building":
            reward = self.smart_building_reward(agent, world)
            '''
            reward -= (agent.demands[world.time-1]-agent.energy)**2
            reward -= self.load**2
            self.reward 
            '''
        elif agent.name == "Charging_Station":
            reward = self.charging_station_reward(agent, world)
            '''
            if world.time-1 == 2 and agent.required > 0.33:
                print(agent.required)
                reward -= 200
            '''
            pass
            
        reward -= ((self.load**2)/2)

        if (self.day_reward == False):
            return reward
        elif(self.day_reward == True and world.time-1 == 2):
            self.total_reward += reward
            tmp = self.total_reward
            return tmp
        else:
            self.total_reward += reward
            return 0

    #Reward functions used from 
    def smart_building_reward(self, agent, world):
        reward = 0
        '''
        if self.occupation[world.time-1] == 1:
            reward -= self.cost(world)* max(agent.energy, 0) - agent.comfort_coef*((self.peak - agent.energy)**2)
        else:
            reward -= self.cost(world)* max(agent.energy, 0)
        '''
        reward -= (agent.demands[world.time-1] - agent.energy)**2
        return reward

    def charging_station_reward(self, agent, world):
        reward = 0
        reward -= (agent.required)**2
        #reward -= abs(agent.energy-agent.prev_energy)*50
        return reward


    def reset_world(self, world):
        self.total_reward = 0
        world.time = 0
        self.load = 0
        self.comfort = 0
        self.done = False
        world.energy_costs = []
        self.total_reward = 0
        
        #filling out agent detail
        for agent in world.agents:
            if agent.name == "Smart_Building":
                agent.demands = [3,1,1]
                agent.min = 1
                agent.energy = 0
                agent.color = np.array([0.5,0.5,0.5])
                agent.state.p_pos = np.array([-.2,0.2])
                agent.agent_callback = None
                pass
            elif agent.name == "Charging_Station":
                agent.rates = [i for i in range(self.peak)]
                agent.energy = 0
                agent.occupied = True
                #charging deadline. after deadline, penalty is severe over more time
                agent.required = 2 #2
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
    def observation(self, agent, world):
        agent.prev_energy = agent.energy
        if (self.method != "main"):
            return self.rule(agent, world)

        if agent.name == "Charging_Station":
            world.actions[0].append(agent.state.c)
            agent.energy= agent.state.c[0] * 2
            agent.required -= agent.energy 
        elif agent.name  == "Smart_Building":
            world.actions[1].append(agent.state.c)
            agent.energy = agent.state.c[0] * agent.demands[world.time-1]

        self.load = self.new_load(world)
        return([self.load])



    def done(self,agent, world):
        if world.time == 3:
            return True
        else:
            return False
        

    

        
    def smart_buildings(self, agent, world):
        return [agent for agent in world.agents if (agent.name == "Smart_Building")]

    def charging_stations(self, agent, world):
        return [agent for agent in world.agents if (agent.name == "Charging_Station")]
    
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
            if agent.name == "Charging_Station":
                if world.time-1 != 2:
                    agent.energy = agent.rates[-1]
                    agent.required -= agent.energy
                else:
                    agent.energy = 0
            if agent.name == "Smart_Building":
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
                if agent.name =="Charging_Station":
                    agent.energy = 0.66
                if agent.name =="Smart_Building":
                    agent.energy = 1.5
            if world.time-1 == 1:
                if agent.name =="Charging_Station":
                    agent.energy = 0.66
                if agent.name =="Smart_Building":
                    agent.energy = .5
            if world.time-1 == 2:
                if agent.name =="Charging_Station":
                    agent.energy = 0.66
                if agent.name =="Smart_Building":
                    agent.energy = 0.5
        if agent.name == "Charging_Station":
            agent.required-= agent.energy
        pass

    
    '''Gets the energy cost from multiple features attached to the world'''
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

        


