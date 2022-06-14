import numpy as np
from datetime import datetime

now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
#sourceDir = "/Users/Patrick Wilk/Documents/RL/JJ_first_Look/MADDPG_DPR/SURPL_MARL"
#fileOut = open(sourceDir + "/results/PPO1.txt", "w+")

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # entity can move / be pushed
        self.movable = False
        # color
        self.color = None
        # state
        self.state = EntityState()


# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self):
        self.time = 0
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3



    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
   

        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)    

           # if agent.name == 'SB1':
                #print('TimeStep', self.time, file=fileOut)
               # print("SB1 stepping: ", agent.energyList, file=fileOut)
                
                #print("SB1 Reward:" , )
            #elif agent.name == 'SB2':
               # print("SB2 stepping: ", agent.energyList, file=fileOut)
               # print("*"*25, file=fileOut)
               # print("*"*25, file=fileOut)

        self.time += 1

 

    #changes state action to the state communication
    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            #noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c #+ noise      

