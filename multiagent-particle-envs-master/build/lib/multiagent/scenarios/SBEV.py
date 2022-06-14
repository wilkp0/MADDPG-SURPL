from turtle import clear
import os, sys, glob, shutil, re, random, time
from datetime import datetime
from auxFuncs import *

# global golbalTime 

now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
sourceDir = "/Users/Patrick Wilk/Documents/RL/JJ_first_Look/SURPL-V2-JJ_first_look/JJ_cust"
fileOut = open(sourceDir + "/results/PPO.txt", "w+")
figDir = sourceDir + "/plt_figs/"
# fileOut = open(sourceDir + "/results/DDPG.txt", "w+")
# fileOut = open(sourceDir + "/results/" + now + "/PPO", "w+")



# ------------------------------
#       SMART BUILDING
# ------------------------------
class SmartBuildingEnv(Env):
    def __init__(self):
        self.viewer = None
        self.load = []
        self.count = 0 
        self.iter = 0
        self.newDemand =[3,1,1]
        self.shuffleDemand = [3,1,1]
        self.demand = []

        self.optimalAverage = [] 
        self.calculatedAverage = []
        self.performAverage= []

        # self.demandCharge = 2 * self.load
        self.demandCharge = 0
        self.timeStep = 0
        self.deltaUtilization = 0
        self.penalty = 0
        self.totalReward = 0
        self.cost = 1
        self.totalCost = 0
        self.penaltyList =[]
        self.optimalCostList = []
        self.totalCostList = []
        self.totalOptimalCostList = []
        self.totalperformanceList = []

        self.action_space = None
        self.observation_space = None
        self.reset()
        
    def reset(self):
        # self.load = 0
        # MOVE TO 24 HOURS FOR INITAL RESULTS NOW THAT PRELIM RESULTS ESTABLISHED
        self.demand = self.newDemand
        self.resetDemand = self.shuffleDemand
        self.load = []
        self.timeWindow = 2
        self.cost = 1
        self.totalCost = 0
        self.penaltyList = []
        self.optimalCostList = []
        self.performanceList = []
        # self.timeWindow = len(self.demand) - 1
        self.timeStep = 0
        self.totalReward = 0
        # self.deltaUtilization = abs(self.demand[self.timeStep] - self.load[self.timeStep])
        self.deltaUtilization = 0
        self.penalty = (self.deltaUtilization) ** 2
        self.action_space = Box(low=0, high=4, shape=(1,), dtype="float32")
        # OBSERVATION SPACE CAN ALSO INCLUDE DEMAND CHARGE, TIME, ETC -> 1 INSUFFICIENT FOR MORE RESULTS
        # self.observation_space = Box(low=np.array([0, 0]), high=np.array([self.deltaUtilization, self.timeWindow]), shape=(2,), dtype=float)
        self.observation_space = Box(low=np.array([0, 0, 0]), high=np.array([self.deltaUtilization, 4, self.timeWindow]), shape=(3,), dtype="float32")
        # return np.array(0)
        return np.array([0, 0, 0])
        # pass
        # return state
    
    def step(self, action):
        info = {}
        reward = 0
        done:bool = False
        # action[0] = abs(action[0])
        self.load.append(action[0])
        self.demandCharge = max(self.load)

        print("SB stepping: ", round(self.load[self.timeStep], 2))
        print("*"*25)
        print("*"*25)
        print("SB stepping: ", round(self.load[self.timeStep], 2), file=fileOut)
        print("*"*25, file=fileOut)
        print("*"*25, file=fileOut)
        

        print("Timestep: ", self.timeStep)
        print("-"*5)
        print("Load: ", [round(i, 2) for i in self.load][self.timeStep])
        print("Demand: ", self.demand[self.timeStep])
        print("Timestep: ", self.timeStep, file=fileOut)
        print("-"*5, file=fileOut)
        print("Load: ", [round(i, 2) for i in self.load][self.timeStep], file=fileOut)
        print("Demand: ", self.demand[self.timeStep], file=fileOut)


        self.deltaUtilization = abs(self.demand[self.timeStep] - self.load[self.timeStep])
        self.penalty = (self.deltaUtilization) ** 2
        self.penaltyList.append(self.penalty)

        observation = self.demand[self.timeStep]
        
        print("Penalty: ", round(self.penalty, 2))
        print("Delta ", round(self.deltaUtilization, 2))
        print("Observation: ", observation)
        print("-"*25)
        print("Penalty: ", round(self.penalty, 2), file=fileOut)
        print("Delta ", round(self.deltaUtilization, 2), file=fileOut)
        print("Observation: ", observation, file=fileOut)
        print("-"*25, file=fileOut)
        
        # STILL NEED PEAK DEMAND IN REWARD
        # HOW DO WE CONTRIBUTE TO THIS REWARD?
               
        if self.timeStep == self.timeWindow:
            reward -= self.load[self.timeStep] + self.penalty + 2*(self.demandCharge)
        # IF HAVEN'T REACHED END OF SIMULATION YET
        elif self.timeStep < self.timeWindow and self.load[self.timeStep] != 0:
            reward -= self.load[self.timeStep] + self.penalty
            
        else:
            reward -= self.load[self.timeStep] + 1 + self.penalty
            
        # REWARD CAN BE DIFFERENCE BETWEEN COSTS
        
        self.timeStep += 1
        self.totalReward += reward
        done = True if self.timeStep > 2 else done
        print("Total Reward: ", round(self.totalReward, 2))
        print("-"*25)
        print("Total Reward: ", round(self.totalReward, 2), file=fileOut)
        print("-"*25, file=fileOut)
        print("-"*25, file=fileOut)
        print("-"*25)
        print("-"*25)
        if done:


            self.totalCost = sum([i * self.cost for i in self.load])+ sum(self.penaltyList) + 2*(max(self.load))
            self.totalCostList.append(self.totalCost)

            self.performanceList = calculatePerformanceSB(demand = self.demand)
            print (self.performanceList)
            self.totalperformanceList.append(sum(self.performanceList)+ 2*(max(self.performanceList)))

            self.optimalCostList = calculateSBOptimal(demand=self.demand)
            SubtractionPenalty = np.subtract(self.demand,self.optimalCostList)
            comfortPenalty = sum(i ** 2 for i in SubtractionPenalty)
            self.optimalCostList = [i * self.cost for i in self.optimalCostList] 
            self.totalOptimalCostList.append((sum(self.optimalCostList))+ 2*(max(self.optimalCostList)) + comfortPenalty)

            self.iter +=1
            if self.iter % 200 ==0:
                self.newDemand =  random.randrange(-1,2) * np.round(np.random.rand(3),3) + self.resetDemand
            elif self.iter % 1000 == 0: 
                shuffle = np.array(self.resetDemand)
                np.random.shuffle(shuffle)
                self.shuffleDemand = shuffle

            self.count += 1
            if self.count % 100 ==0:
                self.optimalAverage.append(np.average(self.totalOptimalCostList))
                self.calculatedAverage.append(np.average(self.totalCostList))
                self.performAverage.append(np.average(self.totalperformanceList))



        
            print("DONE: \nLoad: ", [round(i, 2) for i in self.load], "\nTotal Reward: ", round(self.totalReward, 2))
            print("DONE: \nLoad: ", [round(i, 2) for i in self.load], "\nTotal Reward: ", round(self.totalReward, 2), file=fileOut)
            print("Demand: ", self.demand )
            print("Demand: ", self.demand, file=fileOut)            
            print("-"*25)
            print("-"*25)
            print("-"*25, file=fileOut)
            print("-"*25, file=fileOut)
            
        
        # return np.array([0, observation]), reward, done, info
        return np.array([self.deltaUtilization, observation, self.timeWindow]), reward, done, info
    
    # def render(self, mode="rgb_array"):
    def render(self, mode="human"):
        screen_h = 600
        screen_w = 1000
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_w, screen_h, "SB")
            
            # Y -> OBJECTIVE FUNCTION
            # NEED: COMFORT PENALTY, ENERGY COST 
            # OPTIMAL, LOAD (NO ADJUSTMENTS/SCHEDULE), TOTAL
            
            fig = plt.figure(1, figsize = (15,7))

            tc = [range(len(self.totalOptimalCostList))]
            tcc = np.reshape(tc, (len(self.totalOptimalCostList),))
            
            print(np.shape(tcc))
            print(np.shape(self.totalOptimalCostList))
            print(np.shape(self.totalCostList))
            
            # ------------------------------
            #       PLOTS
            # ------------------------------
            
            #x1, y1 -> OPTIMAL             
            plt.plot(tcc, self.totalOptimalCostList, color="red", label="Optimal", linestyle="dashed", zorder=10)
            
            #x2, y2 -> TOTAL
            plt.plot(tcc, self.totalCostList, color="blue", label="Agent", linestyle="solid", zorder=5)
            
            #x3, y3 -> RL PERFORMANCE (LOAD)
            plt.plot(tcc, self.totalperformanceList, color="green", label="Conventional", linestyle="dotted", zorder=0)
            
            # ------------------------------
            # ------------------------------
            
            labels =["Optimal, Total", "Performance"]
            plt.ylabel = "Cost"
            plt.xlabel = "Time (ep_per_batch % 2000 == 0)"
            plt.title("SB Total Cost")
            # plt.legend(labels)
            plt.legend()
            print('before show')
            #plt.show()
            #time.sleep(30)
            #plt.close()
            print("after show")
            fig.savefig(figDir + 'SB_' + now + '.png')
            
            imgName = open(figDir + 'SB_' + now + '.png')            
            
            
            fig2 = plt.figure(2, figsize = (15,7))

            rc = [range(len(self.optimalAverage))]
            rcc = np.reshape(rc, (len(self.optimalAverage),))

            plt.plot(rcc, self.optimalAverage, color="red", label="Optimal", linestyle="dashed", zorder=10)
            plt.plot(rcc, self.calculatedAverage, color="blue", label="Agent", linestyle="solid", zorder=5)
            plt.plot(rcc, self.performAverage, color="green", label="Conventional", linestyle="dotted", zorder=0)
            plt.ylabel = "Cost"
            plt.xlabel = "Time (ep_per_batch % 2000 == 0)"
            plt.title("SB Average Cost")
            fig2.savefig(figDir + 'SBav_' + now + '.png')



            imgName = open(figDir + 'SB_' + now + '.png')
            
            #img = rendering.Image(imgName, 1., 1.)
            #imgTrans = rendering.Transform()
            #img.add_attr(imgTrans)
            
            #self.viewer.add_onetime(img)
            print('Image saved')
            
            # print("Saved '" + figDir + "/SB" + now + ".png'.\n%s\n" % ("-"*50))
            # print("Saved '" + figDir + "/SB" + now + ".png'.\n%s\n" % ("-"*50), file=fileOut)
            
        return self.viewer.render(return_rgb_array=mode == "rgb_array")
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


# ------------------------------
#            EV 
# ------------------------------
class ChargingStationEnv(Env):
    def __init__(self):
        self.viewer = None
        self.load = []

        self.optimalEVAverage = [] 
        self.calculatedEVAverage = []
        self.performEVAverage= []

        # self.required = []
        self.required = 2
        self.count = 0 
        self.iter = 0
        self.timeStep = 0
        self.totalReward = 0
        self.cost = 1
        self.chargingDeadline = 2
        self.newRequired = 2
        self.resetRequired = 2
        self.demandCharge = 0
        self.avgCost = 0
        self.optimalCost = 0
        self.optimalCostList = []
        self.totalCostList = []
        self.totalOptimalCostList = []
        self.totalperformanceList = []   
        # self.time = 0
        # self.reward = 0
        self.action_space = None
        self.observation_space = None
        self.reset()
        
    def reset(self):
        # self.time = 0
        self.timeStep = 0
        self.totalReward = 0
        self.cost = 1
        self.avgCost = 0
        self.optimalCost = 0
        self.performanceList = []
        self.optimalCostList = []    
        # self.required = [1, 1, 0]
        self.load = []
        self.chargingDeadline = 2
        self.required = self.newRequired
        # self.chargingDeadline = len(self.required) - 1
        # self.chargingDeadline = len(self.required) - 1
        # print("Charging deadline: ", self.chargingDeadline, file=fileOut)
        # self.load = sum([l for l in self.required])
        # self.action_space = Box(low=0, high=self.required - sum([i for i in self.load]), shape=(1,), dtype=float)
        # self.action_space = Box(low=0, high=self.required, shape=(1,), dtype=float)
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype="float32")
        # INCREASE OBSERVATION SPACE IF TRAIN LOSS CURVE IS UNIMPRESSIVE
        # self.observation_space = Box(low=np.array([-self.required, -self.chargingDeadline]), high=np.array([self.required, self.chargingDeadline]), shape=(2,), dtype=float)
        self.observation_space = Box(low=np.array([0, 0, 0]), high=np.array([self.chargingDeadline, self.resetRequired, self.resetRequired]), shape=(3,), dtype="float32")
    
    
        # TRY CONVERTING DIST SPACE TO [0, 1] USING LOGISTIC FUNCTION 
        # Gaussian to beta (uniform) transformation using CDF (but may not work for non-trivial)
        
        
        return np.array([0,0,0])
        # pass
        # return state
    
    def step(self, action):
        info = {}
        reward = 0
        done: bool = False
        action[0] = abs(action[0])
        self.load.append(action[0])
        self.demandCharge = max(self.load)
        
        
        print("EV stepping: ", round(self.load[self.timeStep], 2))
        print("*"*25)
        print("*"*25)
        print("EV stepping: ", round(self.load[self.timeStep], 2), file=fileOut)
        print("*"*25, file=fileOut)
        print("*"*25, file=fileOut)
        
        
        print("Timestep: ", self.timeStep)
        print("-"*5)
        # print("Load: ", round(self.load, 2))
        print("Load: ", [round(i, 2) for i in self.load][self.timeStep])
        # print("Required: ", self.required[self.timeStep])
        print("Required: ", self.required)
        print("Timestep: ", self.timeStep, file=fileOut)
        print("-"*5, file=fileOut)
        # print("Load: ", round(self.load, 2), file=fileOut)
        print("Load: ", [round(i, 2) for i in self.load][self.timeStep], file=fileOut)
        # print("Required: ", self.required[self.timeStep], file=fileOut)
        print("Required: ", self.required, file=fileOut)
    
            
        # observation = self.required[self.timeStep] 
        # observation = self.required - self.load[0]
        # observation = self.required - self.load[self.timeStep]
        observation = (self.required - sum([i for i in self.load]))
        
        print("Observation: ", round(observation, 2))
        print("-"*25)
        print("Observation: ", round(observation, 2), file=fileOut)
        print("-"*25, file=fileOut)
        
        # REWARD ONLY NEEDS TO MAKE SURE REQUIRED ENERGY IS CHARGED IN A FLAT ENERGY CHARGE DISTRIBUTION DUE TO PRICE FOR NOW
        # MEET DEADLINE BY REQUIRED CHARGING AMOUNT (REQUIRED -> TOTAL ENERGY)
        # CAN STOP AND RESTART MODEL PER 24 HOURS FOR NOW -> CONVOLUTION (SLIDING WINDOW)
        
        toleranceWindowLow, toleranceWindowHigh = abs(self.required - .03),  abs(self.required + .03)
        
        if self.timeStep == self.chargingDeadline:
            reward -= self.load[self.timeStep] + 12*(self.demandCharge) + 11*(abs(self.required - sum([i for i in self.load])))    
            # reward -= self.load[self.timeStep] + 6*(self.demandCharge) + 5.5*(abs(self.required - sum([i for i in self.load])))    

        elif self.timeStep < self.chargingDeadline:
            reward -= self.load[self.timeStep]
        
        elif self.timeStep == self.chargingDeadline \
            and (sum([i for i in self.load]) > toleranceWindowLow \
                and sum([i for i in self.load]) < toleranceWindowHigh):
            reward -= self.load[self.timeStep] + 2*(self.demandCharge)
            

    
        #DOES NOT ACTUALLY CONVERGE BASED ON PREVIOUS EPOCH'S REWARD; 
        #   -> INSTEAD, REWARD IS TOO RANDOM AT BEGINNING OF EACH EPOCH SO THE 
        #       AGENT NEVER LEARNS THE PROPER POLICY
            
        self.timeStep += 1
        self.totalReward += reward
        done = True if self.timeStep > self.chargingDeadline else done
        #done = True if self.timeStep > self.required or sum([i for i in self.load]) == self.required else done
        
        print("Total Reward: ", round(self.totalReward, 2))
        print("-"*25)
        print("Total Reward: ", round(self.totalReward, 2), file=fileOut)
        print("-"*25, file=fileOut)
        print("-"*25, file=fileOut)
        print("-"*25)
        print("-"*25)
        if done:
            
            # can add an margin for error
            if sum([i for i in self.load]) < self.required:    
                penalty = 2.25* (self.required - sum([i for i in self.load])) 
            else: penalty= 0
            self.totalCost = sum([i * self.cost for i in self.load])+ 2*(max(self.load))+ penalty
            self.totalCostList.append(self.totalCost)

            self.performanceList = calculatePerformanceEV(required = self.required)
            self.totalperformanceList.append(sum(self.performanceList)+2*(max(self.performanceList)))

            self.optimalCostList = calculateEVOptimal(required=self.required)
            self.optimalCostList = [i * self.cost for i in self.optimalCostList] 
            self.totalOptimalCostList.append( (sum(self.optimalCostList))+ 2*(max(self.optimalCostList)))

            self.iter +=1
            if self.iter % 200 == 0:
                self.newRequired =  random.randrange(-1,2) * round(random.random(),3) + self.resetRequired


            self.count += 1
            if self.count % 100 ==0:
                self.optimalEVAverage.append(np.average(self.totalOptimalCostList))
                self.calculatedEVAverage.append(np.average(self.totalCostList))
                self.performEVAverage.append(np.average(self.totalperformanceList))
 

            print("DONE: \nLoad: ", [round(i, 2) for i in self.load], "\nTotal Reward: ", round(self.totalReward, 2))
            print("DONE: \nLoad: ", [round(i, 2) for i in self.load], "\nTotal Reward: ", round(self.totalReward, 2), file=fileOut)
            print("-"*25)
            print("-"*25)
            print("-"*25, file=fileOut)
            print("-"*25, file=fileOut)
            
        
        # return np.array([0, observation]), reward, done, info
        return np.array([self.timeStep, observation, self.required]), reward, done, info
    
    def render(self, mode="human"):
        screen_h = 600
        screen_w = 1000
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_w, screen_h, "SB")
            
            # Y -> OBJECTIVE FUNCTION
            # NEED: COMFORT PENALTY, ENERGY COST 
            # OPTIMAL, LOAD (NO ADJUSTMENTS/SCHEDULE), TOTAL
            
            fig = plt.figure(figsize = (15,7))
            
            tc = [range(len(self.totalOptimalCostList))]
            tcc = np.reshape(tc, (len(self.totalOptimalCostList),))
            
            print(np.shape(tcc))
            print(np.shape(self.totalOptimalCostList))
            print(np.shape(self.totalCostList))
            
            # ------------------------------
            #       PLOTS
            # ------------------------------
            
            #x1, y1 -> OPTIMAL             
            plt.plot(tcc, self.totalOptimalCostList, color="red", label="Optimal", linestyle="dashed", zorder=10)
            
            #x2, y2 -> TOTAL
            plt.plot(tcc, self.totalCostList, color="blue", label="Calculated", linestyle="solid", zorder=5)
            
            #x3, y3 -> RL PERFORMANCE (LOAD)
            plt.plot(tcc, self.totalperformanceList, color="green", label="Performance", linestyle="dotted", zorder=0)
            
            # ------------------------------
            # ------------------------------
            
            labels =["Optimal, Total", "Performance"]
            plt.ylabel = "Cost"
            plt.xlabel = "Time (ep_per_batch % 2000 == 0)"
            plt.title("EV Total Cost")
            # plt.legend(labels)
            plt.legend()
            #plt.show()
            
            fig.savefig(figDir + 'EV_' + now + '.png')
            imgName = open(figDir + 'EV_' + now + '.png')


            fig3 = plt.figure( figsize = (15,7))

            rc = [range(len(self.optimalEVAverage))]
            rcc = np.reshape(rc, (len(self.optimalEVAverage),))

            plt.plot(rcc, self.optimalEVAverage, color="red", label="Optimal", linestyle="dashed", zorder=10)
            plt.plot(rcc, self.calculatedEVAverage, color="blue", label="Agent", linestyle="solid", zorder=5)
            plt.plot(rcc, self.performEVAverage, color="green", label="Conventional", linestyle="dotted", zorder=0)
            plt.ylabel = "Cost"
            plt.xlabel = "Time (ep_per_batch % 2000 == 0)"
            plt.title("EV Average Cost")
            fig3.savefig(figDir + 'EVav_' + now + '.png')
            
            #img = rendering.Image(imgName, 1., 1.)
            #imgTrans = rendering.Transform()
            #img.add_attr(imgTrans)
            
            #self.viewer.add_onetime(img)
            
            # print("Saved '" + figDir + "/SB" + now + ".png'.\n%s\n" % ("-"*50))
            # print("Saved '" + figDir + "/SB" + now + ".png'.\n%s\n" % ("-"*50), file=fileOut)
            
        return self.viewer.render(return_rgb_array=mode == "rgb_array")
            
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None