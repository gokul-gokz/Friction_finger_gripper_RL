import numpy as np
import math
import scipy.optimize as opt
from sympy import *
import random

FINGER_END = 10
FINGER_START = 7
PALM_WIDTH = 5
TH1_MAX= 2.485 #142.5 degrees
TH2_MIN= 0.65 #37.5
FINGER_WIDTH=1
K=0.1
OBJECT_SIZE=2.5



def calculate_th1(th2, d2):
    # Calculate theta2, d2
    d2v = np.array([d2 * np.cos(np.float64(th2)), d2 * np.sin(np.float64(th2))])
    w0v = np.array([OBJECT_SIZE * np.sin(np.float64(th2)), -OBJECT_SIZE * np.cos(np.float64(th2))])
    wpv = np.array([PALM_WIDTH, 0])
    f1v = np.array([FINGER_WIDTH * np.sin(np.float64(th2)), -FINGER_WIDTH * np.cos(np.float64(th2))])
    av = d2v - f1v - w0v + wpv

    d1 = np.sqrt((av * av).sum() - FINGER_WIDTH * FINGER_WIDTH)
    th1 = np.arctan2(av[1], av[0]) + np.arctan2(FINGER_WIDTH, d1)

    return th1

def calculate_th2(th1, d1):
    d1v = np.array([d1 * np.cos(th1), d1 * np.sin(th1)])
    w0v = np.array([OBJECT_SIZE * np.sin(th1), -OBJECT_SIZE * np.cos(th1)])
    wpv = np.array([PALM_WIDTH, 0])
    f2v = np.array([FINGER_WIDTH * np.sin(th1), -FINGER_WIDTH * np.cos(th1)])
    av = d1v + w0v + f2v - wpv

    d2 = np.sqrt((av * av).sum() - FINGER_WIDTH * FINGER_WIDTH)
    th2 = np.arctan2(av[1], av[0]) - np.arctan2(FINGER_WIDTH, d2)

    return th2

def action_right_equations(variables) :
    (th1,th2) = variables
    eqn1 = FINGER_WIDTH*sin(th1)+FINGER_WIDTH*sin(th2)+left_position * cos(th1) + OBJECT_SIZE * sin(th1) - PALM_WIDTH - right_position * cos(th2)
    eqn2 =-FINGER_WIDTH*cos(th1)-FINGER_WIDTH*cos(th2)+left_position * sin(th1) - OBJECT_SIZE * cos(th1) - right_position * sin(th2)
    return [eqn1, eqn2]

def action_left_equations(variables) :
    global left_position
    global right_position


    (th1, th2) = variables
    eqn1 = FINGER_WIDTH * sin(th1) + FINGER_WIDTH * sin(th2) + left_position * cos(th1) + OBJECT_SIZE * sin(th2) - PALM_WIDTH - right_position * cos(th2)
    eqn2 = -FINGER_WIDTH * cos(th1) - FINGER_WIDTH * cos(th2) + left_position * sin(th1) - OBJECT_SIZE * cos(th2) - right_position * sin(th2)
    return [eqn1, eqn2]

def theta_conversion(left, right, action_name):
    global left_position
    global right_position

    left_position =left
    right_position=right
    if (action_name == 2 or action_name == 3):
        for i in range(31):
            initial_guess=(i/10.0,i/10.0)
            solution = opt.fsolve(action_right_equations, initial_guess, full_output=True)
            if solution[2]==1 and solution[0][0]>0 and solution[0][0]<3.14 and solution[0][1]<3.14 and solution[0][1]>0:
                return solution[0]

                #solution = opt.fsolve(action_right_equations, (0.1, 1.0))

        # print "right"
        # print "left",left_position,"right",right_position
        # #print solution
        return (None,None)
    elif (action_name == 0 or action_name == 1):
        for i in range(31):
            initial_guess=(i/10.0,i/10.0)
            solution = opt.fsolve(action_left_equations, initial_guess, full_output=True)
            if solution[2] == 1 and solution[0][0] > 0 and solution[0][0] < 3.14 and solution[0][1] < 3.14 and solution[0][1] > 0:
                return solution[0]

                #solution = opt.fsolve(action_right_equations, (0.1, 1.0))

        # print "right"
        # print "left",left_position,"right",right_position
        # #print solution
        return (None,None)
    elif (action_name==4):
        solution= np.pi - np.arccos((((right_position-OBJECT_SIZE)**2 + OBJECT_SIZE**2 - PALM_WIDTH**2 - (left_position + OBJECT_SIZE)**2)/(2*PALM_WIDTH*(left_position+OBJECT_SIZE))))
        return (solution)

    elif (action_name==5):
        solution=np.arccos(((left_position - OBJECT_SIZE)**2 + OBJECT_SIZE**2 - (right_position+OBJECT_SIZE)**2 - PALM_WIDTH**2)/(2*PALM_WIDTH*(right_position + OBJECT_SIZE)))

        return (solution)

def limit_check(left_pos, right_pos, action,OBJECT_SIZE):
    #Calculate next state
    if(action==0):
        left_position = left_pos+0.1
        right_position = right_pos
    if (action == 1):
        left_position = left_pos - 0.1
        right_position = right_pos
    if (action == 2):
        left_position = left_pos
        right_position = right_pos+0.1
    if (action == 3):
        left_position = left_pos
        right_position = right_pos - 0.1
    if (action == 4):
        left_position = left_pos + OBJECT_SIZE
        right_position = right_pos - OBJECT_SIZE
    if (action == 5):
        left_position = left_pos - OBJECT_SIZE
        right_position = right_pos + OBJECT_SIZE
    if (action == 0 or action == 1 or action == 2 or action == 3):
        if (
                left_position <= FINGER_END and left_position >= FINGER_START and right_position <= FINGER_END and right_position >= FINGER_START):
            sol = theta_conversion(left_position, right_position, action)
            TH2_MAX = calculate_th2(TH1_MAX, left_position)
            TH1_MIN = calculate_th1(TH2_MIN, right_position)
            th1 = sol[0]
            th2 = sol[1]

            if (th1 <= TH1_MAX and th1 >= TH1_MIN and th2 >= TH2_MIN and th2 <= TH2_MAX):
                return True
            else:

                return False
        else:
            return False

    elif action == 4:
        if (left_position<= FINGER_END and left_position >= FINGER_START and right_position <= FINGER_END and right_position >= FINGER_START):
            th1 = theta_conversion(left_position-OBJECT_SIZE, right_position+OBJECT_SIZE, action)
            th2 = calculate_th2(th1, left_position)
            TH2_MAX = calculate_th2(TH1_MAX, left_position)
            TH1_MIN = calculate_th1(TH2_MIN, right_position)


            if (th1 <= TH1_MAX and th1 >= TH1_MIN and th2 >= TH2_MIN and th2 <= TH2_MAX):

                return True
            else:

                return False
        else:
            return False

    elif action == 5:
        if ( left_position<= FINGER_END and left_position>= FINGER_START and right_position  <= FINGER_END and right_position>= FINGER_START):
            th2 = theta_conversion(left_position+OBJECT_SIZE, right_position-OBJECT_SIZE, action)
            th1 = calculate_th1(th2, right_position)
            TH2_MAX = calculate_th2(TH1_MAX, left_position)
            TH1_MIN = calculate_th1(TH2_MIN, right_position)


            if (th2 >= TH2_MIN and th2 <= TH2_MAX and th1 <= TH1_MAX and th1 >= TH1_MIN):
                return True
            else:

                return False
        else:
            return False

#Friction Finger gripper environment
class Friction_finger_env:
    def __init__(self,start=(7,7),object_s=2.5,low_limit=FINGER_START,high_limit=FINGER_END):
        self.finger_low_limit=low_limit
        self.finger_high_limit=high_limit
        self.current_state=self.update_start_state(start)
        self.object_size=object_s
        self.actions = (0,1,2,3,4,5)
        self.valid_Actions = self.calculate_action_table()
        self.next_state=(0,0)
        self.reward=0
        self.done=0
        self.prev_action=-1
        self.goal=(7.4,10.0)

        #Action list
        # 1 -> Left slide up
        # 2 -> Left slide down
        # 3 -> Right slide up
        # 4 -> Right slide down
        # 5 -> Rotate clockwise
        # 6 -> Rotate anticlockwise


    def calculate_action_table(self):
        action_table=dict()
        i=self.finger_low_limit
        while(i<=self.finger_high_limit):
            j=self.finger_low_limit
            while (j <= self.finger_high_limit):
                s=(i,j)
                action=[]
                for a in self.actions:
                    if (limit_check(s[0], s[1], a, self.object_size)):
                     action.append(a)
                action_table[s]=action
                j=round(j+0.1,10)  #round function is used to approximate the float values so that they can be compared
            i=round(i+0.1,10)
        return action_table


    def calculate_next_state(self,action):
        if action in self.valid_Actions[self.current_state]:
            if action == 0:
                return(round(self.current_state[0]+0.1,10),round(self.current_state[1],10))

            elif action == 1:
                return(round(self.current_state[0]-0.1,10),round(self.current_state[1],10))

            elif action == 2:
                return(round(self.current_state[0],10),round(self.current_state[1]+0.1,10))

            elif action == 3:
                return(round(self.current_state[0],10),round(self.current_state[1]-0.1,10))

            elif action == 4:
                return(round(self.current_state[0]+self.object_size,10),round(self.current_state[1]-self.object_size,10))

            elif action == 5:
                return(round(self.current_state[0]-self.object_size,10),round(self.current_state[1]+self.object_size,10))
        else:

            return self.current_state


    def calculate_reward(self,action,next_state):
        if(self.current_state==self.goal):
            print(self.next_state,self.goal)
            return 1

        elif (action==self.prev_action) or (self.prev_action==-1):
            return -0.8

        elif(self.current_state==next_state):
            return -100

        else:
            return -1
    def update_start_state(selfself,start):
        return start

    def reset(self):
        self.done = 0
        self.prev_action = 0
        #self.current_state= (7.0+int(int(np.random.random()*10)/20.0),7.0+int(int(np.random.random()*10)/20.0))
        self.current_state = (random.randrange(FINGER_START*10,FINGER_END*10)/10,random.randrange(FINGER_START*10,FINGER_END*10)/10)
        return self.current_state

    def step(self,action):
        next_state= self.calculate_next_state(action)
        reward = self.calculate_reward(action,next_state)
        self.current_state=next_state

        done= 1 if (self.current_state==self.goal) else 0
        self.prev_action = action
        return next_state,reward,done


if __name__ == '__main__':
    a=env((7.0,7.0))
    print(a.valid_Actions[(7.0,7.0)])
    print(a.step(2))
    print(a.step(3))
    print(a.step(1))
    print(a.step(1))
    print(a.step(5))
    print(a.reset())





