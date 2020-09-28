import sys
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from numpy.core._multiarray_umath import ndarray
import argparse

from env_mg import Friction_finger_env
from dqn_agent_HER import Agent
from collections import deque
import matplotlib.pyplot as plt
import gym
import csv
import json
import time
import math
from sympy import *
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import yaml



#Gripper params
PALM_WIDTH = 5.0
TH1_MAX= 2.485 #142.5 degrees
TH2_MIN= 0.65 #37.5
FINGER_WIDTH=1
K=0.1

#Finger params(Changeable)
THETA_LOW=-90
THETA_HIGH=90
FINGER_END = 10.0
FINGER_START = 8.0
OBJECT_SIZE=1.5

#Helper variable for plotting(X,Y) positions
X=[]
Y=[]

left_position =0
right_position=0

def finger_to_cartesian(L,R,A,th):
    if A==2 or A==3:
        x_square = (L - OBJECT_SIZE/2.0)*np.cos(np.float64(th[0])) + (FINGER_WIDTH + OBJECT_SIZE/2.0)*np.sin(np.float64(th[0]))
        # x_square = (R - (OBJECT_SIZE/2.0))
        y_square = (L - OBJECT_SIZE/2.0)*np.sin(np.float64(th[0])) - (FINGER_WIDTH + OBJECT_SIZE/2.0)*np.cos(np.float64(th[0]))


    elif A==0 or A==1:
        x_square = PALM_WIDTH + (R - OBJECT_SIZE/2.0)* np.cos(th[1]) - (OBJECT_SIZE/2.0 + FINGER_WIDTH)* np.sin(th[1])
        y_square = (R - OBJECT_SIZE/2.0)* np.sin(th[1]) + (OBJECT_SIZE/2.0 + FINGER_WIDTH)* np.cos(th[1])


    return x_square,y_square

def ik_finger(x, y, w0, wp, fw):
    t2, d2 = symbols('t2 d2')
    eqn1 = (d2 - w0 / 2) * cos(t2) + (fw + w0 / 2) * sin(t2)
    eqn2 = (d2 - w0 / 2) * sin(t2) - (fw + w0 / 2) * cos(t2)
    eqn3 = x ** 2 + y ** 2 - eqn1 ** 2 - eqn2 ** 2
    sold2 = solve(eqn3, d2)
    solt2 = solve(eqn1.subs(d2, sold2[1]) - x, t2)

    d2v = np.array([sold2[1] * cos(solt2[1]), sold2[1] * sin(solt2[1])])
    w0v = np.array([w0 * sin(solt2[1]), -w0 * cos(solt2[1])])
    wpv = np.array([wp, 0])
    f1v = np.array([fw * sin(solt2[1]), -fw * cos(solt2[1])])
    av = d2v + f1v + w0v - wpv

    d1 = sqrt((av * av).sum() - fw * fw)
    t1 = np.arctan2(float(av[1]), float(av[0])) - np.arctan2(float(fw), float(d1))
    return t1, solt2[1], d1, sold2[1]

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
        # print "-----------------------------------------------"
        # print left_position,right_position,solution
        return (solution)

    elif (action_name==5):
        solution=np.arccos(((left_position - OBJECT_SIZE)**2 + OBJECT_SIZE**2 - (right_position+OBJECT_SIZE)**2 - PALM_WIDTH**2)/(2*PALM_WIDTH*(right_position + OBJECT_SIZE)))

        return (solution)


def calculate_theta(states,actions):
    thetas=[]
    i=0
    for state,action in zip(states,actions):
        if action==0 or action ==1 or action ==2 or action ==3:
            theta = theta_conversion(state[0],state[1],action)
        elif action==4:
            theta=[0,0]
            theta[0] = theta_conversion(state[0],state[1], action)
            theta[1] = calculate_th2(theta[0], state[0])
        elif action==5:
            theta = [0, 0]
            theta[1] = theta_conversion(state[0],state[1], action)
            theta[0] = calculate_th1(theta[1], state[1])
        #print(i,"=",theta)
        i=i+1
        thetas.append(theta)
    return thetas

def orientation_solvet1(t2,d1,d2,fw,w0,wp):
    t1 = symbols('t1')
    # eqn1 = np.array([[d1 * cos(t1)], [d1 * sin(t1)]]) + np.array([[fw * sin(t1)], [fw * cos(t1)]])
    eqn1 = np.array([[d1 * cos(t1)], [d1 * sin(t1)]]) + np.array([[fw * sin(t1)], [-fw * cos(t1)]]) + np.array([[fw * sin(t2)], [-fw * cos(t2)]]) - np.array([[(d2 + w0) * cos(t2) + wp], [(d2 + w0) * sin(t2)]])
    # print t2
    # eqn1 = sin(t1)

    a = simplify(np.dot(np.transpose(eqn1), eqn1))
    # print np.asscalar(simplify(np.dot(np.transpose(eqn1), eqn1)))
    eqn2 = a[0, 0] - 2 * (w0**2)
    solt1 = solve(eqn2, t1)
    t1 = solt1[1]
    ph = np.pi / 4 + np.arctan2(float((d2 + w0) * sin(t2) + fw * cos(t2) - (d1 * sin(t1) - fw * cos(t1))), float((d2 + w0) * cos(t2) - fw * sin(t2) + wp - (d1 * cos(t1) + fw * sin(t1))))
    return t1, ph


def orientation_solvet2(t1,d1,d2,fw,w0,wp):

    t2 = symbols('t2')
    eqn1 = np.array([[(d1 + w0) * cos(t1)], [(d1 + w0) * sin(t1)]]) + np.array([[fw * sin(t1)], [-fw * cos(t1)]]) + np.array([[fw * sin(t2)], [-fw * cos(t2)]]) - np.array([[(d2) * cos(t2) + wp], [(d2) * sin(t2)]])
    a = simplify(np.dot(np.transpose(eqn1), eqn1))
    #print(a)
    eqn2 = a[0, 0] - 2 * (w0**2)
    print(eqn2)
    solt2 = solve(eqn2, t2)
    print(solt2)
    return solt2[0]

def xy_rotation_clockwise(d1,d2,t1,t2,t1f,fw,w0,wp):

    x=[]
    y=[]
    theta1=[]
    theta2=[]

    while t1 >= t1f:

        t2 = orientation_solvet2(t1,d1,d2,fw,w0,wp)
        t2 = float(t2)
        object_x = ((d1) * np.cos(t1) + fw * np.sin(t1) - fw * np.sin(t2) + (d2-w0) * np.cos(t2) + wp) / 2
        object_y = ((d1 ) * np.sin(t1) - fw * np.cos(t1) + fw * np.cos(t2) + (d2-w0) * np.sin(t2)) / 2
        t1 = t1 - 0.05
        x.append(object_x)
        y.append(object_y)
        theta1.append(t1)
        theta2.append(t2)
    return x,y,theta1,theta2




def xy_rotation_anticlockwise(d1,d2,t1,t2,t2f,fw,w0,wp):

    d1 = d1-w0
    d2 = d2-w0
    t2f = np.arccos(((d1 - w0)**2 + (w0 + 2 * fw)**2 - (d2 + w0)**2 - wp**2) / (2 * wp * (d2 + w0)))

    x=[]
    y=[]
    theta1 = []
    theta2 = []
    while t2 <= t2f:
        t1, ph = orientation_solvet1(t2,d1,d2,fw,w0,wp)
        t1 = float(t1)
        ph = float(ph)

        object_x = (d1 * np.cos(t1) + fw * np.sin(t1) - fw * np.sin(t2) + (d2 + w0) * np.cos(t2) + wp) / 2
        object_y = (d1 * np.sin(t1) - fw * np.cos(t1) + fw * np.cos(t2) + (d2 + w0) * np.sin(t2)) / 2
        t2 = t2 + 0.05
        x.append(object_x)
        y.append(object_y)
        theta1.append(t1)
        theta2.append(t2)
    return x, y, theta1, theta2



def plot(L,R,theta,A):
    global X, Y
    del X[:]
    del Y[:]
    OFFSET=0
    n=len(A)
    #print("length of actions=",n)
  
    th1=[]
    th2=[]
    count=[]

    for i in range(n):


        if(A[i]==4 ):
            #print(i)
            if i==0:
                j=i
            else:
                j=i-1

            # x_r,y_r,t1,t2=xy_rotation_clockwise(L[i]-OBJECT_SIZE,R[i]+OBJECT_SIZE,theta[j][0],theta[j][1],theta[i][0],FINGER_WIDTH,OBJECT_SIZE,PALM_WIDTH)
            x_r, y_r, t1, t2 = xy_rotation_clockwise(L[i] , R[i] , theta[j][0], theta[j][1],
                                                     theta[i][0], FINGER_WIDTH, OBJECT_SIZE, PALM_WIDTH)


            X=X+x_r
            Y=Y+y_r
            th1=th1+t1
            th2 = th2 + t2

            count.append((i+OFFSET,len(x_r)))
            OFFSET += len(x_r)
        elif (A[i] == 5):
            #print(i)
            if i == 0:   #if the rotate action is the first action
                j = i
            else:           #if the rotate actions is the last
                j = i - 1


            # x_r, y_r,t1,t2 = xy_rotation_anticlockwise(L[i]+OBJECT_SIZE, R[i]-OBJECT_SIZE, theta[j][0], theta[j][1],theta[i][0],FINGER_WIDTH,OBJECT_SIZE,PALM_WIDTH)
            x_r, y_r, t1, t2 = xy_rotation_anticlockwise(L[i] , R[i], theta[j][0],
                                                         theta[j][1], theta[i][0], FINGER_WIDTH, OBJECT_SIZE,
                                                         PALM_WIDTH)
            # print "------------------------------------------------------------------------------"
            # print "l=",L[i]+OBJECT_SIZE
            # print "r=",R[i]-OBJECT_SIZE
            #print("th1=","th2=",theta[j][0], theta[j][1])
            # print x_r, y_r
            X = X + x_r
            Y = Y + y_r
            th1 = th1 + t1
            th2 = th2 + t2
            count.append((i+OFFSET, len(x_r)))
            OFFSET += len(x_r)
        else:
            x,y=finger_to_cartesian(L[i],R[i],A[i],theta[i])
            X.append(x)
            Y.append(y)
            th1.append(theta[i][0])
            th2.append(theta[i][1])

            # print  "L=",L[i],"R=",R[i]
            # print "theta1=",theta[i][0],"theta2=",theta[i][1]
            # print "x=",x,"y=",y

    if(len(X)>0):
        global closed_list
        # print closed_list
        C_X = []
        C_Y = []
        i = 0
        # for node in closed_list:
        #     i = i + 1
        #     if (node[3] == 'rotate_clockwise' or node[3] == 'rotate_anticlockwise' or i == 1):
        #         continue
        #     else:
        #         # print node[3]
        #         x, y = finger_to_cartesian(node[0], node[1], node[3], node[4])
        #         C_X.append(x)
        #         C_Y.append(y)
        # plt.plot(C_X, C_Y, 'y.',markersize=3,label='Expanded nodes')

        # plotting the points
        plt.plot(X, Y,'b-',label='Sliding action')

        for j in range(len(count)):
            # print"*********************************************************************************88"
            # print X[count[j][0]:count[j][0] + count[j][1]], Y[count[j][0]:count[j][0] + count[j][1]]
            plt.plot(X[count[j][0]:count[j][0]+count[j][1]],Y[count[j][0]:count[j][0]+count[j][1]],'r',label='Rotation action')


        plt.plot(X[len(X)-1],Y[len(Y)-1],'g*',label='Goal state',markersize=12)

        plt.plot(X[1],Y[1],'go',label='Start state',markersize=12)

        plt.xlim([-4, 10])
        plt.ylim([2, 12])

        # naming the x axis
        plt.xlabel('x - axis')
        # naming the y axis
        plt.ylabel('y - axis')

        # giving a title to my graph
        plt.title('Planned path for the object')
        plt.legend(loc='lower right')

        # textstr = '\n'.join((
        #     r'$\beta=%.1f$' % (current_w_p,),
        #     r'$\epsilon=%.1f$' % (current_w_s,)))
        # props = dict( facecolor='white', alpha=0.5)
        # plt.text(-9.5, 14.5, textstr, fontsize=14,
        #         verticalalignment='top', bbox=props)


        # function to show the plot
        Filename='Test_results/data'+'.png'
        plt.show()
        # plt.savefig(Filename)
        plt.clf()
        plt.close()

def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
  """Returns the current epsilon for the agent's epsilon-greedy policy.
  This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
  al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.
  Args:
    decay_period: float, the period over which epsilon is decayed.
    step: int, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before epsilon is decayed.
    epsilon: float, the final value to which to decay the epsilon parameter.
  Returns:
    A float, the current epsilon value computed according to the schedule.
  """
  steps_left = decay_period + warmup_steps - step
  bonus = (1.0 - epsilon) * steps_left / decay_period
  bonus = np.clip(bonus, 0., 1. - epsilon)
  return epsilon + bonus


def linearly_decaying_lr(decay_period, step, warmup_steps, epsilon):
 
  steps_left = decay_period + warmup_steps - step
  bonus = (1.0 - epsilon) * steps_left / decay_period
  bonus = np.clip(bonus, 0., 1. - epsilon)
  return epsilon + bonus

def identity_epsilon(unused_decay_period, unused_step, unused_warmup_steps,
                     epsilon):
  return epsilon




if __name__ == '__main__':
    global OBJECT_SIZE
    global FINGER_START
    global FINGER_END
    # parser=argparse.ArgumentParser()
    # parser.add_argument("params",type=str,help="Input path to hyperparameters file")
    # parser.add_argument("-v","--verbose", action="store_true", help="increase output verbosity")
    # args=parser.parse_args()
    
    filepath = './param10.yaml'

    with open(filepath) as file:
        param_list = yaml.load(file, Loader=yaml.FullLoader)

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    
    
    options=['TRAIN', 'TEST', 'PLOT']
    command= param_list['MODE']
    # command= 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if(command==0):
        writer = SummaryWriter(param_list['TENSORBOARD_PATH'])


        FINGER_START=param_list['FINGER_START']
        FINGER_END=param_list['FINGER_END']
        OBJECT_SIZE=param_list['OBJECT_SIZE']

        # Create the environment by sending  the parameters
        env = Friction_finger_env(start=(7.0,7.0,0),action_table_generate=param_list['TABLE_GEN'],object_s=OBJECT_SIZE,low_limit=FINGER_START,high_limit=FINGER_END,action_table_path=param_list['ACTION_TABLE_PATH'])

        # Create the DQN agent
        agent = Agent(state_size=6, action_size=6, fc1_units=param_list['FC1_units'], fc2_units=param_list['FC2_units'], seed=0,buffer_size=param_list['BUFFER_SIZE'],batch_size=param_list['BATCH_SIZE'],gamma=param_list['GAMMA'], tau=param_list['TAU'], lr=param_list['LR'], HER_update_episodes=param_list['HER_UPDATE_EPISODES'], bootstrap=param_list['BOOTSTRAP'],network_path=param_list['NETWORK_PATH'])  #state_size =6 because start->3 and goal->3

        # epsilon = 1.0
        # epsilon_decay = 0.999
        # epsilon_min = 0.06
        cum_reward=[]
        cum_avg_rewards=[]

        training_steps=0
        avg_rewards=deque(maxlen=20)

        EPSILON_MAX =param_list['EPSILON_MAX']
        TOTAL_EPOCHS = param_list['TOTAL_EPOCHS']
        EPSILON_MIN=param_list['EPSILON_MIN']
        EXP_FACTOR=param_list['EXP_FACTOR']
        NUM_CYCLES=param_list['NUM_CYCLES']
        MAX_EPISODE=param_list['MAX_EPISODE']
        MAX_STEPS=param_list['MAX_STEPS']
        HER_ENABLE=param_list['HER_ENABLE']
        FUTURE_K=param_list['FUTURE_K']
        
        ep_count=0
        
        for epoch in range(TOTAL_EPOCHS):
            epsilon = max(EPSILON_MAX - epoch * (EPSILON_MAX - EPSILON_MIN) / int(TOTAL_EPOCHS * EXP_FACTOR), EPSILON_MIN)
            # lr = max(LR_MAX - epoch * (LR_MAX - LR_MIN) / int(TOTAL_EPOCHS * EXP_FACTOR_LR), LR_MIN)
            #print("Epoch: {}, exploration: {:.0f}%, success rate: {:.2f}".format(epoch + 1, 100 * epsilon, success_rate))

            successes = 0
            step_count=0
            for cycle in range(NUM_CYCLES):
                for episode in range(MAX_EPISODE):
                    state,goal =env.reset()
                    state = np.array(state)
                    goal=np.array(goal)
                    print('start',state,'goal',goal,'eps',epsilon,'epoch',epoch,'cycle',cycle,'episode',episode)
                    sum_reward=0

                    for step in range(MAX_STEPS):
                        training_steps+=1
                        state_mg= np.concatenate((state,goal), axis=0, out=None)

                        action = agent.act(state_mg,epsilon)
                        next_state, reward, done= env.step(action)
                        next_state = np.array(next_state)
                        agent.step(state, action, reward, next_state, done, goal,HER_ENABLE,FUTURE_K,MAX_STEPS)
                        state=next_state
                        sum_reward+=reward
                        # writer.add_scalar('loss/train', agent.Q_loss,training_steps)
                        if done:
                            print("Reached goal in ", step, "steps")
                            successes+=1
                            cum_reward.append(sum_reward)
                            break
                    step_count+=step
                    ep_count+=1
                   
                    # writer.add_scalar('reward/train', sum_reward, ep_count)
                    #
                    # writer.add_histogram('Weights1layer/train',agent.qnetwork_local.fc1.weight.data, ep_count )
                    # writer.add_histogram('Weights2layer/train',agent.qnetwork_local.fc2.weight.data, ep_count )
                    avg_rewards.append(sum_reward)
                    cum_avg_rewards.append(np.mean(avg_rewards))
               
            # writer.add_scalar('SuccessPerEpoch/train',successes/50.0, epoch)
            # writer.add_scalar('Avg_steps/train',step_count/50.0, epoch)
		

        # Plotting the rewards and loss
        # plt.plot(cum_avg_rewards)
        # plt.show()
        # plt.savefig(param_list['R_PLOT_PATH'])
        # plt.plot(agent.Q_loss_list)
        # plt.show()
        # plt.savefig(param_list['L_PLOT_PATH'])


    
    elif(command==1):
        # Evaluator code(print actions of state)
        completion_rate=[]
        obj_size_list=[]
        FINGER_START=param_list['FINGER_START']
        FINGER_END=param_list['FINGER_END']
        for i in range(5,21,1):
          OBJECT_SIZE=i/10.0
          obj_size_list.append(OBJECT_SIZE)

          success=0
          env = Friction_finger_env(start=(7.0,7.0,0),action_table_generate=False,object_s=OBJECT_SIZE,low_limit=FINGER_START,high_limit=FINGER_END,action_table_path=param_list['ACTION_TABLE_PATH'])
          agent = Agent(state_size=6, action_size=6, fc1_units=param_list['FC1_units'], fc2_units=param_list['FC2_units'], seed=random.randint(0,1000),buffer_size=param_list['BUFFER_SIZE'],batch_size=param_list['BATCH_SIZE'],gamma=param_list['GAMMA'], tau=param_list['TAU'], lr=param_list['LR'], HER_update_episodes=param_list['HER_UPDATE_EPISODES'], bootstrap=param_list['BOOTSTRAP'],network_path=param_list['NETWORK_PATH'])  #state_size =6 because start->3 and goal->3


          # load the weights from file
          agent.qnetwork_local.load_state_dict(torch.load(param_list['NETWORK_PATH'],map_location=torch.device(device)))
          start = (7.5, 7.5, 90)
          goal = (7.5, 7.5, -90)
          
          
          for i in range(1,1000,1):

              start, goal = env.reset()
              c_state = np.concatenate((np.array(start), np.array(goal)))

              env.current_state = start
              env.goal = goal
              next_state = start
              done = 0
              count = 0
              print("Start=", c_state)
              while (not done and count<150):

                  state = torch.from_numpy(c_state).float().unsqueeze(0).to(device)
                  agent.qnetwork_local.eval()
                  with torch.no_grad():
                      action_values = agent.qnetwork_local(state)
                  a = np.argmax(action_values.cpu().data.numpy())
                  next_state, reward, done = env.step(a)
                  # if(abs(next_state[0]-goal[0])<0.3 and abs(next_state[1]-goal[1])<0.3 and abs(next_state[2]-goal[2])==0):
                  #   done=1
                  c_state = np.concatenate((np.array(next_state), np.array(goal)))
                  #print(a)
                  #print(reward)
                  count = count + 1
              print("Final=",c_state)
              if(done):
                  success=success+1
                  print("Goal reached","Episode_number=",i,"Success_rate=",success*100/i,"obj_size=",OBJECT_SIZE,"steps=",count)

              else:
                  print("Goal not reached","Episode_number=",i,"Success_rate=",success*100/i)
          completion_rate.append(success*100/i)
        plt.plot(obj_size_list,completion_rate)


    elif(command==2):
        global X,Y
        FINGER_START=param_list['FINGER_START']
        FINGER_END=param_list['FINGER_END']
        OBJECt_SIZE=param_list['OBJECT_SIZE']
        #Evaluator code(print policy)
        env = Friction_finger_env(start=(7.0,7.0,0),action_table_generate=False,object_s=OBJECT_SIZE,low_limit=FINGER_START,high_limit=FINGER_END,action_table_path=param_list['ACTION_TABLE_PATH'])
        agent = Agent(state_size=6, action_size=6, fc1_units=param_list['FC1_units'], fc2_units=param_list['FC2_units'], seed=random.randint(0,1000),buffer_size=param_list['BUFFER_SIZE'],batch_size=param_list['BATCH_SIZE'],gamma=param_list['GAMMA'], tau=param_list['TAU'], lr=param_list['LR'], HER_update_episodes=param_list['HER_UPDATE_EPISODES'], bootstrap=param_list['BOOTSTRAP'],network_path=param_list['NETWORK_PATH'])  #state_size =6 because start->3 and goal->3

        # load the weights from file
        agent.qnetwork_local.load_state_dict(torch.load(param_list['NETWORK_PATH'],map_location=torch.device(device)))
        # start = (8.0, 10.0, 0)
        # goal = (9.0, 9.0, -90)



        # c_state = np.concatenate((np.array(start), np.array(goal)))

        # # state=start
        # env.current_state = start
        # env.goal = goal
        # next_state = start
        # done = 0
        # count = 0

        # states=[]
        # actions=[]


        # env.goal=np.array(goal)

        #Remove
        states=[]
        actions=[]
        start, goal = env.reset()
        start = (7.9, 7.2, -90)
        goal = (7.9, 7.2, 0)
        c_state = np.concatenate((np.array(start), np.array(goal)))

        env.current_state = start
        env.goal = goal
        next_state = start
        done = 0
        count = 0
        print("Start=", c_state)

        states.append(start)


        n=0
        
        while(not done and n<150):

            n =n+1
            # state = torch.from_numpy(c_state).float().unsqueeze(0)
            state = torch.from_numpy(c_state).float().unsqueeze(0).to(device)
            agent.qnetwork_local.eval()
            with torch.no_grad():
                action_values = agent.qnetwork_local(state)
            print(action_values)
            action=np.argmax(action_values.cpu().data.numpy())
            actions.append(action)
            print(action)
            next_state, reward, done = env.step(action)
            print(next_state)
            states.append(next_state)
            c_state = np.concatenate((np.array(next_state), np.array(goal)))


        L = []
        R = []
        thetas = calculate_theta(states, actions)
        print(len(states))

       
        
       
        for state in states:
            L.append(state[0])
            R.append(state[1])
            # print (state)

        plot(L, R, thetas, actions)

        with open('./PLOTS/Path.txt', 'w') as filehandle:
        # filehandle.writelines("start=%s\n"% str(start))
        # filehandle.writelines("goal=%s\n"% str(goal))
          filehandle.writelines("%s\n" % str([state[0], state[1], action, theta[0], theta[1]]) for state,action,theta in zip(states,actions,thetas))
        # print(X)
        # for state,action,x,y in zip(states,actions,X,Y):
        #    print( str([state[0], state[1], action, x, y]) )
    else:
        print("Invalid mode")


       





