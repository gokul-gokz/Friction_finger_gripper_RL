from agent import Agent
from monitor import interact
from env import Friction_finger_env
import csv

#Load policy
policy=dict()
reader = csv.reader(open('Policy.csv'))

for row in reader:

    if(len(row)>0):
        l = len(row[0])

        key1,key2 = float(row[0][1:4]),float(row[0][6:9])

        if(row[1]=="Left up"):
         val=0
        elif (row[1] == "Left down"):
         val = 1
        elif (row[1] == "Right up"):
         val = 2
        elif (row[1] == "Right down"):
         val = 3

        policy[(key1,key2)]=val


#Test_cases
start_state=(7.0,7.0)
env1=Friction_finger_env(start_state)
action=policy[start_state]
i=0
print("\n")
while(i<100):
    next_state,reward,done=env1.step(action)
    print(next_state,action)
    action=policy[next_state]
    i=i+1
    if(done):
        break