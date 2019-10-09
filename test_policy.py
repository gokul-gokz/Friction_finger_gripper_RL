from agent import Agent
from monitor import interact
from env import Friction_finger_env
import csv
import json
# #Load policy
# policy=dict()
# reader = csv.reader(open('Policy.csv'))
#
# for row in reader:
#
#     if(len(row)>0):
#         l = len(row[0])
#
#         key1,key2 = float(row[0][1:4]),float(row[0][6:9])
#
#         if(row[1]=="Left up"):
#          val=0
#         elif (row[1] == "Left down"):
#          val = 1
#         elif (row[1] == "Right up"):
#          val = 2
#         elif (row[1] == "Right down"):
#          val = 3
#         elif (row[1] == "Rotate clock"):
#          val = 4
#         elif (row[1] == "Rotate anticlock"):
#          val = 5
#
#         policy[(key1,key2)]=val

with open('policy.txt') as json_file:
    policy = json.load(json_file)
    
    # print(data['(1,2)'])

#Test_cases
start_state=(7.2,7.2,-90)
env1=Friction_finger_env(start_state,False)
action=policy[str(start_state)]
i=0
print("\n")
while(i<150):
    next_state,reward,done=env1.step(action)
    print(next_state,action)
    action=policy[str(next_state)]
    i=i+1
    if(done):
        print("No of steps",i)
        break