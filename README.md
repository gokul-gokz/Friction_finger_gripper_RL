# Friction Finger gripper

### Getting Started

- Read this [paper](https://github.com/gokul-gokz/Friction_finger_gripper_RL/blob/master/Friction_Finger_Gripper_ICRA_2020%20(26).pdf) to understand about the Friction finger environment.
- Checkout the dev branch of this repository [WIHM for Variable friction finger using visual servoing and motion planner](https://github.com/gokul-gokz/Variable_friction_finger) to find a working code of VF finger planning and control with Modified A* planner and visual servoing.
- Checkout the master branch, if you just want the driver code for controlling the Variable friction finger gripper.

## Description of files in this repository:
- `agent.py`:Contains the agent description.
- `env.py`: Contains the environmet description.
- `monitor.py`: Conatins the code for interaction between agent and the environment..
- `main.py`: Run this file in the terminal to check the performance of your agent.
- `Policy.txt`: Contains the policy for the given environment(specific goal state).
- `Q_table.txt`: Contains the Q_table for all the state in the state action space.
- `test_policy.py`: Contains a testing code where any valid start state can be given and based on the polcicy stored in policy.txt, a plan will be generated and plotted.

## Parameter setting and environment description
1. Set the hyperparameters for the agent in agent.py.  [default values are given here]
	- epsilon=1.0
	- epsilon_decay=0.0001
	- epsilon_min=0.0003
	- alpha=0.7
	- gamma=0.995
 Note: gamma is set using the thumb rule: 1-1/No of steps needed to ccomplete an episode

2. Friction finger environment is defined in env.py which follows the same structure with the paper defined above.
         - Statespace: (L,R,theta)    (7<=L,R<=12), theta=(-90,0,90)  
	 - Actionspace: (0-5):Left Slide up, Left Slide Down, Right Slide Up, Right slide Down, Rotate Clockwise, Rotate Antiiclockwise

3. Friction finger environment defined in a new setting.(inside New_state_action_space_RL_test folder)
	 - Statespace:(x,y,theta,friction_surface_setting)  (7<=L,R<=12), theta=(-90,0,90), friction_surface_setting=('lh','hl','hh')
	 - Actionspace:(0-4),Actuator left, Actuator right, Set friction 'lh', Set friction 'hl', Set friction 'hh'

4. Friction finger environment for a multi goal setting.(inside Multi_goal_training)
	 - Statespace:(x,y,theta,friction_surface_setting,(goal_L,goal_R,goal_theta))  (7<=L,R<=12), theta=(-90,0,90), friction_surface_setting=('lh','hl','hh')
	 - Actionspace:(0-4),Actuator left, Actuator right, Set friction 'lh', Set friction 'hl', Set friction 'hh'

4. Number of episodes and Max_number_of_steps in an episode can be set in monitor.py 


## Steps to run the code:
1. Inside main.py
	- Create an object for the agent with required hyperparameters.
	- Create an object for the corresponding environment by defining the goal.
	- Create an object of monitor defining the number of training episodes and max number of steps in an epsiode. 
2. Run python main.py  to train the agent.
3. After the completion of the training, Q_table.txt and Policy.txt will be created.
4. The generated policy can be tested by specifying any start state in test_policy.py and running the code. A plot will be generated showing the path in cartesian space from start to goal state.




  
