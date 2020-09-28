# Friction Finger Gripper Contol 
This branch contains the code for policy generation of FFG using Deep Q Networks

## Installation Dependencies
Create a virtual environment with python 2.7 and with the following dependencies
1. torch
2. matplotlib
3. Pyyaml
4. sympy
5. numpy
6. tensorboard
7. scipy
8. random
9. gym

## Description of files in this repository:
- `dqn_agent_HER.py`:Contains the agent description.
- `env_mg.py`: Contains the multigoal environmet description.
- `model.py`: Conatins the description of the model.
- `Friction_Gripper_learning_algorithm.py`: This is the main file which contains the code for train/test/evaluate.
- `checkpoint11.pth`:Current trained network parameters.
- `params.yaml`: Contains the parameters for to be loaded in the main
- 'Valid_action_table.txt': Contains the valid actions for the corresponding object size in the params.yaml

## Hyper parameter setting
All the hyperparameters and environment params are set in parameters.yaml

## Train/Test/Evaluate
Set the corresponding values in the params.yaml to do the corresponding function.
- 0: train
- 1: test
- 2: plot

## Creating Valid action table
Whenever object size changes, need to create a valid action table. Set the corresponding flag in params.yaml. And run the Friction_Gripper_learning_algorithm.py.

Note: The current action table in the repository can be used for differnet object sizes as well but the resulting policy might not be optimal.

## Running the code
1. Set the parameters in the params.yaml
2. Run the main.py

## Example Test plot
Start Pose: (7.9,7.2,-90)
Goal Pose: (7.9,7.2,0)

#### DQN policy
![RL path](https://github.com/gokul-gokz/Friction_finger_gripper_RL/blob/master/New_state_action_space_RL_test/Data/Q-learning/Test2.png)

The path.txt file inside the PLOTS folder stores the path(coordinates in Finger space) in the format of text file.
 
## Things to do
1. Remove the global variable and rewrite the code structure in a better way.  
2. Rewrite parts of the code to make it readable
3. Move the plotting functions in the main.py to a different file


