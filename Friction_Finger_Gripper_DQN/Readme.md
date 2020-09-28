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

## Things to do
Remove the global variable and rewrite the code structure in a better way.  

