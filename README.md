# Curiosity driven exploration for robotics Pytorch (Proximal policy optimization)
Implementation of curiosity with PPO on OpenAI fetch. Here the robot is trained only on intrinsic rewards without any external reward shaping. Intriguingly, the robot discovers the path to the cube and does random perturbations to the cube in order to maximise rewards from unpredictability.
<td><img src="/images/curiosity.gif?raw=true" width="500" height="400"></td>

## Code structure
*dyn_model.py* contains a normalizer for the state and a class which predicts the next state given the current state and action. *PPO.py* contains the reinforcement learning algorithm.

## Requirements
- Python3.5+
- PyTorch 0.4.0
- OpenAI Gym ==0.10.8
- **Mujoco** physics engine- Find the guide on how to install it on OpenAI website

## Training
Clone the repository
```
git clone https://github.com/Ameyapores/Curiosity-for-robotics
cd Curiosity-for-robotics
```
start training
```
python3 main.py 
```

