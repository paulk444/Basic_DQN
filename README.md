# Basic Deep Q-Network

An implementation of a basic deep Q-network. The core of this follows the description in this [blog](https://becominghuman.ai/lets-build-an-atari-ai-part-0-intro-to-rl-9b2c5336e0ec), which uses DeepMind's paper 
[Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). My DQN is pretty 
basic, and doesn't utilise all of the techniques discussed in the blog and paper. As a result, it converges very slowly! This DQN implementation is also very basic in the sense that the only real output is a trained DQN model; it doesn't produce or record any metrics that can be used to understand and diagnose the DQN.

## Setup

This library requires:
- gym (including atari_py)
- Tensorflow (& keras)

## Contents and Usage

```Teach_Breakout.py```: This script trains a DQN to play Breakout. Without specifying any arguments, 
we use the hyperparameters discussed in the [blog](https://becominghuman.ai/lets-build-an-atari-ai-part-0-intro-to-rl-9b2c5336e0ec), and every 100k training steps the 
model is saved as "saved_models/Run_[# training steps]" (so you need to create the directory "saved_models/" before running the script).

```View_Breakout.py```: This is used to view a model that has already been trained. Use argument "-m [name of model]"
 to specify which model to load. By default, it loads "Trained_models/7M_steps".
