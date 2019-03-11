# Basic Deep Q-Network

An implementation of a basic deep Q-network. The core of this follows the description in this [blog]
(https://becominghuman.ai/lets-build-an-atari-ai-part-0-intro-to-rl-9b2c5336e0ec), which uses DeepMind's paper 
[Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). My DQN is pretty 
basic, and doesn't utilise all of the techniques discussed in the blog and paper. As a result, it converges very slowly!

## Setup

This library requires:
- gym (including atari_py)
- Tensorflow (& keras)

## Contents and Usage

'''Teach_Breakout.py''': This script trains a DQN to play Breakout, following this [blog]
(https://becominghuman.ai/lets-build-an-atari-ai-part-0-intro-to-rl-9b2c5336e0ec). Without specifying any arguments, 
we use the hyperparameters discussed in the [blog] 
(https://becominghuman.ai/lets-build-an-atari-ai-part-0-intro-to-rl-9b2c5336e0ec), and every 100k training steps the 
model is saved as "saved_models/Run_[# training steps]".

'''View_Breakout.py''': This is used to view a model that has already been training. Use argument "-m [name of model]"
 to specify which model to load and use. By default, it loads "Trained_models/7M_steps".