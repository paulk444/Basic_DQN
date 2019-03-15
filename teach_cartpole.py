"""
We make a DQN agent for learning in gym environments, following (and modifying/extended) the tutorial here:
https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26

This code is specifically for CartPole.
"""

import argparse
import os
import random
import time

import gym
import numpy as np
from keras import layers
from keras.models import Model
from tensorflow import keras

# TODO: Pre-process states to make them numpy arrays: "state = np.array([state])"
# TODO: Try different loss (mse?) and optimiser (Adam?)

# TODO STYLE: Keep comments up to date when the code changes!
# TODO STYLE: Ensure that: Python variables should be written_in_lowercase, except constants that are ALL_CAPS

# TODO CODE: Don't use magic numbers. E.g. 2 and (1, 4)
# TODO CODE: CONSTANTS_IN_CAPS, other variables in_lowercase... need to think about which variables are indeed constants

# TODO OTHER: Make the choise of environment a command line parameter, so this code is flexible for other environments


# State and action sizes for this particular environment
# TODO: make these two general, i.e. get it from env.observation_space.shape
STATE_SHAPE = (1, 4)  # input image shape (but note that we actually have to input shape (batch,105,80,4) to the model)
ACTION_SIZE = 2


def atari_model_mask():
    # With the functional API we need to define the inputs:
    states_input = layers.Input(STATE_SHAPE, name='states')
    actions_input = layers.Input((ACTION_SIZE,), name='action_mask')  # Masking!

    hidden1 = layers.Dense(32, activation='relu')(actions_input)
    hidden2 = layers.Dense(32, activation='relu')(hidden1)
    output = layers.Dense(ACTION_SIZE)(hidden2)  # Activation not specified, so it's linear activation by default?
    filtered_output = layers.Multiply(name='QValue')([output, actions_input])  # Multiply the output by the mask

    model = Model(inputs=[states_input, actions_input], outputs=filtered_output)
    model.compile(loss='logcosh', optimizer=keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01))

    return model
