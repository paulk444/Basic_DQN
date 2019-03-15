
"""
We make a DQN agent for learning in gym environments, following (and modifying/extended) the tutorial here:
https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26

This code is specifically for CartPole.
"""

import gym
import random
import numpy as np
import time
import os
from tensorflow import keras
from keras import layers
from keras.models import Model

# TODO: Pre-process states to make them numpy arrays: "state = np.array([state])"
# TODO: Try different loss (mse?) and optimiser (Adam?)

# Make the model

STATE_SHAPE = (1, 4)  # input image size to model (but note that we actually have to input shape: (batch,105,80,4))
ACTION_SIZE = 2


def atari_model_mask():
    # With the functional API we need to define the inputs.
    states_input = layers.Input(STATE_SHAPE, name='states')
    actions_input = layers.Input((ACTION_SIZE,), name='action_mask')  # Masking!
    hidden1 = layers.Dense(32, activation='relu')(actions_input)
    hidden2 = layers.Dense(32, activation='relu')(hidden1)
    output = layers.Dense(ACTION_SIZE)(hidden2)  # Activation not specified, so it's linear activation by default?
    # Finally, we multiply the output by the mask!
    filtered_output = layers.Multiply(name='QValue')([output, actions_input])
    model = Model(inputs=[states_input, actions_input], outputs=filtered_output)
    model.compile(loss='logcosh', optimizer=keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01))

    return model
