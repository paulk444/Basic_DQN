
"""We make a DQN agent for playing Atari games, following, modifying, and extended the tutorial here: https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26"""

import gym
import random
import numpy as np
import time
import os

import tensorflow as tf
from tensorflow import keras
import keras
from keras import layers
from keras.models import Model
#from keras.datasets import mnist
#from keras.layers import Dense, Flatten
#from keras.layers import Conv2D, Lambda, merge
#from keras.models import Sequential
#from keras import backend as K


# Make the model

ATARI_SHAPE = (105, 80, 4)  # input image size to model (but note that we actually have to input shape: (batch,105,80,4))
ACTION_SIZE = 4

def atari_model_mask():
    # With the functional API we need to define the inputs.
    frames_input = layers.Input(ATARI_SHAPE, name='frames')
    actions_input = layers.Input((ACTION_SIZE,), name='action_mask')  # Masking!

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = layers.Lambda(lambda x: x / 255.0, name='normalization')(frames_input)

    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv_1 = layers.convolutional.Conv2D(
        16, (8, 8), strides=(4, 4), activation='relu'
    )(normalized)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = layers.convolutional.Conv2D(
        32, (4, 4), strides=(2, 2), activation='relu'
    )(conv_1)
    # Flattening the second convolutional layer.
    conv_flattened = layers.core.Flatten()(conv_2)
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = layers.Dense(256, activation='relu')(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = layers.Dense(ACTION_SIZE)(hidden)

    # Finally, we multiply the output by the mask!
    filtered_output = layers.Multiply(name='QValue')([output,
                                                      actions_input])  # Tutorial gives: filtered_output = keras.layers.merge([output, actions_input], mode='mul')

    model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)

    # Using logcosh loss function because it sounds similar to Huber, yet it's easier to implement (custon loss functions get complicated when creating the target model)
    model.compile(loss='logcosh', optimizer=keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01))

    return model

model=atari_model_mask()
model.summary()


# Make reward only -1,0,+1

def transform_reward(reward):
    return np.sign(reward)

# Preprocess the frames

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    return to_grayscale(downsample(img))


# Choose epsilon based on the iteration

greedy_after = 10**6    # 10^6 in the paper
start_at = 1            # 1 in the paper

def get_epsilon_for_iteration(iteration):

    # epsilon should be 1 for 0 iterations, 0.1 for greedy_after iterations, and 0.1 from then onwards

    if iteration > greedy_after:

        epsilon = 0.1

    else:

        # epsilon = 1-0.9*iteration/greedy_after
        epsilon = start_at - (start_at - 0.1) * iteration / greedy_after

    return epsilon


# Choose the best action

def choose_best_action(model, state):
    # Need state in correct form/shape
    state_batch = np.zeros((1, 105, 80, 4))
    state_batch[0, :, :, :] = state

    Q_values = model.predict([state_batch, np.ones((1, 4))])  # ASSUMING ONLY 4 ACTIONS

    action = np.argmax(Q_values)
    return action


# Make a "state", which is defined as the frame and the previous 3 frames
# Modified to fit with the simplified RingBuf

def make_state(mem_frames, iteration):

    state = np.zeros((105, 80, 4))

    for i in range(4):
        state[:, :, i] = mem_frames.recall(iteration-i) # Recalls the frame associated to iteration (and for (-i))
    return state


# RingBuf_simple: the memory to store the frames <-- Simplified (from the blog) so we don't worry about .start and .end

class RingBuf_simple:
    def __init__(self, size):
        self.data = [None] * (size)  # size instead of size + 1
        # start and end removed completely
        self.size = size

    def append(self, element, iteration):
        iteration_mod_size = np.mod(iteration, self.size)  # Turns interation into an index in the memory
        self.data[iteration_mod_size] = element

    def recall(self, iteration):  # Recalls the memory element in the right position for iteration
        iteration_mod_size = np.mod(iteration, self.size)  # Turns interation into an index in the memory
        return self.data[iteration_mod_size]


# Copying the model

def copy_model(model):
    """Returns a copy of a keras model."""
    model.save('tmp_model_b')
    new_model = keras.models.load_model('tmp_model_b')
    #os.remove("tmp_model_a")
    return new_model


# Modified slightly from the blog:

def fit_batch_target(model, target_model, gamma, start_states, actions, rewards, next_states, is_terminal, number_in_batch):
    """Do one deep Q learning iteration.

    Params:
    - model: The DQN
    - target_model: The target DQN (copied from the DQN every e.g. 10k iterations)
    - gamma: Discount factor (should be 0.99)
    - start_states: numpy array of starting states
    - actions: numpy array of actions corresponding to the start states (NOT one-hot encoded; normal integer encoding)
    - rewards: numpy array of rewards corresponding to the start states and actions
    - next_states: numpy array of the resulting states corresponding to the start states and actions
    - is_terminal: numpy boolean array of whether the resulting state is terminal

    """

    # First, predict the Q values of the next states (using the target model)
    next_Q_values = target_model.predict([next_states, np.ones((number_in_batch, 4))])

    # The Q values of the terminal states is 0 by definition
    next_Q_values[is_terminal] = 0
    # The Q values of each start state is the reward + gamma * the max next state Q value
    Q_values = rewards + gamma * np.max(next_Q_values, axis=1)
    # TODO: Doublecheck: correct axis for max? And that next_Q_values gives 4 values (one for each action) for each item in the batch?

    # One hot actions
    one_hot_actions = np.zeros((number_in_batch, 4))
    for i in range(number_in_batch):
        for j in range(4):
            if j == actions[i]:
                one_hot_actions[i, j] = 1

    # Fit the keras model, using the mask
    model.fit([start_states, one_hot_actions], one_hot_actions * Q_values[:, None],
        epochs=1, batch_size=len(start_states), verbose=0)


# Make a batch then use it to train the model

def make_n_fit_batch(model, target_model, gamma, iteration, mem_size, mem_frames, mem_actions, mem_rewards, mem_is_terminal, number_in_batch):

    if iteration < mem_size:
        # In this case, sample 'number_in_batch' numbers from 4 to 'iterations'
        # 4 so that there are 5 frames minimum, so we can still make 2 states (because each state is the frame plus the previous 3 frames)
        indices_chosen = random.sample(range(4, iteration), number_in_batch)

    else:
        # Now the memory is full, so we can take any elements from it
        indices_chosen = random.sample(range(0, mem_size), number_in_batch)

    # Initialise some lists (model.fit for the NN expects input as lists (or does it?!))
    start_states = np.zeros((number_in_batch, 105, 80, 4))  # Array instead of list
    next_states = np.zeros((number_in_batch, 105, 80, 4))  # Array instead of list
    actions = np.zeros((number_in_batch))  # Array instead of list

    rewards = list()  # List
    is_terminals = list()  # List

    for i in range(len(indices_chosen)): # Probably more efficient way to do this: "for index in ?(indices_chosen)?"
        index = indices_chosen[i] # index corresponds to the iterations
        start_states[i, :, :, :] = make_state(mem_frames, index - 1)
        next_states[i, :, :, :] = make_state(mem_frames, index) # state given by index was arrived at by taking action given by index and reward received is give by index
                                                                # in contrast, index-1 labels the previous state, before the action was taken
        actions[i] = mem_actions.recall(index)

        rewards.append(mem_rewards.recall(index))
        is_terminals.append(mem_is_terminal.recall(index))

    # We should now have a full batch, which the NN can train on
    fit_batch_target(model, target_model, gamma, start_states, actions, rewards, next_states, is_terminals, number_in_batch)


# Do one iteration of acting and learning

def q_iteration(env, model, target_model, iteration, mem_frames, mem_actions, mem_rewards, mem_is_terminal, mem_size):

    # Choose epsilon based on the iteration
    epsilon = get_epsilon_for_iteration(iteration)

    start_state = make_state(mem_frames, iteration)

    # Choose the action
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = choose_best_action(model, start_state)

    # Play one game iteration (note: according to some paper, you should actually play 4 times here)
    new_frame, reward, is_terminal, _ = env.step(action)

    reward = transform_reward(reward)  # Make reward just +1 or -1
    new_frame = preprocess(new_frame)  # Preprocess frame before saving it

    # Add to memory
    mem_frames.append(new_frame, iteration+1)  # The frame we ended up in after the action
    mem_actions.append(action, iteration+1)  # The action we did
    mem_rewards.append(reward, iteration+1)  # The reward we received after doing the action
    mem_is_terminal.append(is_terminal, iteration+1)  # Whether or not the new frame is terminal (assuming is_terminal is the same as is_terminal)

    # Make then fit a batch (gamma=0.99, num_in_batch=?)
    number_in_batch = 32
    make_n_fit_batch(model, target_model, 0.99, iteration, mem_size, mem_frames, mem_actions, mem_rewards, mem_is_terminal, number_in_batch)

    # If DONE, reset model!
    if is_terminal:
        env.reset()

    return action, reward, is_terminal, epsilon


# Now run the DQN to learn Atari!

number_random_actions = 5*10**4 # Should be at least 36. Try 5*10**4?
number_training_steps = 10 ** 8 # It should start doing well after 10**7??
save_model_after_steps = 10**5 # Try 10**5?
mem_size = 10**6 # Try 10**6

# Make the memories
mem_frames = RingBuf_simple(mem_size)
mem_actions = RingBuf_simple(mem_size)
mem_rewards = RingBuf_simple(mem_size)
mem_is_terminal = RingBuf_simple(mem_size)

# Create and reset the Atari env, and process the initial screen
env = gym.make('BreakoutDeterministic-v4')
# Probably don't need these 3 lines:
env.reset()
prev_screen = env.render(mode='rgb_array')
preproc_screen = preprocess(prev_screen)

## First make some random actions, and initially fill the memories with these
for i in range(number_random_actions+1):

    iteration = i
    # Random action
    action = env.action_space.sample()
    new_frame, reward, is_terminal, _ = env.step(action)

    reward = transform_reward(reward)  # Make reward just +1 or -1
    new_frame = preprocess(new_frame)  # Preprocess frame before saving it

    # Add to memory
    mem_frames.append(new_frame, iteration)  # The frame we ended up in after the action
    mem_actions.append(action, iteration)  # The action we did
    mem_rewards.append(reward, iteration)  # The reward we received after doing the action
    mem_is_terminal.append(is_terminal, iteration)  # Whether or not the new frame is terminal (assuming is_terminal is the same as is_terminal)


## Now do actions using the NN, and train as we go...
# (Initialise the model (the NN) <-- this has already been done above)
tic = 0
print('Finished the', number_random_actions, 'random actions...')

for i in range(number_training_steps):

    iteration = number_random_actions+i

    # Copy model every now and then and fit to this: makes it more stable
    if np.mod(i, 10**4) == 0:
        target_model = copy_model(model)

    action, reward, is_terminal, epsilon = q_iteration(env, model, target_model, iteration, mem_frames, mem_actions, mem_rewards, mem_is_terminal, mem_size)

    # Print progress, time, and SAVE the model
    if np.mod(i+1, 10**2) == 0:
        print('Training steps done: ', i+1,', Action = ',action,', Reward = ',reward,', Terminal = ',is_terminal,', Epsilon',epsilon)
    if np.mod(i+1, save_model_after_steps) == 0:
        toc = time.time()
        print('Time: ', np.round(toc - tic), end=" ")  # (The end thing makes it print on the same line)
        tic = time.time()
        # Save model
        file_name = 'saved_models/Mac1_'
        file_name += str(i + 1)
        model.save(file_name)
        print(', model saved')

