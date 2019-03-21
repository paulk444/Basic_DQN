"""
We make a DQN agent for playing Atari games, following (and modifying/extended) the tutorial here:
https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26

This code is specifically for Breakout.
"""

import argparse
import os
import random
import time
import logging

import gym
import keras
import numpy as np
from keras import layers
from keras.models import Model

# TODO Throughout: Don't use magic numbers. E.g. 4 and (105, 80, 4)
# TODO Throughout: Python variables should be written_in_lowercase, except constants that are ALL_CAPS


# Make the model
# TODO: make these two general, i.e. get it from env.observation_space.shape:
ATARI_SHAPE = (105, 80, 4)  # input image size to model (but note that we actually have to input (batch,105,80,4))
ACTION_SIZE = 4


def atari_model_mask():
    # With the functional API we need to define the inputs.
    frames_input = layers.Input(ATARI_SHAPE, name='frames')
    actions_input = layers.Input((ACTION_SIZE,), name='action_mask')  # Masking!

    # TODO: Do normalisation outside of tensorflow (since we're not computing gradients etc) - only use the ML framework
    #  when you actually need to.
    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = layers.Lambda(lambda x: x / 255.0, name='normalization')(frames_input)

    # "The first hidden layer convolves 16 8×8 filters (stride 4) with the input & applies a rectifier nonlinearity."
    conv_1 = layers.convolutional.Conv2D(
        16, (8, 8), strides=(4, 4), activation='relu')(normalized)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = layers.convolutional.Conv2D(
        32, (4, 4), strides=(2, 2), activation='relu')(conv_1)
    # Flattening the second convolutional layer.
    conv_flattened = layers.core.Flatten()(conv_2)
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = layers.Dense(256, activation='relu')(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = layers.Dense(ACTION_SIZE)(hidden)

    # Finally, we multiply the output by the mask!
    filtered_output = layers.Multiply(name='QValue')([output, actions_input])

    model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)

    # Using logcosh loss function because it's similar to Huber, but easier to implement than a custom loss function
    model.compile(loss='logcosh', optimizer=keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01))

    return model


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
def get_epsilon_for_iteration(iteration):
    # TODO Perhaps: make it so these can be modified in main (without using global variables!)
    greedy_after = 10 ** 6  # 1e6 in the paper
    start_at = 1  # 1 in the paper
    # epsilon should be 1 for 0 iterations, 0.1 for greedy_after iterations, and 0.1 from then onwards
    if iteration > greedy_after:
        epsilon = 0.1
    else:
        # epsilon = 1-0.9*iteration/greedy_after
        epsilon = start_at - (start_at - 0.1) * iteration / greedy_after
    return epsilon


# Choose the best action
def choose_best_action(model, state):
    # TODO: Replace magic numbers in this function
    # Need state in correct form/shape
    state_batch = np.zeros((1, 105, 80, 4))
    state_batch[0, :, :, :] = state
    Q_values = model.predict([state_batch, np.ones((1, 4))])  # ASSUMING ONLY 4 ACTIONS (e.g. Breakout)
    action = np.argmax(Q_values)
    return action


""" Make a "state", which is defined as the frame and the previous 3 frames "iteration" is the index for storing frames 
in memory. So here we take the frame stored with index "iteration", and take the previous 3 frames, then make a state 
from this. (Note if iteration > memory size (mem_size, then .recall does iteration mod mem_size)
"""
def make_state(mem_frames, iteration):
    state = np.zeros((105, 80, 4))
    for i in range(4):
        state[:, :, i] = mem_frames.recall(iteration - i)  # Recalls the frame associated to iteration (and for (-i))
    return state


# RingBufSimple: the memory to store the frames (simplified from the RingBuf in the blog)
class RingBufSimple:
    def __init__(self, size):
        self.data = [None] * size
        self.size = size

    def append(self, element, iteration):
        iteration_mod_size = iteration % self.size  # Turns interation into an index in the memory [note % means 'mod']
        self.data[iteration_mod_size] = element

    def recall(self, iteration):  # Recalls the memory element in the position corresponding to "iteration"
        iteration_mod_size = iteration % self.size  # Turns interation into an index in the memory
        return self.data[iteration_mod_size]


# Copying the model
def copy_model(model):
    """Returns a copy of a keras model."""
    model.save('tmp_model_x')
    new_model = keras.models.load_model('tmp_model_x')
    os.remove('tmp_model_x')  # delete the model once it's been loaded. (Is this working correctly?)
    # TODO: Is there no way to copy a model other than saving it to disc?!
    return new_model


# Turn the actions into one-hot-encoded actions (required to use the mask)
# TODO: Only need to input actions, then use len(actions)
def into_one_hot(number_in_batch, actions):
    one_hot_actions = np.zeros((number_in_batch, ACTION_SIZE))
    for i in range(number_in_batch):
        for j in range(ACTION_SIZE):
            if j == actions[i]:
                one_hot_actions[i, j] = 1
    return one_hot_actions


# Modified slightly from the blog:
def fit_batch_target(
        model, target_model, gamma, start_states, actions, rewards, next_states, is_terminal, number_in_batch):
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

    # First, predict the Q values of the next states (using the target model):
    next_Q_values = target_model.predict([next_states, np.ones((number_in_batch, 4))])
    # The Q values of the terminal states is 0 by definition:
    next_Q_values[is_terminal] = 0
    # The Q values of each start state is the reward + gamma * the max next state Q value:
    Q_values = rewards + gamma * np.max(next_Q_values, axis=1)
    # TODO: Doublecheck: axis for max? & next_Q_values gives 4 values (one for each action) for each item in the batch?

    # TODO: into_one_hot only needs to take actions as input
    one_hot_actions = into_one_hot(number_in_batch, actions)  # Turn the actions into one-hot-encoded actions
                                                              # (required to use the mask)
    # Fit the keras model, using the mask
    model.fit([start_states, one_hot_actions], one_hot_actions * Q_values[:, None],
              epochs=1, batch_size=len(start_states), verbose=0)  # Should it be epochs=1??


def make_n_fit_batch(
        model, target_model, gamma, iteration,
        mem_size, mem_frames, mem_actions, mem_rewards, mem_is_terminal, number_in_batch):
    """Make a batch then use it to train the model"""

    if iteration < mem_size:
        # In this case, sample 'number_in_batch' numbers from 4 to 'iterations'
        # (We use 4 so that there are 5 frames minimum, so we can still make 2 states (because each state is the frame
        # plus the previous 3 frames))
        indices_chosen = random.sample(range(4, iteration), number_in_batch)
    else:
        # Now the memory is full, so we can take any elements from it
        indices_chosen = random.sample(range(0, mem_size), number_in_batch)

    # Initialise the batches
    start_states = np.zeros((number_in_batch, ) + ATARI_SHAPE)  # (Array instead of list)
    next_states = np.zeros((number_in_batch, ) + ATARI_SHAPE)  # (Array instead of list)
    actions = np.zeros((number_in_batch))  # (Array instead of list)

    rewards = list()  # List
    is_terminals = list()  # List

    for i in range(len(indices_chosen)):  # Probably more efficient way to do this: "for index in ?(indices_chosen)?"
        index = indices_chosen[i]  # index corresponds to the iterations
        start_states[i, :, :, :] = make_state(mem_frames, index - 1)
        next_states[i, :, :, :] = make_state(mem_frames, index)  # State given by index was arrived at by taking action
                                                                 # given by index and reward received is give by index.
        # In contrast, index-1 labels the previous state, before the action was taken
        actions[i] = mem_actions.recall(index)

        rewards.append(mem_rewards.recall(index))
        is_terminals.append(mem_is_terminal.recall(index))

    # We should now have a full batch, which the DNN can train on
    fit_batch_target(
        model, target_model, gamma, start_states, actions, rewards, next_states, is_terminals, number_in_batch)


def q_iteration(env, model, target_model, iteration, mem_frames, mem_actions, mem_rewards, mem_is_terminal, mem_size):
    """
    Do one iteration of acting then learning
    """

    # Choose epsilon based on the iteration
    epsilon = get_epsilon_for_iteration(iteration)

    start_state = make_state(mem_frames, iteration)
    # Choose the action:
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = choose_best_action(model, start_state)
    # Play one game iteration
    # TODO: According to the paper, you should actually play 4 times here
    new_frame, reward, is_terminal, _ = env.step(action)

    # TODO: These next 6-7 lines can be factored out into a helper function (they're duplicated below)
    reward = transform_reward(reward)  # Make reward just +1 or -1
    new_frame = preprocess(new_frame)  # Preprocess frame before saving it
    # Add to memory
    mem_frames.append(new_frame, iteration + 1)  # The frame we ended up in after the action
    mem_actions.append(action, iteration + 1)  # The action we did
    mem_rewards.append(reward, iteration + 1)  # The reward we received after doing the action
    mem_is_terminal.append(is_terminal, iteration + 1)  # Whether or not the new frame is terminal

    # Make then fit a batch (gamma=0.99, num_in_batch=?)
    number_in_batch = 32
    make_n_fit_batch(
        model, target_model, 0.99, iteration,
        mem_size, mem_frames, mem_actions, mem_rewards, mem_is_terminal, number_in_batch)

    # If DONE, reset model!
    if is_terminal:
        env.reset()

    return action, reward, is_terminal, epsilon


def main():
    """ Train the DQN to play Breakout
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--num_rand_acts', help="Random actions before learning starts",
                        default=5 * 10 ** 4, type=int)
    parser.add_argument('-s', '--save_after', help="Save after this number of training steps",
                        default=10 ** 5, type=int)
    parser.add_argument('-m', '--mem_size', help="Size of the experience replay memory",
                        default=10 ** 6, type=int)
    parser.add_argument('-sn', '--save_name', help="Name of the saved models", default=None, type=str)
    # parser.add_argument('-g', '--greedy_after', default=1e6, type=int)
    args = parser.parse_args()

    # Set up logging:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Other things to modify
    number_training_steps = 10 ** 8  # It should start doing well after 1e6??
    print_progress_after = 10 ** 2
    Copy_model_after = 10 ** 4  # 1e4 in the blog?

    number_random_actions = args.num_rand_acts  # Should be at least 33 (batch_size+1). 5e4 in the paper?
    save_model_after_steps = args.save_after  # Try 1e5?
    mem_size = args.mem_size  # 1e6 in the paper?

    logger.info(' num_rand_acts = %s, save_after = %s, mem_size = %s', number_random_actions, save_model_after_steps, mem_size)

    # Make the model
    model = atari_model_mask()
    model.summary()

    # Make the memories
    mem_frames = RingBufSimple(mem_size)
    mem_actions = RingBufSimple(mem_size)
    mem_rewards = RingBufSimple(mem_size)
    mem_is_terminal = RingBufSimple(mem_size)

    print('Setting up Breakout and pre-filling memory with random actions...')

    # Create and reset the Atari env, and process the initial screen:
    env = gym.make('BreakoutDeterministic-v4')
    env.reset()

    # TODO:  Rename i to iteration, and combined the two loops below. And factor out the random actions loop and the
    #  learning loop into two helper functions.

    # First make some random actions, and initially fill the memories with these
    for i in range(number_random_actions+1):
        iteration = i
        # Random action
        action = env.action_space.sample()
        new_frame, reward, is_terminal, _ = env.step(action)

        # TODO: These next 6-7 lines can be factored out into a helper function (they're dublicated above)
        reward = transform_reward(reward)  # Make reward just +1 or -1
        new_frame = preprocess(new_frame)  # Preprocess frame before saving it
        # Add to memory
        mem_frames.append(new_frame, iteration)  # The frame we ended up in after the action
        mem_actions.append(action, iteration)  # The action we did
        mem_rewards.append(reward, iteration)  # The reward we received after doing the action
        mem_is_terminal.append(is_terminal, iteration)  # Whether or not the new frame is terminal

    # Now do actions using the DQN, and train as we go...
    print('Finished the {} random actions...'.format(number_random_actions))
    tic = 0

    for i in range(number_training_steps):

        iteration = number_random_actions + i

        # Copy model every now and then and fit to this: makes it more stable
        if i % Copy_model_after == 0:
            target_model = copy_model(model)

        action, reward, is_terminal, epsilon = q_iteration(
            env, model, target_model, iteration, mem_frames, mem_actions, mem_rewards, mem_is_terminal, mem_size)

        # Print progress, time, and SAVE the model
        if (i + 1) % print_progress_after == 0:
            print('Training steps done: {}, Epsilon: {}'.format(i + 1, epsilon))
        if (i + 1) % save_model_after_steps == 0:
            toc = time.time()
            print('Time since last save: {}'.format(np.round(toc - tic)), end=" ")
            tic = time.time()
            # Save model:
            file_name = os.path.join('saved_models', 'Run_{}_{}'.format(args.save_name, i + 1))
            model.save(file_name)
            print('; model saved')

if __name__ == '__main__':
    main()
