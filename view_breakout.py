
"""
This script loads a pre-trained model, then runs Breakout to view the model playing.
"""

import argparse
import random
import time

import gym
import numpy as np
from keras.models import load_model

from teach_breakout import choose_best_action, preprocess


def main():
    """
    Use a pre-trained model to play Breakout
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', help="Name of the model to be used", default='Trained_models/8M_steps', type=str)
    args = parser.parse_args()

    # Load the model!
    model = load_model(args.model_name)
    print('Loaded model: ',args.model_name)
    #
    random_action_fraction = 0.05

    number_viewing_steps = 2000

    print('Running for ',number_viewing_steps,' viewing steps, with ',random_action_fraction,' of the actions being random.')

    # Reset the Atari env, and process the initial screen
    env = gym.make('BreakoutDeterministic-v4')
    env.reset()
    prev_screen = env.render(mode='rgb_array')
    frame = preprocess(prev_screen)

    # TODO: Remove magic numbers here:
    state = np.zeros((105, 80, 4))
    new_state = np.zeros((105, 80, 4))


    # First make up to 10 random actions, so that each time we view the model it's different

    number_random_this_episode = random.sample(range(1, 10), 1)[0]
    print('First doing ',number_random_this_episode+4,' random actions')
    for i in range(number_random_this_episode):
        action = env.action_space.sample()
        _, _, _, _ = env.step(action)


    # Now make 4 more random actions, to make the first state (the model takes a state as input)
    # TODO: For loop instead of this silly cutting and pasting!!
    state[:,:,3] = frame
    # Random action
    action = env.action_space.sample()
    frame, _ , _ , _ = env.step(action)
    frame = preprocess(frame)
    # View
    env.render()
    time.sleep(0.01)  # Pause, as it's too fast otherwise
    # Add to state
    state[:,:,2] = frame
    # Random action
    action = env.action_space.sample()
    frame, _ , _ , _ = env.step(action)
    frame = preprocess(frame)
    # View
    env.render()
    time.sleep(0.02)  # Pause, as it's too fast otherwise
    # Add to state
    state[:, :, 1] = frame
    # Random action
    action = env.action_space.sample()
    frame, _ , _ , _ = env.step(action)
    frame = preprocess(frame)
    # View
    env.render()
    time.sleep(0.02)  # Pause, as it's too fast otherwise
    # Add to state
    state[:, :, 0] = frame


    # We now have an initial state, so we can feed it into the model to choose the best action

    for i in range(number_viewing_steps):

        # TODO: Normally in evaulating the model we just do greedy. The reason I do it here is because sometimes it
        #  doesn't ask for a new ball... but there's a better way to do this!
        # Choose, make and view action
        if random.random() < random_action_fraction:
            action = env.action_space.sample()
        else:
            action = choose_best_action(model, state)
        #print(action)
        frame, _ , is_done, _ = env.step(action)
        frame = preprocess(frame)
        env.render() # View
        time.sleep(0.03) # Pause, as it's too fast otherwise

        # TODO: Replace with for loop
        new_state[:,:,0] = frame
        new_state[:, :, 1] = state[:, :, 0]
        new_state[:, :, 2] = state[:, :, 1]
        new_state[:, :, 3] = state[:, :, 2]

        state = new_state

        # We now have a new state...

        if is_done:
            break


if __name__ == '__main__':
    main()
