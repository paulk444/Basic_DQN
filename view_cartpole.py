
"""
This script loads a pre-trained model, then runs Cartpole to view the model playing.
"""

import argparse
import random
import time

import gym
from keras.models import load_model

from teach_cartpole import choose_best_action


def main():
    """
    Use a pre-trained model to play Cartpole
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', help="Name of the model to be used",
                        default='saved_models/Run_None_80000', type=str)
    args = parser.parse_args()

    # Load the model!
    model = load_model(args.model_name)
    print('Loaded model: ',args.model_name)

    random_action_fraction = 0
    number_viewing_steps = 2000

    print('Running for ',number_viewing_steps,' viewing steps, with ',random_action_fraction,' actions being random.')

    # Reset the Atari env, and process the initial screen
    env = gym.make('CartPole-v1')
    env.reset()
    prev_screen = env.render(mode='rgb_array')

    # First make up to n random actions, so that each time we view the model it's different
    n=20

    number_random_this_episode = random.sample(range(1, n), 1)[0]
    print('First doing ', number_random_this_episode,' random actions')
    for i in range(number_random_this_episode):
        action = env.action_space.sample()
        state, _, _, _ = env.step(action)
        env.render()
        time.sleep(0.1)

    score = 0

    # We now have an initial state, so we can feed it into the model to choose the best action
    for i in range(number_viewing_steps):

        # TODO: Normally in evaulating the model we just do greedy...
        # Choose, make and view action
        if random.random() < random_action_fraction:
            action = env.action_space.sample()
        else:
            action = choose_best_action(model, state)
        #print(action)
        state, reward , is_terminal, _ = env.step(action)
        env.render() # View
        score += reward
        print(score)
        time.sleep(0.1) # Pause, as it's too fast otherwise

        if is_terminal:
            break


if __name__ == '__main__':
    main()
