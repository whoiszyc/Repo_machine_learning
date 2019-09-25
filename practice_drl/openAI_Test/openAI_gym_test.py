
# First
# To test the installation of gym
# (1) open cmd
# (2) cd /Users/whoiszyc/Github/gym
# (3) pytest

# Second
# Let’s understand above code line by line.
# 1 Imported gym package.
# 2 Created ‘CartPole’ environment.
# 3 Reset the environment.
# 4 Running a loop to do several actions to play the game. For now, let’s play as much as we can. That’s why trying here to play up to 1000 steps max.
# 5 env.render() — This is for rendering the game. So, that we can see what actually happens when we are taking any steps/actions.
# 6 Getting a random action which we are going to take. Here “env.action_space.sample()” code will give us a random action which is allowed to play this game.
# 7 Doing that random action through the step function. This will return us observation, reward, done, info.
# 8–13. Printing them to know what we did, what exactly happened, what reward we got and whether game completed or not.
# 14–15. If the game is completed/done stop taking next step/action.

import gym
env = gym.make('CartPole-v1')
env.reset()
for step_index in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print("Step {}:".format(step_index))
    print("action: {}".format(action))
    print("observation: {}".format(observation))
    print("reward: {}".format(reward))
    print("done: {}".format(done))
    print("info: {}".format(info))
    # if done:
    #     break