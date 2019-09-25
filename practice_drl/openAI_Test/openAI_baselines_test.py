import gym

from baselines import deepq


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def train():
    env = gym.make("CartPole-v0")
    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=100,
        buffer_size=50,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")

def play(num_play=10):
    env = gym.make("CartPole-v0")
    act = deepq.learn(env, network='mlp', total_timesteps=0, load_path="cartpole_model.pkl")
    run = 0
    while run <= num_play:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew

            # if done, play for some extra time steps (100 here) to see the falling effect
            if done:
                extra_step = 0
                while done and extra_step <= 100:
                    env.render()
                    obs, rew, done, _ = env.step(act(obs[None])[0])
                    extra_step = extra_step + 1

        print("Episode reward", episode_rew)
        run += 1


if __name__ == '__main__':

    # train()

    play(num_play=10)