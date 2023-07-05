import gym
from ppo_torch import Agent

if __name__ == '__main__':
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    agent = Agent(n_actions=env.action_space.n, input_dims=env.observation_space.shape)
    agent.load_models()  # Load the saved models

    n_games = 10

    for i in range(n_games):
        observation = env.reset()
        print("Printing..............", observation)
        done = False
        score = 0

        while not done:
            print(observation)
            env.render()  # Render the environment (optional)
            action, _, _ = agent.choose_action(observation[0])
            observation, reward, done, info = env.step(action)
            score += reward

        print('Game:', i+1, 'Score:', score)

    env.close()