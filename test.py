

import gym
env = gym.make('Humanoid-v2')

n_eps = 200

rewards = []
for i_episode in range(n_eps):
    env.reset()
    sum_reward = 0
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        print(action)
        observation, reward, done, info = env.step(action)
        # break
        sum_reward += reward
        if(t % 100 == 0):
            pass
            #print(t)
        if done or t == 999:
            #print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            #print("Reward: {}".format(sum_reward))
            rewards.append(sum_reward)
        if done:
            break
