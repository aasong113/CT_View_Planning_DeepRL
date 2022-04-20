import gym
from DQN_test import Agent
import numpy as np
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
                  input_dims=[8], lr=0.001)
    scores, eps_history = [], []
    n_games = 1
    
    print(env.action_space)
#     for i in range(n_games):
#         score = 0
#         done = False
#         observation = env.reset()
#         while not done:
#             action = agent.choose_action(observation)
#             observation_, reward, done, info = env.step(action)
#             score += reward
#             agent.store_transition(observation, action, reward, 
#                                     observation_, done)
#             agent.learn()
#             observation = observation_
#         scores.append(score)
#         eps_history.append(agent.epsilon)

#         avg_score = np.mean(scores[-100:])

#         print('episode ', i, 'score %.2f' % score,
#                 'average score %.2f' % avg_score,
#                 'epsilon %.2f' % agent.epsilon)
#     x = [i+1 for i in range(n_games)]
#     filename = 'lunar_lander.png'
#     plotLearning(x, scores, eps_history, filename)


#     # create figure and axis objects with subplots()
# fig,ax = plt.subplots()
# # make a plot
# ax.plot(x, scores, color="red", marker="o")
# # set x-axis label
# ax.set_xlabel("Training Steps",fontsize=14)
# # set y-axis label
# ax.set_ylabel("Scores",color="red",fontsize=14)

# # twin object for two different y-axis on the sample plot
# ax2=ax.twinx()
# # make a plot with different y-axis using second axis object
# ax2.plot(x, eps_history,color="blue",marker="o")
# ax2.set_ylabel("Epsilon History",color="blue",fontsize=14)
# plt.show()