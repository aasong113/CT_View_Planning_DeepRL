from gym.envs.registration import register


register(
    id='maze-v0',
    entry_point='gym_teen.envs:MazeEnvSample5x5',
    max_episode_steps=10000,
)


