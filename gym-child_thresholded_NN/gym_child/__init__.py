from gym.envs.registration import register


register(
    id='maze-v0',
    entry_point='gym_child.envs:MazeEnvSample5x5',
    max_episode_steps=2000,
)

