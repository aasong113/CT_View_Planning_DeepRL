import numpy as np

import gym
import time
from gym import error, spaces, utils
from gym.utils import seeding
from gym_child.envs.maze_view_2d import MazeView2D


class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    #ACTION = ["N", "S", "E", "W"]
    ACTION = [0, 1, 2, 3]

    def __init__(self, maze_file = None, grid_size=None, mode=None, enable_render=True):

        self.viewer = None
        self.enable_render = enable_render
        self.isopen = True


        #self.grid_view = MazeView2D()
        self.grid_view = MazeView2D(screen_size=(405,405),grid_size=(9,9))
        self.grid_size = self.grid_view.grid_size

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2*len(self.grid_size))

        # observation is the x, y coordinate of the grid
        low = np.zeros(len(self.grid_size), dtype=int)
        high =  np.array(self.grid_size, dtype=int) - np.ones(len(self.grid_size), dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        # initial condition
        self.state = None
        self.steps_beyond_done = None

        # Simulation related variables.
        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self.configure()

    def __del__(self):
        if self.enable_render is True:
            self.grid_view.quit_game()

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if isinstance(action, int):
            self.grid_view.move_robot(self.ACTION[action])
        else:
            self.grid_view.move_robot(action)
        
        
        # negative parabola - not as steep ascent (actually descent) closer to values
        #reward = -(20-((self.grid_view.get_avg_value)**2)/20)
        
        # i like this one better. 
        # had to hardcode some limits based on the image... not ideal, you know. but here we are
        
        if 38 < self.grid_view.get_avg_value < 44:
            reward = 1
            done = True

        else:
            # need to make sure this is always negative!!!
            reward = -(50-np.exp(self.grid_view.get_avg_value/12))
            done = False
            
          
        
        #reward = -(0.2-((self.grid_view.get_avg_value)**2)/0.2)

        self.state = self.grid_view.robot

        info = {}
        #done = False
        return self.state, reward, done, info

    def reset(self):
        self.grid_view.new_entrance()
        self.grid_view.reset_robot()
        self.state = np.zeros(2)
        self.steps_beyond_done = None
        self.done = False
        return self.state

    def is_game_over(self):
        return self.grid_view.game_over

    def render(self, mode="human", close=False):
        if close:
            self.grid_view.quit_game()

        return self.grid_view.update(mode)
    
    def close(self):
        import pygame
        pygame.display.quit()
        pygame.quit()
        #if self.screen is not None:
        #    import pygame

         #   pygame.display.quit()
         #   pygame.quit()
         #   self.isopen = False

class MazeEnvSample5x5(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample5x5, self).__init__(maze_file="maze2d_5x5.npy", enable_render=enable_render)
