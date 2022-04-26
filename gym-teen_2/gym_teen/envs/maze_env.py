import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_teen.envs.maze_view_2d import MazeView2D


class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self, maze_file = None, width=None,height=None, mode=None, enable_render=True):
    #def __init__(self,maze_file=None,grid_size=(9,9),mode=None,enable_render=True):

        self.viewer = None
        self.enable_render = enable_render
        self.isopen = True
        self.width = width
        self.height = height

        self.grid_view = MazeView2D(screen_size=(405,405),width=self.width,height = self.height)
        self.grid_size = self.grid_view.SCREEN_SIZE
        self.maxval = -100000

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

            
        #if np.array_equal(self.grid_view.robot, self.grid_view.goal):
        ## dude idk. 
        #if 47 < self.grid_view.get_avg_value < 50:
        #print(self.grid_view.get_avg_value)
        #if 115 < self.grid_view.get_avg_value:
        if self.grid_view.get_avg_value > 130:
            #self.grid_view.__draw_robot(transparency=0)
            reward = 1
            done = True
        else:
            #reward = -0.1*(self.grid_view.get_value)
            #reward = -(50-np.exp(self.grid_view.get_avg_value/30))
            reward = -(100-np.exp(self.grid_view.get_avg_value/32))
            done = False

        self.state = self.grid_view.robot

        info = {}

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

    #def __init__(self,width,height,enable_render=True):
        #self.width = width
        #self.height = height
        #super(MazeEnvSample5x5, self).__init__(maze_file="maze2d_5x5.npy", width=self.width, height=self.height, enable_render=enable_render)
        
    def __init__(self,width,height,enable_render=True):
        super(MazeEnvSample5x5,self).__init__(maze_file="maze2d_5x5.npy",width=width,height=height,enable_render=enable_render)
        
