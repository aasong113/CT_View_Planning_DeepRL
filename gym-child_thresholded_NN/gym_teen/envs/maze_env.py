from importlib.resources import path
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_teen.envs.maze_view_2d import MazeView2D
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION = ["N", "S", "E", "W"]

    #def __init__(self, maze_file = None, width=None,height=None, mode=None, enable_render=True):
    
    def __init__(self, maze_file = None, width=None,height=None,target_x = None, target_y = None, mode=None, enable_render=True):
    #def __init__(self,maze_file=None,grid_size=(9,9),mode=None,enable_render=True):

        self.viewer = None
        self.enable_render = enable_render
        self.isopen = True
        self.width = width
        self.height = height


        self.target_x2 = target_x
        self.target_y2 = target_y
        self.newBackground = None
        
        self.min_distance = 100000
        self.threshold_distance = 10

        # This is a distance queue that will be used to detect osciallations when the model is close to the solution. 
        # It will take the standadrd deviations of the (x,y) coordinates and if they are less than a threshold and 
        # close to the target, then the episode will be done. 
        self.distance_queue = []
        

        self.grid_view = MazeView2D(screen_size=(405,405),width=self.width,height = self.height, target_x2 = self.target_x2, target_y2 = self.target_y2)
        #self.grid_view = MazeView2D(screen_size=(405,405),width=self.width,height = self.height)
        
        self.initial_stepsize = self.grid_view.get_stepsize
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

    def set_target_xy(self, x, y):
        self.target_x2 = x
        self.target_y2 = y
        self.grid_view.set_target_pos(self.target_x2, self.target_y2)
    
    def set_background_env(self, path):
        self.newBackground = path
        self.grid_view.set_background_real(self.newBackground)
        print("New background Set!")

    def step(self, action):
        
        # D(P_{i-1}, P_t)
        distance_current = self.grid_view.euclidean_distance_from_goal
        if distance_current < self.min_distance:
            self.min_distance = distance_current

        # move the robot. 
        if isinstance(action, int):
            self.grid_view.move_robot(self.ACTION[action])
        else:
            self.grid_view.move_robot(action)

        # D(P_i, P_t)
        distance_future = self.grid_view.euclidean_distance_from_goal
        if distance_future < self.min_distance:
            self.min_distance = distance_current

        # If this Distance_current or Distance_future equals zero then we are at the desired view. 
        if distance_future <= self.threshold_distance or distance_current <= self.threshold_distance:
            reward = 1
            done = True
            # reset the stepsize 
            self.grid_view.set_stepSize(self.initial_stepsize)
            print(self.grid_view.get_current_position)
            print(f"current distance is: {distance_current}, and future distance is {distance_future}")
        else:
            # R = sign(D(Pi−1, Pt)−D(Pi, Pt))
            reward = np.sign(distance_current - distance_future )
            done = False

        # Add current position to a distance queue, this will be used to detect oscillation around either a corner or the goal. 
        self.distance_queue.append(self.grid_view.get_current_position)
        if len(self.distance_queue) >= 10:

            # if the standard deviation of all the points is small, and the distance is smaller than our threshold + a constant then its oscillating near the target.
            if np.std(self.distance_queue) <= 8 and (distance_future <= self.threshold_distance+4 or distance_current <= self.threshold_distance+4):
                reward = 1
                done = True
                # reset the stepsize 
                self.grid_view.set_stepSize(self.initial_stepsize)
                print(self.grid_view.get_current_position)
                print(f"We STOPPED USING THE OSCILLATION: current distance is: {distance_current}, and future distance is {distance_future}")

            # remove the front element. 
            self.distance_queue.pop(0)


        # Step size needs to be adaptive. 
        # if the distance current or distance future is close, then divide it by 2. 
        if distance_current <= self.min_distance/2 or distance_future <= self.min_distance/2:
            self.grid_view.set_stepSize(int(self.grid_view.get_stepsize/2))
            print(f"the stepsize has decreased and is now {self.grid_view.get_stepsize}")

        # need extra condition to make the step size 1, or complete optimal tuning. 
        if self.grid_view.get_stepsize == 2 and (distance_current <= self.min_distance or distance_future <= self.min_distance):
            self.grid_view.set_stepSize(int(self.grid_view.get_stepsize/2))
            print(f"the stepsize has decreased and is now {self.grid_view.get_stepsize}")


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


    def render_goal(self, mode = "human"):

        fig, ax = plt.subplots()
        ax.imshow(self.grid_view.update(mode))
        rect = patches.Rectangle((self.target_x2, self.target_y2), self.width, self.height, linewidth = 1, edgecolor = 'r', facecolor = 'none')
        ax.add_patch(rect)
        plt.show()

    
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
        
