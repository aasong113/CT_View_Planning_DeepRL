import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random

from gym import Env, spaces
import time

# https://blog.paperspace.com/creating-custom-environments-openai-gym/

class GoDown_Toy(gym.Env):

    """"
    This custom environment, will teach the agent to go up or down to maximize the number total intensity of pixels. 
    """

    UP = 0
    DOWN = 1

    def __init__(self, grid_size = )

    def step(self):

    def render(self):

    def reset(self):