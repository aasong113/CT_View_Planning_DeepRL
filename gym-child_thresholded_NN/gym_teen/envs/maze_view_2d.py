from typing_extensions import Self
import pygame
import random
import numpy as np
import os
#from gym_teen.envs.maze_env import MazeEnv


class MazeView2D:

    def __init__(self, maze_name="2D CT example",
                 width=9, height=9, step_size = 10, screen_size=(405, 405),
                 has_loops=False, enable_render=True):

        # PyGame configurations
        pygame.init()
        pygame.display.set_caption(maze_name)
        self.clock = pygame.time.Clock()
        self.__game_over = False
        self.__enable_render = enable_render

        self.__width = width
        self.__height = height
        self.__stepsize = step_size

        #self.MazeEnvParams = MazeEnv
        # need to figure out how to get this from the MazeEnv Class. 
        self.target_x = 162
        self.target_y = 136

        self.COMPASS = {
        "N": (0, -1*self.__stepsize),
        "E": (1*self.__stepsize, 0),
        "S": (0, 1*self.__stepsize),
        "W": (-1*self.__stepsize, 0),
        
    }
        #self.__grid = Grid(grid_size = grid_size,has_loops=has_loops)


        #self.grid_size = grid_size
        if self.__enable_render is True:
            # to show the right and bottom border
            self.screen = pygame.display.set_mode(screen_size)
            self.__screen_size = tuple(map(sum, zip(screen_size, (-1, -1))))

        x_start = int(random.randrange(screen_size[0]))
        y_start = int(random.randrange(screen_size[1]))
        # Set the starting point
        #self.__entrance = np.zeros(2,dtype=int)
        self.__entrance = [x_start,y_start]

        # Set the Goal
        #self.__goal = np.array(self.maze_size) - np.array((1, 1))
        # for now hardcoded because i know what i'm loading in, hypo-thetically
        #self.__goal = np.array([6,4])
        #self.__goal = np.array([1,2])

        # Create the Robot
        self.__robot = self.entrance
        self.x = self.__robot[0]
        self.y = self.__robot[1]

        if self.__enable_render is True:
            # Create a background
            self.background = pygame.Surface(self.screen.get_size()).convert()
            self.image = pygame.image.load(r'./cadaver_slice_117_grey.png')
            self.image = pygame.transform.scale(self.image,(self.screen.get_size()))
            
            self.background.fill((255, 255, 255))
            self.background.blit(self.image,(0,0))

            # Create a operational layzer
            self.grid_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
            self.grid_layer.fill((0, 0, 0, 0,))

            # show the maze
            #self.__draw_grid()

            # show the robot
            self.__draw_robot()
            
            # wah
            #self.__draw_goal()

    def update(self, mode="human"):
        try:
            img_output = self.__view_update(mode)
            self.__controller_update()
        except Exception as e:
            self.__game_over = True
            self.quit_game()
            raise e
        else:
            return img_output

    def quit_game(self):
        try:
            self.__game_over = True
            if self.__enable_render is True:
                import pygame
                pygame.display.quit()
                pygame.quit()
            #pygame.quit()
        except Exception:
            pass

    def move_robot(self, dir):
        if dir not in self.COMPASS.keys():
            raise ValueError("dir cannot be %s. The only valid dirs are %s."
                             % (str(dir), str(self.COMPASS.keys())))
            
        tx = int(self.__robot[0])
        ty = int(self.__robot[1])
        w = int(self.width)
        h = int(self.height)

        
        #x2 = int(self.__robot[0]*self.width +1+w)
        tx1 = int(tx+w)
        tx2 = int(ty+h)
        
        if ((tx==0) or (ty==0) or (tx1==self.SCREEN_W-1) or (tx2==self.SCREEN_H-1)):
            #print('ouch')
            #tuple([10*x for x in img.size])
            #dir = tuple([10*x for x in self.])
            toadd = tuple([-5*x for x in self.COMPASS[dir]])
            #print(dir)
        else:
            toadd = self.COMPASS[dir]

        
        if self.__valid_move(self.__robot,toadd):

            # update the drawing
            self.__draw_robot(transparency=0)

            # move the robot
            #self.__robot += np.array(self.COMPASS[dir])
            self.__robot+= np.array(toadd)
            self.__draw_robot(transparency=255)

    def reset_robot(self):

        self.__draw_robot(transparency=0)
        #self.__robot = np.zeros(2, dtype=int)
        self.__robot = self.entrance
        self.__draw_robot(transparency=255)

    def __controller_update(self):
        if not self.__game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.__game_over = True
                    self.quit_game()
                    
    def new_entrance(self):
        self.__draw_entrance(transparency=0)
        x_start = int(random.randrange(0+self.width+10, self.SCREEN_W-self.width-10))
        y_start = int(random.randrange(0+self.height+10, self.SCREEN_H-self.height-10))
        self.__entrance = [x_start,y_start]
        self.__draw_entrance(transparency=235)

    def __view_update(self, mode="human"):
        if not self.__game_over:
            # update the robot's position
            self.__draw_entrance()
            self.__draw_robot()
            #self.__draw_goal()
            # update the screen
            self.screen.blit(self.image, (0, 0))
            self.screen.blit(self.grid_layer,(0, 0))

            if mode == "human":
                pygame.display.flip()

            return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))



    def __draw_robot(self, colour=(150, 0, 0), transparency=255):

        if self.__enable_render is False:
            return
        
        #x = int(self.__robot[0] * self.CELL_W + self.CELL_W * 0.5 + 0.5)
        #y = int(self.__robot[1] * self.CELL_H + self.CELL_H * 0.5 + 0.5)
        #w = int(self.CELL_W + 0.5 -1)
        #h = int(self.CELL_H + 0.5 -1)
        
        #x = int(self.__robot[0] * self.CELL_W + 0.5 + 1)
        #y = int(self.__robot[1] * self.CELL_H + 0.5 + 1)
        #w = int(self.CELL_W + 0.5 - 1)
        #h = int(self.CELL_H + 0.5 - 1)
        
        
        
        #x0 = int(self.__robot[0] * self.width +1+5)
        #y0 = int(self.__robot[1] * self.height +1)
        
        #self.x = int(self.__robot[0] *self.width + 1)
        #self.y = int(self.__robot[1] *self.height + 1)
        self.x = int(self.__robot[0]+1)
        self.y = int(self.__robot[1]+1)
        w = int(self.width -1)
        h = int(self.height-1)
        
        x3 = int(self.x+w)
        y3 = self.y
        
        #x2 = int(self.__robot[0]*self.width +1+w)
        x2 = int(self.x+w)
        y2 = int(self.y+h)
        
        #x1 = int(self.__robot[0]*self.height +1)
        x1 = self.x
        y1 = int(self.y+h)
        
        pygame.draw.polygon(self.grid_layer, colour+ (transparency,), [(self.x,self.y),(x1,y1),(x2,y2),(x3,y3)],width=2)




    def __draw_entrance(self, colour=(0, 0, 150), transparency=235):
        self.__colour_cell(self.entrance,colour=colour,transparency=transparency)
        
        
        #if self.__valid_move(self.entrance):
        #    self.__colour_cell(self.entrance, colour=colour, transparency=transparency)
        #else:
        #    self.entrance = self.new_entrance()
        
        
    def __draw_goal(self, colour=(0, 0, 150), transparency=100):

        self.__colour_cell(self.goal, colour=colour, transparency=transparency)

    def __colour_cell(self, cell, colour, transparency):

        if self.__enable_render is False:
            return

        if not (isinstance(cell, (list, tuple, np.ndarray)) and len(cell) == 2):
            raise TypeError("cell must a be a tuple, list, or numpy array of size 2")

        #x = int(cell[0] * self.CELL_W + 0.5 + 1)
        #y = int(cell[1] * self.CELL_H + 0.5 + 1)
        #w = int(self.CELL_W + 0.5 - 1)
        #h = int(self.CELL_H + 0.5 - 1)
        x = int(cell[0]*self.width +1)
        y = int(cell[1]*self.height +1)
        w = int(self.width -1)
        h = int(self.height-1)
        pygame.draw.rect(self.grid_layer, colour + (transparency,), (x, y, w, h))


    def __valid_move(self,cell_id,toadd):
        
        # need to check if any of the points will be out of bounds... 
        x0 = cell_id[0] + toadd[0]
        y0 = cell_id[1] + toadd[1]
        
        x2 = cell_id[0]+self.width + toadd[0]
        y2 = cell_id[1]+self.height + toadd[1]
        
        
        x3 = x2
        y3 = y0
        
        #x1 = int(self.__robot[0]*self.height +1)
        x1 = x0
        y1 = y2
        
        
        
        if ((self.is_within_bound(x1,y1)) and (self.is_within_bound(x2,y2)) and (self.is_within_bound(x0,y0))
           and (self.is_within_bound(x3,y3))):
            return True
        return False
        
        
    def is_within_bound(self,x,y):
        # true if cell is still within bounds after move
        return 0 <= x < self.SCREEN_W and 0 <= y < self.SCREEN_H

    @property
    def robot(self):
        return self.__robot
    
    
    @property
    def get_avg_value(self):
        x0 = self.x
        y0 = self.y
        npix = int(self.height*self.width)
        totval = 0
        for i in range(int(self.width)):
            iix = int(x0+i)
            for j in range(int(self.height)):
                iij = int(y0+j)
                totval += self.get_value((iix,iij))
        return totval/npix
        
    #TODO
    @property
    #Reward = Sign( D(P_{i-1}, P_t) - D(P_i, P_t) )

    def euclidean_distance_from_goal(self):#, MazeEnv.target_x, MazeEnv.target_y ):        

        # #find the center coordinate
        # self.x = int(self.__robot[0]+1)
        # self.y = int(self.__robot[1]+1)

        return  np.sqrt((self.x-self.target_x)**2+(self.y-self.target_y)**2)



    @property
    def get_value(self,pt):
        #print(pt)
        #imx = int(self.SCREEN_W - self.__robot[0]*self.CELL_W)
        #imy = int(self.SCREEN_H - self.__robot[1]*self.CELL_H)
        #return np.mean(self.image.get_at((imx,imy)))
        
        #since this is a grey image, we can do:
        return self.image.get_at(pt)[0]
    
    @property
    def entrance(self):
        return self.__entrance
    
    @property 
    def get_current_position(self):
        return (self.x, self.y)

    @property
    def get_stepsize(self):
        return self.__stepsize
    

    @property
    def game_over(self):
        return self.__game_over

    @property
    def SCREEN_SIZE(self):
        return tuple(self.__screen_size)

    @property
    def SCREEN_W(self):
        return int(self.SCREEN_SIZE[0])

    @property
    def SCREEN_H(self):
        return int(self.SCREEN_SIZE[1])

    @property
    def CELL_W(self):
        return float(self.SCREEN_W) / float(self.grid.MAZE_W)

    @property
    def CELL_H(self):
        return float(self.SCREEN_H) / float(self.grid.MAZE_H)
    
    @property
    def height(self):
        return self.__height
    @property
    def width(self):
        return self.__width
    
    @height.setter
    def height(self,new_val):
        self.__height = new_val
        
    @width.setter
    def width(self,new_val):
        self.__width = new_val

    def set_stepSize(self, new_val):
        self.__stepsize = new_val




if __name__ == "__main__":

    maze = MazeView2D(screen_size= (405, 405), width=9,height=9)
    maze.update()
    input("Enter any key to quit.")


