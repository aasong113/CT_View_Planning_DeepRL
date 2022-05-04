import pygame
import random
import numpy as np
import os


class MazeView2D:

    def __init__(self, maze_name="cadaver_example",
                 grid_size=(9, 9), screen_size=(405, 405),
                 has_loops=False, enable_render=True):

        # PyGame configurations
        pygame.init()
        pygame.display.set_caption(maze_name)
        self.clock = pygame.time.Clock()
        self.__game_over = False
        self.__enable_render = enable_render

        
        self.__grid = Grid(grid_size = grid_size,has_loops=has_loops)


        self.grid_size = grid_size
        if self.__enable_render is True:
            # to show the right and bottom border
            self.screen = pygame.display.set_mode(screen_size)
            self.__screen_size = tuple(map(sum, zip(screen_size, (-1, -1))))

        # Set the starting point
        x_start = int(random.randrange(grid_size[0]))
        y_start = int(random.randrange(grid_size[1]))
        #self.__entrance = np.zeros(2, dtype=int)
        self.__entrance = [x_start,y_start]

        # Create the Robot
        self.__robot = self.entrance
        self.x = self.__robot[0]
        self.y = self.__robot[1]

        if self.__enable_render is True:
            # Create a background
            self.background = pygame.Surface(self.screen.get_size()).convert()
            # Change the background of the image
            #self.image = pygame.image.load(r'./cad_117_threshNoBin.png')
            self.image = pygame.image.load(r'./cadaver_slice_117_grey.png')
            self.image = pygame.transform.scale(self.image,(self.screen.get_size()))
            
            self.background.fill((255, 255, 255))
            self.background.blit(self.image,(0,0))

            # Create a grid layzer
            self.grid_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
            self.grid_layer.fill((0, 0, 0, 0,))

            # show the robot
            self.__draw_robot()


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
        if dir not in self.__grid.COMPASS.keys():
            raise ValueError("dir cannot be %s. The only valid dirs are %s."
                             % (str(dir), str(self.__grid.COMPASS.keys())))

        if self.__grid.is_open(self.__robot, dir):

            # update the drawing
            self.__draw_robot(transparency=0)

            # move the robot
            self.__robot += np.array(self.__grid.COMPASS[dir])
            self.__draw_robot(transparency=255)

    def reset_robot(self):

        self.__draw_robot(transparency=0)
        #self.__robot = np.zeros(2, dtype=int)
        self.__robot = self.entrance
        self.__draw_robot(transparency=255)
        
        
        
    def new_entrance(self):

        #self.__draw_entrance(transparency=0)
        #self.__robot = np.zeros(2, dtype=int)
        x_start = int(random.randrange(self.__grid.MAZE_W))
        y_start = int(random.randrange(self.__grid.MAZE_H))
        self.__entrance = [x_start,y_start]
        #self.__draw_entrance(transparency=255)

    def __controller_update(self):
        if not self.__game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.__game_over = True
                    self.quit_game()

    def __view_update(self, mode="human"):
        if not self.__game_over:
            # update the robot's position
            #self.__draw_entrance()
            self.__draw_robot()
            #self.__draw_goal()
            # update the screen
            self.screen.blit(self.image, (0, 0))
            self.screen.blit(self.grid_layer,(0, 0))

            if mode == "human":
                pygame.display.flip()

            return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))



    def __draw_robot(self, colour=(200, 0, 0), transparency=255):

        if self.__enable_render is False:
            return
        
        
        
        self.x = int(self.__robot[0] * self.CELL_W +1)
        self.y = int(self.__robot[1] * self.CELL_H +1)
        w = int(self.CELL_W -1)
        h = int(self.CELL_H-1 )

        
        #r = int(min(self.CELL_W, self.CELL_H)/5 + 0.5)
        #pygame.draw.circle(self.grid_layer, (0,0,150)+(transparency,), (self.__robot[0],self.__robot[1]),r)
        #pygame.draw.circle(self.grid_layer,(0,0,150)+(transparency,), (self.x,self.y),r)
        #pygame.draw.circle(self.grid_layer,(150,0,0)+(transparency,), (self.x+w,self.y+h),r)
        pygame.draw.rect(self.grid_layer,colour + (transparency,),(self.x,self.y,w,h),width=2)
        #pygame.draw.circle(self.maze_layer, colour + (transparency,), (x, y), r)

    def __draw_entrance(self, colour=(0, 0, 150), transparency=235):

        self.__colour_cell(self.entrance, colour=colour, transparency=transparency)
        
        
    def __draw_goal(self, colour=(0, 0, 150), transparency=100):

        self.__colour_cell(self.goal, colour=colour, transparency=transparency)

    def __colour_cell(self, cell, colour, transparency):

        if self.__enable_render is False:
            return

        if not (isinstance(cell, (list, tuple, np.ndarray)) and len(cell) == 2):
            raise TypeError("cell must a be a tuple, list, or numpy array of size 2")

        x = int(cell[0]*self.CELL_W +1)
        y = int(cell[1]*self.CELL_H +1)
        w = int(self.CELL_W -1)
        h = int(self.CELL_H-1)
        pygame.draw.rect(self.grid_layer, colour + (transparency,), (x, y, w, h))

    @property
    def grid(self):
        return self.__grid

    @property
    def robot(self):
        return self.__robot

    @property
    def get_avg_value(self):
        x0 = self.x
        y0 = self.y
        npix = int(self.CELL_H*self.CELL_W)
        
        totval = 0
        for i in range(int(self.CELL_W)):
            iix = int(x0+i)
            for j in range(int(self.CELL_H)):
                iij = int(y0+j)
                totval+= self.__get_value((iix,iij))
                
        return totval/npix
        
    
    
    #@property
    def __get_value(self,pt):
        # since this is a grey image, we can do:
        return self.image.get_at(pt)[0]
    
    @property
    def entrance(self):
        return self.__entrance
    
    @property
    def goal(self):
        return self.__goal

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


class Grid:
    COMPASS = {
            0: (0, -1),
            1: (1, 0),
            2: (0, 1),
            3: (-1, 0)
        }
    # COMPASS = {
    #     "N": (0, -1),
    #     "E": (1, 0),
    #     "S": (0, 1),
    #     "W": (-1, 0)
    # }

    def __init__(self, maze_cells=None, grid_size=(15,15), has_loops=True):

        # maze member variables
        self.maze_cells = maze_cells
        self.has_loops = has_loops


        # Use existing one if exists
        if self.maze_cells is not None:
            if isinstance(self.maze_cells, (np.ndarray, np.generic)) and len(self.maze_cells.shape) == 2:
                self.grid_size = tuple(maze_cells.shape)
            else:
                raise ValueError("maze_cells must be a 2D NumPy array.")
        # Otherwise, generate a random one
        else:
            # maze's configuration parameters
            if not (isinstance(grid_size, (list, tuple)) and len(grid_size) == 2):
                raise ValueError("maze_size must be a tuple: (width, height).")
            self.grid_size = grid_size

            #self._generate_maze()



    def is_open(self, cell_id, dir):
        # check if it would be out-of-bound
        x1 = cell_id[0] + self.COMPASS[dir][0]
        y1 = cell_id[1] + self.COMPASS[dir][1]

        # if cell is still within bounds after the move
        if self.is_within_bound(x1, y1):
            return True
        return False


    def is_within_bound(self, x, y):
        # true if cell is still within bounds after the move
        return 0 <= x < self.MAZE_W and 0 <= y < self.MAZE_H



    @property
    def MAZE_W(self):
        return int(self.grid_size[0])

    @property
    def MAZE_H(self):
        return int(self.grid_size[1])




if __name__ == "__main__":

    maze = MazeView2D(screen_size= (405, 405), grid_size=(9,9))
    maze.update()
    input("Enter any key to quit.")


