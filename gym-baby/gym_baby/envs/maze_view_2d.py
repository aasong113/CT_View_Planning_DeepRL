import pygame
import random
import numpy as np
import os


class MazeView2D:

    def __init__(self, maze_name="toy_example",
                 grid_size=(10, 10), screen_size=(600, 600),
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
        self.__entrance = np.zeros(2, dtype=int)

        # Set the Goal
        #self.__goal = np.array(self.maze_size) - np.array((1, 1))
        # for now hardcoded because i know what i'm loading in, hypo-thetically
        self.__goal = np.array([6,4])
        #self.__goal = np.array([1,2])

        # Create the Robot
        self.__robot = self.entrance

        if self.__enable_render is True:
            # Create a background
            self.background = pygame.Surface(self.screen.get_size()).convert()
            self.image = pygame.image.load(r'./simple_midRight_10.png')
            self.image = pygame.transform.scale(self.image,(self.screen.get_size()))
            
            self.background.fill((255, 255, 255))
            self.background.blit(self.image,(0,0))

            # Create a grid layzer
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
        self.__robot = np.zeros(2, dtype=int)
        self.__draw_robot(transparency=255)

    def __controller_update(self):
        if not self.__game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.__game_over = True
                    self.quit_game()

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



    def __draw_robot(self, colour=(0, 0, 150), transparency=255):

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
        
        
        
        x = int(self.__robot[0] * self.CELL_W +1)
        y = int(self.__robot[1] * self.CELL_H +1)
        w = int(self.CELL_W -1)
        h = int(self.CELL_H-1 )

        
        #r = int(min(self.CELL_W, self.CELL_H)/5 + 0.5
        pygame.draw.rect(self.grid_layer,colour + (transparency,),(x,y,w,h))
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

        #x = int(cell[0] * self.CELL_W + 0.5 + 1)
        #y = int(cell[1] * self.CELL_H + 0.5 + 1)
        #w = int(self.CELL_W + 0.5 - 1)
        #h = int(self.CELL_H + 0.5 - 1)
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
    def get_value(self):
        imx = int(self.SCREEN_W - self.__robot[0]*self.CELL_W)
        imy = int(self.SCREEN_H - self.__robot[1]*self.CELL_H)
        #print(self.x,self.y)
        #print(self.image.get_at((imx,imy)))
        #imx = int(self.__robot[0])
        #imy = int(self.__robot[1])
        return np.mean(self.image.get_at((imx,imy)))
    
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
        "N": (0, -1),
        "E": (1, 0),
        "S": (0, 1),
        "W": (-1, 0)
    }

    def __init__(self, maze_cells=None, grid_size=(10,10), has_loops=True):

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
            # check if the wall is opened
            #this_wall = bool(self.get_walls_status(self.maze_cells[cell_id[0], cell_id[1]])[dir])
            #other_wall = bool(self.get_walls_status(self.maze_cells[x1, y1])[self.__get_opposite_wall(dir)])
            #return this_wall or other_wall
            return True
        return False

    def is_breakable(self, cell_id, dir):
        # check if it would be out-of-bound
        x1 = cell_id[0] + self.COMPASS[dir][0]
        y1 = cell_id[1] + self.COMPASS[dir][1]

        return not self.is_open(cell_id, dir) and self.is_within_bound(x1, y1)

    def is_within_bound(self, x, y):
        # true if cell is still within bounds after the move
        return 0 <= x < self.MAZE_W and 0 <= y < self.MAZE_H



    @property
    def MAZE_W(self):
        return int(self.grid_size[0])

    @property
    def MAZE_H(self):
        return int(self.grid_size[1])

    @classmethod
    def get_walls_status(cls, cell):
        walls = {
            "N" : (cell & 0x1) >> 0,
            "E" : (cell & 0x2) >> 1,
            "S" : (cell & 0x4) >> 2,
            "W" : (cell & 0x8) >> 3,
        }
        return walls

    @classmethod
    def all_walls_intact(cls, cell):
        return cell & 0xF == 0

    @classmethod
    def num_walls_broken(cls, cell):
        walls = cls.get_walls_status(cell)
        num_broken = 0
        for wall_broken in walls.values():
            num_broken += wall_broken
        return num_broken

    @classmethod
    def __break_walls(cls, cell, dirs):
        if "N" in dirs:
            cell |= 0x1
        if "E" in dirs:
            cell |= 0x2
        if "S" in dirs:
            cell |= 0x4
        if "W" in dirs:
            cell |= 0x8
        return cell

    @classmethod
    def __get_opposite_wall(cls, dirs):

        if not isinstance(dirs, str):
            raise TypeError("dirs must be a str.")

        opposite_dirs = ""

        for dir in dirs:
            if dir == "N":
                opposite_dir = "S"
            elif dir == "S":
                opposite_dir = "N"
            elif dir == "E":
                opposite_dir = "W"
            elif dir == "W":
                opposite_dir = "E"
            else:
                raise ValueError("The only valid directions are (N, S, E, W).")

            opposite_dirs += opposite_dir

        return opposite_dirs



if __name__ == "__main__":

    maze = MazeView2D(screen_size= (600, 600), grid_size=(10,10))
    maze.update()
    input("Enter any key to quit.")


