from config import *
import random

# 'S': start
# 'G': goal
# ' ': empty cell
# '#': mountain
# 'O': water
# 'A': airport (only two allowed in total)

class GridworldEnv():
    def __init__(self):
        """
        Initialize Gridworld
        """
        self.reset()


    def shape(self):
        """
        Returns the shape (number of rows and columns) of the Gridworld environment
        :return: (number of rows, number of columns)
        """
        return len(self.grid), len(self.grid[0])


    def num_states(self):
        """
        Returns the number of states of the Gridworld environment
        :return: number of states (rows times columns)
        """
        return len(self.grid) * len(self.grid[0])


    def num_actions(self):
        """
        Returns the number of possible actions (UP/RIGHT/DOWN/LEFT)
        :return:
        """
        return 4


    def step(self, action):
        """
        Execute a step with the agent in the Gridworld environment. Perform an action and obtain an observation,
        a reward and the information (done) whether the agent ended up in a terminal state. Do not use this function
        for value iteration / policy iteration but the function below.
        :param action: Scalar value of the action (UP/RIGHT/DOWN/LEFT) according to config file
        :return: (state, reward, done flag)
        """
        # use this function normally
        self.state = self._calc_next_state(action)
        reward = self._calc_reward()
        done = self._calc_done()
        obs = self._state_to_obs(self.state)
        return obs, reward, done


    def step_dp(self, obs, action):
        """
        Modified step function used for value iteration / policy iteration:
        Execute a step with the agent in the Gridworld environment. Perform an action for a given state
        and obtain an observation, a reward and the information (done) whether the agent ended up in a terminal state
        :param obs: Scalar value of the state
        :param action: Scalar value of the action (UP/RIGHT/DOWN/LEFT) according to config file
        :return: (state, reward, done flag)
        """
        # use this function only for value iteration / policy iteration
        self.state = self._obs_to_state(obs)
        return self.step(action)


    def reset(self):
        """
        Resets the environment, must be called at the beginning of every episode
        """
        m, n = self.shape()

        # find start state
        for x in range(m):
            for y in range(n):
                if self.grid[x][y] == 'S':
                    self.state = (x, y)
                    return self._state_to_obs(self.state)

        raise ValueError("No start state found")


    def _calc_next_state(self, action):
        m, n = self.shape()
        x, y = self.state

        # hole/goal -> stuck
        if self.grid[x][y] == 'O' or self.grid[x][y] == 'G':
            return x, y

        # calculate movement direction
        if action == G_LEFT:
            x_n, y_n = x, y - 1
        elif action == G_UP:
            x_n, y_n = x - 1, y
        elif action == G_RIGHT:
            x_n, y_n = x, y + 1
        elif action == G_DOWN:
            x_n, y_n = x + 1, y
        else:
            raise ValueError('Unknown action: ' + str(action))

        # movement limited by map boundaries
        x_n = min(max(x_n, 0), m - 1)
        y_n = min(max(y_n, 0), n - 1)

        # movement limited by wall boundaries
        if self.grid[x_n][y_n] == '#':
            x_n, y_n = x, y

        # fly to other airport
        if self.grid[x_n][y_n] == 'A':
            for a in range(len(self.grid)):
                for b in range(len(self.grid[a])):
                    if self.grid[a][b] == 'A' and a != x_n and b != y_n:
                        x_a = a
                        y_a = b
            x_n = x_a
            y_n = y_a

        return x_n, y_n


    # calculate reward flag
    def _calc_reward(self):
        x, y = self.state
        if self.grid[x][y] == 'G':
            return self.g_reward
        elif self.grid[x][y] == 'O':
            return self.o_reward
        else:
            return self.step_cost


    # calculate done flag
    def _calc_done(self):
        x, y = self.state
        if self.grid[x][y] in ('O', 'G'):
            return True
        else:
            return False


    # convert from observation (int) to internal state representation (x,y)
    def _obs_to_state(self, obs):
        n = len(self.grid[0])
        x = obs // n
        y = obs % n
        return x, y


    # convert from internal state representation (x,y) to observation (int)
    def _state_to_obs(self, state):
        n = len(self.grid[0])
        x, y = state
        obs = x * n + y
        return obs


class Test(GridworldEnv):
    def __init__(self):
        self.step_cost = -1
        self.g_reward = -1
        self.o_reward = -100
        self.grid = [['S', ' ', 'A'],
                     [' ', ' ', ' '],
                     ['A', ' ', 'G']]
        GridworldEnv.__init__(self)


class EmptyWorld33(GridworldEnv):
    def __init__(self):
        self.step_cost = -1
        self.g_reward = -1
        self.o_reward = -100
        self.grid = [['S', ' ', ' '],
                     [' ', ' ', ' '],
                     [' ', ' ', 'G']]
        GridworldEnv.__init__(self)


class EmptyWorld55(GridworldEnv):
    def __init__(self):
        self.step_cost = -1
        self.g_reward = -1
        self.o_reward = -100
        self.grid = [['S', ' ', ' ', ' ', ' '],
                     [' ', ' ', ' ', ' ', ' '],
                     [' ', ' ', ' ', ' ', ' '],
                     [' ', ' ', ' ', ' ', ' '],
                     [' ', ' ', ' ', ' ', 'G']]
        GridworldEnv.__init__(self)


class MountainWorld(GridworldEnv):
    def __init__(self):
        self.step_cost = -1
        self.g_reward = -1
        self.o_reward = -100
        self.grid = [['S', ' ', '#', ' ', ' '],
                     [' ', ' ', '#', ' ', ' '],
                     [' ', ' ', ' ', ' ', ' '],
                     [' ', ' ', '#', ' ', ' '],
                     [' ', ' ', '#', ' ', 'G']]
        GridworldEnv.__init__(self)


class WaterWorld(GridworldEnv):
    def __init__(self):
        self.step_cost = -1
        self.g_reward = -1
        self.o_reward = -100
        self.grid = [['S', ' ', 'O', ' ', ' '],
                     [' ', ' ', 'O', ' ', ' '],
                     [' ', ' ', ' ', ' ', ' '],
                     [' ', ' ', 'O', ' ', ' '],
                     [' ', ' ', 'O', ' ', 'G']]
        GridworldEnv.__init__(self)


class Cliff(GridworldEnv):
    def __init__(self):
        self.step_cost = -1
        self.g_reward = -1
        self.o_reward = -100
        self.grid = [[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                     [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                     [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                     ['S', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'G']]
        GridworldEnv.__init__(self)


class ExerciseWorld(GridworldEnv):
    def __init__(self):
        self.step_cost = -1
        self.g_reward = -1
        self.o_reward = -100
        self.grid = [['S', ' ', ' ', ' ', 'A', ' ', 'O'],
                     [' ', ' ', '#', ' ', ' ', ' ', ' '],
                     ['A', ' ', ' ', ' ', '#', ' ', 'G']]
        GridworldEnv.__init__(self)


class MazeWater(GridworldEnv):
    def __init__(self):
        self.step_cost = 0
        self.g_reward = 100
        self.o_reward = -100
        self.grid = [[' ', ' ', ' ', 'O', ' ', ' ', ' ', 'O', ' ', ' ', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     ['S', 'O', ' ', ' ', ' ', 'O', ' ', ' ', ' ', 'O', 'G']]
        GridworldEnv.__init__(self)


class MazeWater2(GridworldEnv):
    def __init__(self):
        self.step_cost = 0
        self.g_reward = 100
        self.o_reward = -100
        self.grid = [[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                     [' ', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ' '],
                     [' ', 'O', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                     [' ', 'O', ' ', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                     [' ', 'O', ' ', 'O', ' ', ' ', ' ', 'O', ' ', ' ', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', ' ', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', 'O', 'O', 'O', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', ' ', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', ' ', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', 'O', 'O', 'O', 'O', ' ', 'O', ' '],
                     ['S', 'O', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'O', 'G']]
        GridworldEnv.__init__(self)


class MazeWall(GridworldEnv):
    def __init__(self):
        self.step_cost = 0
        self.g_reward = 100
        self.o_reward = -100
        self.grid = [[' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     ['S', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', 'G']]
        GridworldEnv.__init__(self)


class EmptyWorldNN(GridworldEnv):
    # create an empty quadratic grid with the start in one corner and the goal in the opposite corner
    def __init__(self, size=10):
        self.step_cost = 0
        self.g_reward = 100
        self.o_reward = -100
        self.grid = [[' '] * size for _ in range(size)]
        self.grid[0][0] = 'S'
        self.grid[size-1][size-1] = 'G'
        GridworldEnv.__init__(self)


class Random(GridworldEnv):
    # create a random quadratic environment with a given percentage of all cells covered by water/mountains
    def __init__(self, size=10, water=0.1, mountain=0.1):
        self.step_cost = 0
        self.g_reward = 100
        self.o_reward = -100

        # create new environments until one with a valid start-goal path is found
        valid_grid = False
        while not valid_grid:
            self._create_grid(size, water, mountain)
            valid_grid = self._check_valid_grid()

        GridworldEnv.__init__(self)


    # create the environment
    def _create_grid(self, size, water, mountain):
        self.grid = [[' ']*size for _ in range(size)]

        # set water cells
        for i in range(int(size**2*water)):
            self.grid[random.randint(0,size-1)][random.randint(0,size-1)] = 'O'

        # set mountain cells
        for i in range(int(size**2*mountain)):
            self.grid[random.randint(0,size-1)][random.randint(0,size-1)] = '#'

        # define start
        while True:
            r_start, c_start = random.randint(0,size-1), random.randint(0,size-1)
            # prevent start from being in the center of the map
            if abs(r_start-size/2) < size/3 and abs(c_start-size/2) < size/3:
                continue
            break

        # define goal
        while True:
            r_end, c_end = random.randint(0,size-1), random.randint(0,size-1)
            # prevent goal from being in the center of the map
            if abs(r_end-size/2) < size/3 and abs(r_end-size/2) < size/3:
                continue
            # prevent goal from being too close to start
            if abs(r_end-r_start) < size/2 and abs(c_end-c_start) < size/2:
                continue
            if r_end != r_start or c_end != c_start:
                break

        # clear area around start/goal from obstacles
        clear_pixel = 1
        for r, c in ((r_start, c_start), (r_end, c_end)):
            for i in range(max(0, r-clear_pixel), min(size, r+clear_pixel+1)):
                for k in range(max(0, c-clear_pixel), min(size, c+clear_pixel+1)):
                    self.grid[i][k] = ' '

        # set start/goal
        self.grid[r_start][c_start] = 'S'
        self.grid[r_end][c_end] = 'G'


    # check if a path from start to goal can be found
    def _check_valid_grid(self):
        size = len(self.grid)
        visited = [[False]*size for _ in range(size)]

        def dfs(row, col):
            if row < 0 or col < 0 or row >= size or col >= size or \
                    self.grid[row][col] in ['#', 'O'] or \
                    visited[row][col] is True:
                return False

            visited[row][col] = True

            if self.grid[row][col] == 'G' or \
                    dfs(row + 1, col) or \
                    dfs(row - 1, col) or \
                    dfs(row, col + 1) or \
                    dfs(row, col - 1):
                return True

            return False

        for row in range(size):
            for col in range(size):
                if self.grid[row][col] == 'S':
                    return dfs(row,col)

        return ValueError("No start found")


if __name__ == "__main__":
    r = Test()

    for i in range(9):
        x, y = r._obs_to_state(i)
        r.state = (x,y)
        for action in range(4):
            x_n, y_n = r._calc_next_state(action)
            print('{} {} {} {} {}'.format(x, y, action, x_n, y_n))
