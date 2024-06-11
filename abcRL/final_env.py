import abc_py as abcPy
import numpy as np
import graph as G
import torch
from dgl.nn.pytorch import GraphConv
import dgl
from gym import spaces

class EnvGraph(object):
    """
    # @brief the overall concept of environment, the different. use the compress2rs as target
    """
    def __init__(self, verilogfile):
        self._graph = G.extract_dgl_graph(verilogfile)
        self.__step_count = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        self.state = np.random.uniform(low=-10, high=10, size=(1,))

    def get_step_count(self) -> int:
        """Return the step count

        Returns:
            int
        """
        return self.__step_count

    def get_action_space(self) -> int:
        """Return the action space

        Returns:
            int
        """
        return 4

    def get_state_space(self) -> int:
        """Return the state space

        Returns:
            int
        """
        return len(self.__state_list)

    def __is_valid_state(self, state_coord: tuple) -> bool:
        """Check if the state is valid (within the maze and not a wall)

        Args:
            state_coord (tuple)

        Returns:
            bool
        """
        if state_coord[0] < 0 or state_coord[0] >= self.__maze.shape[0]:
            return False
        if state_coord[1] < 0 or state_coord[1] >= self.__maze.shape[1]:
            return False
        if self.__maze[state_coord[0], state_coord[1]] == 1:
            return False
        return True

    def reset(self):
        self.lenSeq = 0
        self._abc.end()
        self._abc.start()
        self._abc.read(self._aigfile)
        self._lastStats = self._abc.aigStats() # The initial AIG statistics
        self._curStats = self._lastStats # the current AIG statistics
        self.lastAct = self.numActions() - 1
        self.lastAct2 = self.numActions() - 1
        self.lastAct3 = self.numActions() - 1
        self.lastAct4 = self.numActions() - 1
        self.actsTaken = np.zeros(self.numActions())
        return self.state()
    
    def __get_next_state(self, state_coord: tuple, action: int) -> tuple:
        """Get the next state given the current state and action

        Args:
            state_coord (tuple)
            action (Action)

        Returns:
            tuple: next_state_coord
            print('')
        """
        next_state_coord = np.array(state_coord)
        if action == 0:
            next_state_coord[0] -= 1
        elif action == 1:
            next_state_coord[0] += 1
        elif action == 2:
            next_state_coord[1] -= 1
        elif action == 3:
            next_state_coord[1] += 1
        if not self.__is_valid_state(next_state_coord):
            next_state_coord = state_coord
        return tuple(next_state_coord)
    def reward(self):
        reward = 0
        if (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.lev < self._lastStats.lev):
            reward += 0.5
        elif (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.lev == self._lastStats.lev):
            reward += 0.2
        elif (self._curStats.numAnd == self._lastStats.numAnd and self._curStats.lev < self._lastStats.lev):
            reward += 0.2
        else:
            reward += -0.2
        if self.lastAct == 5: #term
            reward+= 0
        # print("hi: ", reward, (self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline)*10)
        return (self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline) + reward/20
        return reward
        #return -self._lastStats.numAnd + self._curStats.numAnd - 1
        
    def step(self) -> tuple:
        """Take a step in the environment
        Returns:
            tuple: next_state, reward, done
        """
        action = self.__traj[self.__step_count]
        state_coord = self.__state_list[self.__current_state]
        if self.__is_goal_state(state_coord):
            next_init_state = self.reset()
            return next_init_state, self.__goal_reward, True
        if self.__is_trap_state(state_coord):
            next_init_state = self.reset()
            return next_init_state, self.__trap_reward, True

        next_state_coord = self.__get_next_state(state_coord, action)
        next_state = self.__state_list.index(next_state_coord)
        self.__current_state = next_state
        self.__step_count += 1
        return next_state, self.__step_reward, False

    def reset(self) -> int:
        """Reset the current step"""
        self.__current_state = self.__init_pos_list[np.random.randint(len(self.__init_pos_list))]
        return self.__current_state

    def reset_step_count(self):
        self.__step_count = 0

    def numActions(self):
        return 5
    def dimState(self):
        return 4 + self.numActions() * 1 + 1
    def returns(self):
        return [self._curStats.numAnd , self._curStats.lev]
    def statValue(self, stat):
        return float(stat.lev)  / float(self.initLev)
        return float(stat.numAnd)  / float(self.initNumAnd) #  + float(stat.lev)  / float(self.initLev)
        #return stat.numAnd + stat.lev * 10
    def curStatsValue(self):
        return self.statValue(self._curStats)
    def seed(self, sd):
        pass
    def compress2rs(self):
        self._abc.compress2rs()
