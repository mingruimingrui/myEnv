import os
import types

import numpy as np

def isListLike(x):
    return isinstance(x, np.ndarray) | isinstance(x, list) | isinstance(x, tuple)

def isAscending(l):
    return all(l[i] <= l[i+1] for i in range(len(l)-1))

class Env:
    """
    Home made env, works like OpenAI gym but works with multi dimensional arrays

    timestamps: list-like
        Must be a list-like of int
        If your timestamps are in datetime format, an idea is to convert them
        into unixt timestamps
        however this can really be just any comparable
        change the codes below where I assert the typings for start and end
        into the ones that you want to use

    data: np.ndarray
        The condition len(timestamps) == len(data) must hold

    start_index, end_index: int (optional)
        Set start and end time, inclusive.

    step_size: int
        With every step, how many timesteps to move forward to next state
        Minimum 1
        Default(1)

    lookback: int
        Signifies the number of timesteps to look back when returning state
        Try not to have lookback < step_size for obvious reasons
        Minimum 1
        Default(5)

    getReward: function(cur_state, next_state, action) => int
        Your reward function
        Should take in 2 states and an action to output an int signifying
        the value to maximise
    """

    def __init__(self, timestamps, data, start_index=None, end_index=None,
        step_size=1, lookback=5, getReward):

        assert isListLike(timestamps), 'timestamps must be list-like'
        assert isAscending(timestamps), 'timestamps must be ascending'
        assert isinstance(data, np.ndarray), 'data must be np.ndarray object'
        assert len(timestamps) == len(data), 'there must be just as many timestamps as data entries'

        if start is not None:
            assert isinstance(start, int), 'start must be int'
            data = data[timestamps >= start]
            timestamps = timestamps[timestamps >= start]

        if end is not None:
            assert isinstance(end, int), 'end must be int'
            data = data[timestamps <= end]
            timestamps = timestamps[timestamps <= end]

        assert isinstance(step_size, int), 'step_size must be an int'
        assert step_size >= 1, 'step_size must be 1 or greater'

        assert isinstance(lookback, int), 'lookback must be an int'
        assert lookback >= 1, 'lookback must be 1 or greater'

        assert isinstance(getReward, types.FunctionType), 'getReward must be a function'

        assert len(timestamps) >= lookback + step_size, 'not enough timesteps'

        self.timestamps = timestamps
        self.data = data
        self.step_size = step_size
        self.lookback = lookback
        self.cur_time_index = lookback - 1
        self.cur_data_index = 0
        self.cur_time = timestamps[self.cur_time_index]
        self.next_time = timestamps[self.cur_time_index + self.step_size]
        self.cur_state = self.data[self.cur_data_index:(self.cur_data_index + self.lookback)]

        print('New Env initiated')
        print('Timestamps from', timestamps[0], 'to', timestamps[len(timestamps)-1])
        print('Step size:', step_size)
        print('Lookback:', lookback)

    def step(self, action):
        """
        action: <your condition in getReward>
        """

        assert self.next_time is not None, 'no more steps left in env, please reset'

        # generate new state
        next_state = self.data.copy()[
            (self.cur_data_index + self.step_size):
            (self.cur_data_index + self.step_size + self.lookback)
        ]

        # calculate reward
        reward = getReward(self.cur_state, next_state, action)

        # check if done
        done = self.cur_time_index + self.step_size >= len(self.timestamps)

        # update state
        self.cur_time_index = self.cur_time_index + self.step_size
        self.cur_data_index = self.cur_data_index + self.step_size
        self.cur_time = timestamps[self.cur_time_index]
        self.cur_state = next_state

        if done:
            self.next_time = None
        else:
            self.next_time = timestamps[self.cur_time_index + self.step_size]

        return next_state, reward, done

    def reset():
        """
        resets your env with same init values
        """
        print('Env reset')
        self.cur_time_index = lookback - 1
        self.cur_data_index = 0
        self.cur_time = timestamps[self.cur_time_index]
        self.next_time = timestamps[self.cur_time_index + self.step_size]
        self.cur_state = self.data[self.cur_data_index:(self.cur_data_index + self.lookback)]
