# Custom Env

Home made Env, works like OpenAI gym but works with multi dimensional arrays
This project is licensed under the terms of the MIT license.


# Init variables

timestamps ```list-like```
> Must be a list-like of int
  If your timestamps are in datetime format, an idea is to convert them
  into unixt timestamps
  however this can really be just any comparable
  change the codes below where I assert the typings for start and end
  into the ones that you want to use

data ```np.ndarray```
> The condition len(timestamps) == len(data) must hold

start_index, end_index ```int``` (optional)
> Set start and end time, inclusive.

step_size ```int```
> With every step, how many timesteps to move forward to next state
  Minimum 1
  Default(1)

lookback ```int```
> Signifies the number of timesteps to look back when returning state
  Try not to have lookback < step_size for obvious reasons
  Minimum 1
  Default(5)

getReward ```function(cur_state, next_state, action) => int```
> Your reward function
  Should take in 2 states and an action to output an int signifying
  the value to maximise
