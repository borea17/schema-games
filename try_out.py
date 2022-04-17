from schema_games.breakout import games

import matplotlib.pyplot as plt

# from gym.utils.play import play

environment_class = "StandardBreakout"
env_args = {"return_state_as_image": False, "debugging": False}

env = games.StandardBreakout(**env_args)
state = env.reset()


possible_shapes = []
possible_positions = []
positions_x = []
positions_y = []
for elem in state:
    position = list(state[elem])[0][1]
    position_x = position[0]
    position_y = position[1]
    shape = list(state[elem])[1][1]
    if shape not in possible_shapes:
        possible_shapes.append(shape)
    if position not in possible_positions:
        possible_positions.append(position)
    if position_x not in positions_x:
        positions_x.append(position_x)
    if position_y not in positions_y:
        positions_y.append(position_y)


positions_x.sort()
positions_y.sort()

print("Shapes are ", possible_shapes)
print(f"There are {len(possible_positions)} positions")
print("X positions", positions_x)
print("Y positions", positions_y)


############### PLAYING #################
for i in range(10):
    action = 0
    state, reward, done, new_state = env.step(action)

    keys_to_select = list(range(500))
    sub_state = {}
    for key, value in state.items():
        if key in keys_to_select:
            sub_state[key] = value
    img = env._get_image()
    
    img = env.convert_entity_states_to_image(sub_state)

    import pdb 
    pdb.set_trace()

    # state = env.step(0)
    #state = env.step(0)


def preprocessing(entity_states_over_time):
    """
                


        `given a dataset of entity states over time, we preprocess the entity states into 
         a representation that is more convenient for learning`
    """
    pass
