from schema_games.breakout import games

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

env_args = {"return_state_as_image": False, "debugging": False}
env = games.StandardBreakout(**env_args)
state = env.reset()
env.balls[0].velocity_index = env.velocity_to_index[(0, 1)]


possible_shapes = []
possible_positions = []
positions_x = []
positions_y = []
colors = []
for elem in state:
    position = list(state[elem])[0][1]
    position_x = position[0]
    position_y = position[1]
    shape = list(state[elem])[1][1]
    color = list(state[elem])[2][1]

    if shape not in possible_shapes:
        possible_shapes.append(shape)
    if position not in possible_positions:
        possible_positions.append(position)
    if position_x not in positions_x:
        positions_x.append(position_x)
    if position_y not in positions_y:
        positions_y.append(position_y)
    if color not in colors:
        colors.append(color)

fig = plt.figure(figsize=(14, 7))

ax_1 = plt.subplot(1, 2, 1)
plt.title("Current State")
ax_2 = plt.subplot(1, 2, 2)
plt.title("Predicted State")

############### PLAYING #################
def animate(i):
    action = np.random.randint(3)
    state, reward, done, new_state = env.step(action)
    img = env.convert_entity_states_to_image(state)
    ax_1.imshow(img)

    img = env.get_img_from_entity_states(state)
    ax_2.imshow(img)

animator = animation.FuncAnimation(fig, animate, interval=1)
plt.show()


