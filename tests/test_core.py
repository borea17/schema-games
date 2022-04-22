from schema_games.breakout.games import StandardBreakout
import numpy as np
import matplotlib.pyplot as plt

SEED = 1
np.random.seed(SEED)


def test_GET_IMG_FROM_ENTITY_STATES_works_as_expected():
    env = StandardBreakout(return_state_as_image=False, debugging=False)
    state = env.reset()
    # start with upward moving ball to ensure that a brick is hit
    env.balls[0].velocity_index = env.velocity_to_index[(0, 1)]
    assert _check_equality(env, state)
    for i in range(100):
        action = np.random.randint(3)
        state, reward, done, new_state = env.step(action)
        assert _check_equality(env, state)
    _plot(env, state)
    
def _plot(env, state):
    fig = plt.figure(figsize=(14, 7))
    ax = plt.subplot(1, 2, 1)
    plt.title("Current State")
    img = env.convert_entity_states_to_image(state)
    plt.imshow(img)

    ax = plt.subplot(1, 2, 2)
    plt.title("Predicted Future State")
    img = env.get_img_from_entity_states(state)
    plt.imshow(img)
    plt.show()
    return fig

def _check_equality(env, state):
    img_true = env.convert_entity_states_to_image(state)
    img_calc = env.get_img_from_entity_states(state)
    return np.all(img_true == img_calc)