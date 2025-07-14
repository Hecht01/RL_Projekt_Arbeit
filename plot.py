from config import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pathEffects


def plot_v_table(env, v_table, policy=None):
    '''
    Plot V-function for a given gridworld environment

    :param env: GridworldEnv environment
    :param v_table: 1D Numpy array with shape (number of states). State order must be consistent with
    observations given by the GridworldEnv._state_to_obs function
    :param policy: (optional) 2D Numpy array with shape (number of states, number of actions). State order must be consistent with
    observations given by the GridworldEnv._state_to_obs function. Action order must be consistent with
    G_UP / G_RIGHT / G_DOWN / G_LEFT as described in the config.py file
    '''
    plot_table(env=env, table=v_table, policy=policy)


def plot_q_table(env, q_table, policy=None):
    '''
    Plot Q-function for a given gridworld environment

    :param env: GridWorld environment
    :param q_table: 2D Numpy array with shape (number of states, number of actions). State order must be consistent with
    observations given by the GridworldEnv._state_to_obs function. Action order must be consistent with
    G_UP / G_RIGHT / G_DOWN / G_LEFT as described in the config.py file
    :param policy: (optional) 2D Numpy array with shape (number of states, number of actions). State order must be consistent with
    observations given by the GridworldEnv._state_to_obs function. Action order must be consistent with
    G_UP / G_RIGHT / G_DOWN / G_LEFT as described in the config.py file
    '''
    plot_table(env=env, table=q_table, policy=policy)


def plot_table(env, table, policy):
    m, n = env.shape()

    fig, ax = plt.subplots(figsize=(n, m))
    normalized_table = normalize_table(env, table)

    for x in range(m):
        for y in range(n):
            if env.grid[x][y].upper() in ['#', 'O', 'G']:
                draw_state_type(x=y, y=-x, type=env.grid[x][y])
            else:
                idx = env._state_to_obs((x, y))

                if table.ndim == 2:
                    # plot Q values
                    for action in range(env.num_actions()):
                        draw_q_polygon(x=y, y=-x, action=action, values=normalized_table[idx])
                        draw_q_value(x=y, y=-x, action=action, values=table[idx])
                else:
                    # plot V values
                    draw_v_polygon(x=y, y=-x, value=normalized_table[idx])
                    draw_v_value(x=y, y=-x, value=table[idx])

                # plot resulting policy
                if policy is not None:
                    for action in range(env.num_actions()):
                        draw_policy(x=y, y=-x, action=action, values=policy[idx])

                if env.grid[x][y].upper() in ['A', 'S']:
                    draw_state_type(x=y, y=-x, type=env.grid[x][y])

    ax.axis('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def normalize_table(env, table):
    # normalize V-/Q-values to [0,1] range for plotting
    # consider also cases where the minimum/maximum values are
    # smaller/larger than the ones of the goal / water states
    if np.min(table) < min(env.o_reward, env.g_reward) * 1.01:
        a = np.min(table)
    else:
        a = np.min(table[table > min(env.o_reward, env.g_reward) * 0.99])

    if np.max(table) > max(env.o_reward, env.g_reward) * 1.01:
        A = np.max(table)
    else:
        A = np.max(table[table < max(env.o_reward, env.g_reward) * 0.99])

    return (table - a) / (A - a + 1e-9)


def map_value_to_color(value):
    # maps a value [0,1] to a suitable color (0-red, 0.5-yellow, 1-green)

    if value <= 0:
        return np.array([1, 0, 0])
    elif value <= 0.5:
        return np.array([1, 0, 0]) + 2 * value * np.array([0, 1, 0])
    elif value <= 1:
        return np.array([1, 1, 0]) + 2 * (value - 0.5) * np.array([-1, 0, 0])
    else:
        return np.array([0, 1, 0])


def draw_policy(x, y, action, values):
    if values[action] < 1e-3:
        return

    LENGTH = 0.25
    if action == G_UP:
        dx = 0
        dy = LENGTH
    elif action == G_RIGHT:  # right
        dx = LENGTH
        dy = 0
    elif action == G_DOWN:  # down
        dx = 0
        dy = -LENGTH
    elif action == G_LEFT:  # left
        dx = -LENGTH
        dy = 0

    value = values[action]
    plt.arrow(x, y, dx, dy, head_length=LENGTH, width = value/20)


def draw_q_polygon(x, y, action, values):
    if action == G_UP:
        xs = [x - 0.5, x + 0.5, x]
        ys = [y + 0.5, y + 0.5, y]
    elif action == G_RIGHT:  # right
        xs = [x + 0.5, x + 0.5, x]
        ys = [y - 0.5, y + 0.5, y]
    elif action == G_DOWN:  # down
        xs = [x - 0.5, x + 0.5, x]
        ys = [y - 0.5, y - 0.5, y]
    elif action == G_LEFT:  # left
        xs = [x - 0.5, x - 0.5, x]
        ys = [y - 0.5, y + 0.5, y]

    color = map_value_to_color(values[action])
    plt.fill(xs, ys, facecolor=color)


def draw_q_value(x, y, action, values):
    OFFSET = 0.25
    dx, dy = 0, 0
    if action == G_UP:  # up
        dy = OFFSET
    elif action == G_RIGHT:  # right
        dx = OFFSET
    elif action == G_DOWN:  # down
        dy = -OFFSET
    elif action == G_LEFT:  # left
        dx = -OFFSET

    value = values[action]

    color = "w"
    if value == np.max(values):
        color = "deepskyblue"
    text = plt.text(x + dx, y + dy, value.round(decimals=2), ha="center", va="center", color="k")
    text.set_path_effects([pathEffects.Stroke(linewidth=8, foreground=color), pathEffects.Normal()])


def draw_v_polygon(x, y, value):
    color = map_value_to_color(value)
    rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, facecolor=color)
    plt.gca().add_patch(rect)


def draw_v_value(x, y, value):
    text = plt.text(x, y, value.round(decimals=2), ha="center", va="center", color="k")
    text.set_path_effects([pathEffects.Stroke(linewidth=8, foreground="w"), pathEffects.Normal()])


def draw_state_type(x, y, type):
    if type == 'S' or type == 'A':
        facecolor='none'
        offset = -0.15
    else:
        facecolor='w'
        offset = 0

    rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, facecolor=facecolor)
    plt.gca().add_patch(rect)
    plt.text(x+offset, y+offset, type, ha="center", va="center", color='k', fontsize=24, fontweight='bold')

