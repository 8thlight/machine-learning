"""Agent that handles the training timing and the rewards"""
# pylint: disable=too-many-arguments
import random
import os
from collections import deque
import numpy as np
import tensorflow as tf

from game.snake import Direction, Point

def as_tensors(states, actions, rewards, next_states, dones):
    """Converts arrays to tensors"""
    state = tf.constant(states, dtype=tf.float32)
    action = tf.constant(actions, dtype=tf.float32)
    reward = tf.constant(rewards, dtype=tf.float32)
    next_state = tf.constant(next_states, dtype=tf.float32)
    done = tf.constant(dones, dtype=tf.float32)

    return state, action, reward, next_state, done

def add_dimension(states, actions, rewards, next_states, dones):
    """Expands with a dimension. To use when wanting (1,x) but having (1,)"""
    state = tf.expand_dims(states, axis=0)
    action = tf.expand_dims(actions, axis=0)
    reward = tf.expand_dims(rewards, axis=0)
    next_state = tf.expand_dims(next_states, axis=0)
    done = tf.expand_dims(dones, axis=0)

    return state, action, reward, next_state, done

def snake_state_11(game):
    """
    Gets the 11 state vector state of the snake game.
    It uses 3 places for danger, 4 for direction and 4 for food location
    """
    head = game.snake[0]
    point_l = Point(head.coordx - game.block_size, head.coordy)
    point_r = Point(head.coordx + game.block_size, head.coordy)
    point_u = Point(head.coordx, head.coordy - game.block_size)
    point_d = Point(head.coordx, head.coordy + game.block_size)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        # Danger straight
        (dir_r and game.is_collision(point_r)) or
        (dir_l and game.is_collision(point_l)) or
        (dir_u and game.is_collision(point_u)) or
        (dir_d and game.is_collision(point_d)),

        # Danger right
        (dir_u and game.is_collision(point_r)) or
        (dir_d and game.is_collision(point_l)) or
        (dir_l and game.is_collision(point_u)) or
        (dir_r and game.is_collision(point_d)),

        # Danger left
        (dir_d and game.is_collision(point_r)) or
        (dir_u and game.is_collision(point_l)) or
        (dir_r and game.is_collision(point_u)) or
        (dir_l and game.is_collision(point_d)),

        # Move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,

        # Food location
        game.food.coordx < game.head.coordx,  # food left
        game.food.coordx > game.head.coordx,  # food right
        game.food.coordy < game.head.coordy,  # food up
        game.food.coordy > game.head.coordy  # food down
    ]
    return np.array(state, dtype=int)  # zeroes or ones


def snake_reward(eaten: bool, done: bool) -> int:
    """Gets reward based on the eaten and done parameters"""
    if not isinstance(eaten, bool) or not isinstance(done, bool):
        raise ValueError("Reward function only receives booleans")

    if eaten:
        return 10

    if done:
        return -10

    return 0


class SnakeAgent:
    """Has the data of all past states and trains the model accordingly"""

    def __init__(
            self,
            trainer,
            max_memory=100_000,
            batch_size=1000,
            epsilon=0):
        """Initializes a deque memory so that we don't have to"""
        self.n_games = 0
        self.epsilon = epsilon  # control the randomness
        self.memory = deque(maxlen=max_memory)

        self.trainer = trainer
        self.batch_size = batch_size

    def remember(self, state, action, reward, next_state, done):
        """
        Appens the variables to memory and pops left if maximum memory
        is reached.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Gets a random sample of the data of states to replay in training"""
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(
                self.memory, self.batch_size)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        states, actions, rewards, next_states, dones = as_tensors(
            states, actions, rewards, next_states, dones
        )

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Trains using the given single state"""
        state, action, reward, next_state, done = as_tensors(
            state, action, reward, next_state, done
        )
        state, action, reward, next_state, done = add_dimension(
            state, action, reward, next_state, done)
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """Gets an action using an exploration/eploitation tradeoff"""
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]  # left, right, forward
        if random.randint(0, 200) < self.epsilon:  # explore
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = tf.constant([state], dtype=tf.float32)  # create a tensor
            prediction = self.trainer.model.predict(state0)
            # Get index of maximum value
            move = tf.argmax(prediction[0]).numpy()
            final_move[move] = 1

        return final_move

    def save_model(self, model_dir_path="./model", file_name='model.pth'):
        """Saves the model as a .pth folder"""
        if not os.path.exists(model_dir_path):
            os.mkdir(model_dir_path)

        file_name = os.path.join(model_dir_path, file_name)
        self.trainer.model.save(file_name)


def ai_direction_to_snake(action: list):
    """Transforms a vector of actions to the snake game's perspective string"""
    if not isinstance(action, list):
        raise ValueError("Action must be a list")

    if action == [1, 0, 0]:
        return "left"

    if action == [0, 1, 0]:
        return "right"

    if action == [0, 0, 1]:
        return "forward"

    raise ValueError("Unknown action: " + str(action))
