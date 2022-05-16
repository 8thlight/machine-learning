"""Tests the agent.py file"""
# pylint: disable=missing-function-docstring
import pytest
import tensorflow as tf
import numpy as np

from game.snake import Direction, SnakeGame, Point
from ..agent import SnakeAgent, ai_direction_to_snake, snake_state_11, snake_reward
from ..model import QTrainer, linear_qnet

placeholder_trainer = QTrainer(linear_qnet(11, 5, 3))


def test_ai_direction_left():
    snake_direction = ai_direction_to_snake([1, 0, 0])
    assert snake_direction == "left"


def test_ai_direction_right():
    snake_direction = ai_direction_to_snake([0, 1, 0])
    assert snake_direction == "right"


def test_ai_direction_forward():
    snake_direction = ai_direction_to_snake([0, 0, 1])
    assert snake_direction == "forward"


def test_ai_direction_correct_parameter():
    with pytest.raises(ValueError) as e_info:
        ai_direction_to_snake([1, 1, 1])

    assert str(e_info.value) == "Unknown action: [1, 1, 1]"


def test_ai_direction_cannot_receive_tensor():
    with pytest.raises(ValueError) as e_info:
        ai_direction_to_snake(tf.constant([1, 0, 0]))

    assert str(e_info.value) == "Action must be a list"


def test_ai_direction_cannot_receive_numpy_array():
    with pytest.raises(ValueError) as e_info:
        ai_direction_to_snake(np.array([1, 0, 0]))

    assert str(e_info.value) == "Action must be a list"


def test_reward_correct_parameters():
    with pytest.raises(ValueError) as e_info:
        snake_reward(False, 0.56)

    assert str(e_info.value) == "Reward function only receives booleans"

    with pytest.raises(ValueError) as e_info:
        snake_reward(0.42, True)

    assert str(e_info.value) == "Reward function only receives booleans"


def test_reward_on_done():
    reward = snake_reward(False, True)
    assert reward == -10


def test_reward_on_eaten():
    reward = snake_reward(True, False)
    assert reward == 10


def test_reward_on_default():
    reward = snake_reward(False, False)
    assert reward == 0


def test_get_action_shape():
    agent = SnakeAgent(placeholder_trainer)
    game = SnakeGame()
    state = snake_state_11(game)
    action = agent.get_action(state)

    assert isinstance(action, list)
    assert len(action) == 3
    assert sum(action) == 1
    for elem in action:
        assert isinstance(elem, int)


def test_defaults():
    agent = SnakeAgent(placeholder_trainer)

    assert agent.n_games == 0
    assert agent.epsilon == 0
    assert agent.batch_size == 1000

def test_get_state_shape():
    game = SnakeGame()
    state = snake_state_11(game)

    assert isinstance(state, np.ndarray)
    assert state.shape == (11,)


def test_state_danger_right():
    game = SnakeGame()
    unit = game.block_size
    coordx = game.head.coordx
    coordy = game.head.coordy

    directions = [Direction.UP, Direction.RIGHT,
                  Direction.DOWN, Direction.LEFT]
    tails = [Point(coordx + unit, coordy), Point(coordx, coordy + unit),
             Point(coordx - unit, coordy), Point(coordx, coordy - unit)]

    for direction, tail in zip(directions, tails):
        game.direction = direction
        game.snake = [game.head, tail]
        state = snake_state_11(game)
        assert (state[:3] == [0, 1, 0]).all()


def test_state_danger_straight():
    game = SnakeGame()
    unit = game.block_size
    coordx = game.head.coordx
    coordy = game.head.coordy

    directions = [Direction.UP, Direction.RIGHT,
                  Direction.DOWN, Direction.LEFT]
    tails = [Point(coordx, coordy - unit), Point(coordx + unit, coordy),
             Point(coordx, coordy + unit), Point(coordx - unit, coordy)]

    for direction, tail in zip(directions, tails):
        game.direction = direction
        game.snake = [game.head, tail]
        state = snake_state_11(game)
        assert (state[:3] == [1, 0, 0]).all()


def test_state_danger_left():
    game = SnakeGame()
    unit = game.block_size
    coordx = game.head.coordx
    coordy = game.head.coordy

    directions = [Direction.UP, Direction.RIGHT,
                  Direction.DOWN, Direction.LEFT]
    tails = [Point(coordx - unit, coordy), Point(coordx, coordy - unit),
             Point(coordx + unit, coordy), Point(coordx, coordy + unit)]

    for direction, tail in zip(directions, tails):
        game.direction = direction
        game.snake = [game.head, tail]
        state = snake_state_11(game)
        assert (state[:3] == [0, 0, 1]).all()


def test_state_directions():
    game = SnakeGame()
    indexes = [3, 4, 5, 6]
    directions = [Direction.LEFT, Direction.RIGHT,
                  Direction.UP, Direction.DOWN]

    for idx, direction in zip(indexes, directions):
        game.direction = direction
        state = snake_state_11(game)
        other_indexes = [i for i in indexes if i != idx]
        assert state[idx] == 1
        for other_idx in other_indexes:
            assert state[other_idx] == 0


def test_state_food_locations():
    game = SnakeGame()
    unit = game.block_size
    coordx = game.head.coordx
    coordy = game.head.coordy

    # food is up
    game.food = Point(coordx, coordy - unit)
    state = snake_state_11(game)
    assert (state[7:11] == [0, 0, 1, 0]).all()

    # food is up-right
    game.food = Point(coordx + unit, coordy - unit)
    state = snake_state_11(game)
    assert (state[7:11] == [0, 1, 1, 0]).all()

    # food is right
    game.food = Point(coordx + unit, coordy)
    state = snake_state_11(game)
    assert (state[7:11] == [0, 1, 0, 0]).all()

    # food is down-right
    game.food = Point(coordx + unit, coordy + unit)
    state = snake_state_11(game)
    assert (state[7:11] == [0, 1, 0, 1]).all()

    # food is down
    game.food = Point(coordx, coordy + unit)
    state = snake_state_11(game)
    assert (state[7:11] == [0, 0, 0, 1]).all()

    # food is down-left
    game.food = Point(coordx - unit, coordy + unit)
    state = snake_state_11(game)
    assert (state[7:11] == [1, 0, 0, 1]).all()

    # food is left
    game.food = Point(coordx - unit, coordy)
    state = snake_state_11(game)
    assert (state[7:11] == [1, 0, 0, 0]).all()

    # food is up-left
    game.food = Point(coordx - unit, coordy - unit)
    state = snake_state_11(game)
    assert (state[7:11] == [1, 0, 1, 0]).all()
