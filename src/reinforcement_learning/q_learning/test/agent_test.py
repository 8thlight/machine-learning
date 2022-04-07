import pytest
import tensorflow as tf
import numpy as np

from game.snake import Direction, SnakeGame, Point
from ..agent import ai_direction_to_snake, Agent

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
    agent = Agent()
    with pytest.raises(ValueError) as e_info:
        agent.reward(False, 0.56)

    assert str(e_info.value) == "Reward function only receives booleans"

    with pytest.raises(ValueError) as e_info:
        agent.reward(0.42, True)

    assert str(e_info.value) == "Reward function only receives booleans"

def test_reward_on_done():
    agent = Agent()
    reward = agent.reward(False, True)
    assert reward == -10

def test_reward_on_eaten():
    agent = Agent()
    reward = agent.reward(True, False)
    assert reward == 10

def test_reward_on_default():
    agent = Agent()
    reward = agent.reward(False, False)
    assert reward == 0

def test_state_directions():
    agent = Agent()
    game = SnakeGame()
    indexes = [3, 4, 5, 6]
    directions = [Direction.LEFT, Direction.RIGHT, 
        Direction.UP, Direction.DOWN]

    for idx, direction in zip(indexes, directions):
        game.direction = direction
        state = agent.get_state(game)
        other_indexes = [i for i in indexes if i != idx]
        assert state[idx] == 1
        for other_idx in other_indexes:
            assert state[other_idx] == 0

def test_state_food_locations():
    agent = Agent()
    game = SnakeGame()
    unit = game.block_size
    x = game.head.x
    y = game.head.y

    # food is up
    game.food = Point(x, y - unit)
    state = agent.get_state(game)
    assert (state[7:11] == [0, 0, 1, 0]).all()

    # food is up-right
    game.food = Point(x + unit, y - unit)
    state = agent.get_state(game)
    assert (state[7:11] == [0, 1, 1, 0]).all()

    # food is right
    game.food = Point(x + unit, y)
    state = agent.get_state(game)
    assert (state[7:11] == [0, 1, 0, 0]).all()

    # food is down-right
    game.food = Point(x + unit, y + unit)
    state = agent.get_state(game)
    assert (state[7:11] == [0, 1, 0, 1]).all()

    # food is down
    game.food = Point(x, y + unit)
    state = agent.get_state(game)
    assert (state[7:11] == [0, 0, 0, 1]).all()

    # food is down-left
    game.food = Point(x - unit, y + unit)
    state = agent.get_state(game)
    assert (state[7:11] == [1, 0, 0, 1]).all()

    # food is left
    game.food = Point(x - unit, y)
    state = agent.get_state(game)
    assert (state[7:11] == [1, 0, 0, 0]).all()

    # food is up-left
    game.food = Point(x - unit, y - unit)
    state = agent.get_state(game)
    assert (state[7:11] == [1, 0, 1, 0]).all()

def test_state_danger_straight():
    agent = Agent()
    game = SnakeGame()
    unit = game.block_size
    x = game.head.x
    y = game.head.y

    directions = [Direction.UP, Direction.RIGHT,
        Direction.DOWN, Direction.LEFT]
    tails = [Point(x, y-unit), Point(x+unit, y),
        Point(x, y+unit), Point(x-unit, y)]

    for direction, tail in zip(directions, tails):
        game.direction = direction
        game.snake = [game.head, tail]
        state = agent.get_state(game)
        assert (state[:3] == [1, 0, 0]).all()

def test_state_danger_left():
    agent = Agent()
    game = SnakeGame()
    unit = game.block_size
    x = game.head.x
    y = game.head.y

    directions = [Direction.UP, Direction.RIGHT,
        Direction.DOWN, Direction.LEFT]
    tails = [Point(x-unit, y), Point(x, y-unit), 
        Point(x+unit, y), Point(x, y+unit)]

    for direction, tail in zip(directions, tails):
        game.direction = direction
        game.snake = [game.head, tail]
        state = agent.get_state(game)
        assert (state[:3] == [0, 0, 1]).all()

def test_state_danger_right():
    agent = Agent()
    game = SnakeGame()
    unit = game.block_size
    x = game.head.x
    y = game.head.y

    directions = [Direction.UP, Direction.RIGHT,
        Direction.DOWN, Direction.LEFT]
    tails = [Point(x+unit, y), Point(x,y+unit),
        Point(x-unit, y), Point(x, y-unit)]

    for direction, tail in zip(directions, tails):
        game.direction = direction
        game.snake = [game.head, tail]
        state = agent.get_state(game)
        assert (state[:3] == [0, 1, 0]).all()

def test_get_state_shape():
    agent = Agent()
    game = SnakeGame()
    state = agent.get_state(game)

    assert type(state) == np.ndarray
    assert state.shape == (11,)

def test_get_action_shape():
    agent = Agent()
    game = SnakeGame()
    state = agent.get_state(game)
    action = agent.get_action(state)

    assert type(action) == list
    assert len(action) == 3
    assert sum(action) == 1
    for elem in action:
        assert type(elem) == int

def test_defaults():
    agent = Agent()

    assert agent.n_games == 0
    assert agent.epsilon == 0
    assert agent.gamma == 0.9
    assert agent.batch_size == 1000
