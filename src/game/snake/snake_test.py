import pytest

from .snake import SnakeGame, Direction, Point


def test_direction_left():
    game = SnakeGame()

    starts = [Direction.UP, Direction.RIGHT, Direction.DOWN,
              Direction.LEFT]
    expects = [Direction.LEFT, Direction.UP, Direction.RIGHT,
               Direction.DOWN]

    for start, expect in zip(starts, expects):
        game.direction = start
        assert game.direction == start
        chosen_direction = game._choose_direction("left")
        assert chosen_direction == expect


def test_direction_right():
    game = SnakeGame()

    starts = [Direction.UP, Direction.RIGHT, Direction.DOWN,
              Direction.LEFT]
    expects = [Direction.RIGHT, Direction.DOWN, Direction.LEFT,
               Direction.UP]

    for start, expect in zip(starts, expects):
        game.direction = start
        assert game.direction == start
        chosen_direction = game._choose_direction("right")
        assert chosen_direction == expect


def test_direction_forward():
    game = SnakeGame()

    directions = [Direction.UP, Direction.RIGHT,
                  Direction.DOWN, Direction.LEFT]

    for direction in directions:
        game.direction = direction
        assert game.direction == direction
        chosen_direction = game._choose_direction("forward")
        assert chosen_direction == direction


def test_direction_correct_value():
    game = SnakeGame()

    with pytest.raises(ValueError) as e_info:
        game._choose_direction("foobar")

    assert str(e_info.value) == "Unknown action: foobar"
