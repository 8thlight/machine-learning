"""Tests the .snake file"""
# pylint: disable=missing-function-docstring
from .snake import SnakeGame, Direction, Point, player_to_snake_perspective


def test_initial_direction_is_right():
    game = SnakeGame()
    assert game.direction == Direction.RIGHT


def test_snake_is_size_3():
    game = SnakeGame()
    assert len(game.snake) == 3


def test_direction_right():
    game = SnakeGame()
    game.food = Point(0, 0)

    eaten, score, game_over = game.play_step("right")
    assert game.direction == Direction.DOWN
    assert eaten is False
    assert score == 0
    assert game_over is False


def test_direction_forward():
    game = SnakeGame()
    game.food = Point(0, 0)

    eaten, score, game_over = game.play_step("forward")
    assert game.direction == Direction.RIGHT
    assert eaten is False
    assert score == 0
    assert game_over is False


def test_direction_left():
    game = SnakeGame()
    game.food = Point(0, 0)

    eaten, score, game_over = game.play_step("left")
    assert game.direction == Direction.UP
    assert eaten is False
    assert score == 0
    assert game_over is False


def player_to_snake_perspective_when_facing_up():
    orders = ["up", "down", "left", "right"]
    expected_actions = ["forward", "forward", "left", "right"]

    for player_order, expected_action in zip(orders, expected_actions):
        snake_action = player_to_snake_perspective(Direction.UP, player_order)
        assert snake_action == expected_action


def player_to_snake_perspective_when_facing_right():
    orders = ["up", "down", "left", "right"]
    expected_actions = ["left", "right", "forward", "forward"]

    for player_order, expected_action in zip(orders, expected_actions):
        snake_action = player_to_snake_perspective(
            Direction.RIGHT, player_order)
        assert snake_action == expected_action


def player_to_snake_perspective_when_facing_left():
    orders = ["up", "down", "left", "right"]
    expected_actions = ["right", "left", "forward", "forward"]

    for player_order, expected_action in zip(orders, expected_actions):
        snake_action = player_to_snake_perspective(
            Direction.LEFT, player_order)
        assert snake_action == expected_action


def player_to_snake_perspective_when_facing_down():
    orders = ["up", "down", "left", "right"]
    expected_actions = ["forward", "forward", "right", "left"]

    for player_order, expected_action in zip(orders, expected_actions):
        snake_action = player_to_snake_perspective(
            Direction.DOWN, player_order)
        assert snake_action == expected_action


def food_is_eaten():
    game = SnakeGame()
    game.food = Point(game.head.coordx + game.block_size, game.head.coordy)
    eaten, score, game_over = game.play_step("forward")

    assert eaten is True
    assert score == 1
    assert game_over is False


def can_collide_with_body():
    game = SnakeGame()
    game.snake.append(
        Point(game.head.coordx + game.block_size, game.head.coordy)
    )
    eaten, score, game_over = game.play_step("forward")

    assert eaten is False
    assert score == 0
    assert game_over is True


def can_collide_with_border():
    game = SnakeGame()
    new_head = Point(game.width, game.height / 2)
    assert game.is_collision(new_head) is False

    eaten, score, game_over = game.play_step("forward")

    assert eaten is False
    assert score == 0
    assert game_over is True


def border_collisions():
    game = SnakeGame()
    assert game.is_collision(Point(0, game.height - 2)) is False
    assert game.is_collision(Point(-1, game.height - 2)) is True

    assert game.is_collision(Point(game.width / 2, 0)) is False
    assert game.is_collision(Point(game.width / 2, -1)) is True

    assert game.is_collision(Point(game.width, game.height - 2)) is False
    assert game.is_collision(Point(game.width + 1, game.height - 2)) is True

    assert game.is_collision(Point(game.width / 2, game.height)) is False
    assert game.is_collision(Point(game.width / 2, game.height + 1)) is True
