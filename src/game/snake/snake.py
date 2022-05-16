"""SnakeGame. It's interface is suitable for humans an AI programs'"""
# pylint: disable=too-many-instance-attributes

# Taken from
# https://github.com/python-engineer/python-fun/blob/master/snake-pygame/snake_game.py
import random
from enum import Enum
from collections import namedtuple
import pygame


class Direction(Enum):
    """Direction enum used to know where the snake is heading towards"""
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'coordx, coordy')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 200, 50)
BLACK = (0, 0, 0)


class SnakeGame:
    """Simple 2D snake game"""

    def __init__(self, width=640, height=480):
        """Initialize the graphic blocks and a snake of size 3 heading right"""
        self.block_size = 20
        self.width = width
        self.height = height

        self.display = None
        self.font = None

        self.direction = Direction.RIGHT

        self.head = Point(self.width / 2, self.height / 2)
        self.snake = [self.head,
                      Point(self.head.coordx - self.block_size,
                            self.head.coordy),
                      Point(self.head.coordx - (2 * self.block_size),
                            self.head.coordy)]

        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        """Place the food randomly in a non-colliding coordinate"""
        while True:
            coordx = random.randint(0, (self.width - self.block_size) //
                                    self.block_size) * self.block_size
            coordy = random.randint(0, (self.height - self.block_size) //
                                    self.block_size) * self.block_size
            self.food = Point(coordx, coordy)
            if self.food not in self.snake:
                break

    def _choose_direction(self, action):
        """Rotate the snake given the provided `action` string"""
        clockwise = [Direction.UP, Direction.RIGHT,
                     Direction.DOWN, Direction.LEFT]
        current = clockwise.index(self.direction)

        if action == "left":
            return clockwise[(current - 1) % 4]
        if action == "right":
            return clockwise[(current + 1) % 4]
        if action == "forward":
            return self.direction

        raise ValueError("Unknown action: " + str(action))

    def play_step(self, action):
        """Receive an action and execute its effects of moving and colliding"""
        if action not in ["left", "right", "forward"]:
            raise ValueError("Unknown action: " + str(action))

        self.direction = self._choose_direction(action)

        # move
        self._move(self.direction)  # update the head
        self.snake.insert(0, self.head)  # update the snake

        # check if game over
        eaten = False
        game_over = False
        if self.is_collision(self.head):
            game_over = True
            return eaten, self.score, game_over

        # place new food or just move
        if self.head == self.food:
            self.score += 1
            eaten = True
            self._place_food()
        else:
            self.snake.pop()

        # return game over and score
        return eaten, self.score, game_over

    def is_collision(self, point):
        """Returns true if the current state is a collision"""
        # hits boundary
        if point.coordx > self.width - self.block_size or \
                point.coordy > self.height - self.block_size or \
                point.coordx < 0 or point.coordy < 0:
            return True
        # hits itself
        if point in self.snake[1:]:
            return True

        return False

    def pygame_draw(self):
        """Uses pygame to draw the game in the screen"""
        if self.display is None:
            self.font = pygame.font.Font(pygame.font.get_default_font(), 25)

            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake')

        self.display.fill(BLACK)

        for body_point in self.snake:
            pygame.draw.rect(
                self.display,
                GREEN,
                pygame.Rect(
                    body_point.coordx,
                    body_point.coordy,
                    self.block_size,
                    self.block_size))

        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(
                self.food.coordx,
                self.food.coordy,
                self.block_size,
                self.block_size))

        text = self.font.render("Score: " +
                                str(self.score), True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, direction):
        """Adds a new head according to the given direction"""
        coordx = self.head.coordx
        coordy = self.head.coordy
        if direction == Direction.RIGHT:
            coordx += self.block_size
        elif direction == Direction.LEFT:
            coordx -= self.block_size
        elif direction == Direction.DOWN:
            coordy += self.block_size
        elif direction == Direction.UP:
            coordy -= self.block_size

        self.head = Point(coordx, coordy)

    def quit(self):
        """Quit display"""
        if self.display:
            pygame.quit()

# player_direction: ['up', 'left', 'right', 'down']


def player_to_snake_perspective(snake_direction, player_direction):
    """Transforms universal directions (player) to local directions (snake)"""
    if snake_direction == Direction.UP:
        return {
            'up': 'forward',
            'left': 'left',
            'down': 'forward',  # <- no tail crash
            'right': 'right'
        }[player_direction]

    if snake_direction == Direction.LEFT:
        return {
            'up': 'right',
            'left': 'forward',
            'down': 'left',
            'right': 'forward'  # <- no tail crash
        }[player_direction]

    if snake_direction == Direction.DOWN:
        return {
            'up': 'forward',  # <- no tail crash
            'left': 'right',
            'down': 'forward',
            'right': 'left'
        }[player_direction]

    # Direction.RIGHT
    return {
        'up': 'left',
        'left': 'forward',  # <- no tail crash
        'down': 'right',
        'right': 'forward'
    }[player_direction]
