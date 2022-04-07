
# Taken from https://github.com/python-engineer/python-fun/blob/master/snake-pygame/snake_game.py
import pygame
import random
from enum import Enum
from collections import namedtuple

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN = (0, 200, 50)
BLACK = (0,0,0)

class SnakeGame:
    def __init__(self, w=640, h=480):
        self.block_size = 20
        self.w = w
        self.h = h
        # init display
        self.display = None
        
        # uhn  vars] cxa score   
        
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-self.block_size, self.head.y),
                      Point(self.head.x-(2*self.block_size), self.head.y)]
        
        self.score = 0
        self.food = None
        # self.steps_played = 0
        self._place_food()
        
    def _place_food(self):
        while True:
            x = random.randint(0, (self.w-self.block_size )//self.block_size )*self.block_size 
            y = random.randint(0, (self.h-self.block_size )//self.block_size )*self.block_size
            self.food = Point(x, y)
            if self.food not in self.snake:
                break
    
    def _choose_direction(self, action):
        clockwise = [Direction.UP, Direction.RIGHT, 
            Direction.DOWN, Direction.LEFT]
        current = clockwise.index(self.direction)
        
        if action == "left":
            return clockwise[(current - 1) % 4]
        if action == "right":
            return clockwise[(current + 1) % 4]
        if action == "forward":
            return self.direction
        else:
            raise ValueError("Unknown action: " + str(action))

    # actions:
    # "left", "right", "forward"
    def play_step(self, action):
        # self.steps_played += 1
        self.direction = self._choose_direction(action)

        # move
        self._move(self.direction) # update the head
        self.snake.insert(0, self.head) # update the snake

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
        # hits boundary
        if point.x > self.w - self.block_size or point.x < 0 or point.y > self.h - self.block_size or point.y < 0:
            return True
        # hits itself
        if point in self.snake[1:]:
            return True
        
        return False
        
    def pygame_draw(self):
        if self.display == None:
            self.font = pygame.font.Font(pygame.font.get_default_font(), 25)

            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')

        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
         
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size))
        
        text = self.font.render("Score: " + str(self.score), True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += self.block_size
        elif direction == Direction.LEFT:
            x -= self.block_size
        elif direction == Direction.DOWN:
            y += self.block_size
        elif direction == Direction.UP:
            y -= self.block_size
            
        self.head = Point(x, y)

    def quit(self):
        if self.display:
            pygame.quit()
            
# player_direction: ['up', 'left', 'right', 'down']
def player_to_snake_perspective(snake_direction, player_direction):
    if snake_direction == Direction.UP:
        return {
            'up': 'forward',
            'left': 'left',
            'down': 'forward', # <- no tail crash
            'right': 'right'
        }[player_direction]

    if snake_direction == Direction.LEFT:
        return {
            'up': 'right',
            'left': 'forward',
            'down': 'left',
            'right': 'forward' # <- no tail crash
        }[player_direction]

    if snake_direction == Direction.DOWN:
        return {
            'up': 'forward', # <- no tail crash
            'left': 'right',
            'down': 'forward',
            'right': 'left'
        }[player_direction]

    # Direction.RIGHT
    return {
        'up': 'left',
        'left': 'forward', # <- no tail crash
        'down': 'right',
        'right': 'forward'
    }[player_direction]

if __name__ == '__main__':
    pygame.init()

    game = SnakeGame()
    
    speed = 20
    clock = pygame.time.Clock()
    stop = False
    # game loop
    while True:
        direction = game.direction

        # 1. collect user input
        action = "forward"
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.quit()
                stop = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    game.quit()
                    stop = True
                
                if event.key in [pygame.K_LEFT, pygame.K_RIGHT,
                    pygame.K_UP, pygame.K_DOWN]: # any other key keeps forward

                    player_direction = {
                        pygame.K_LEFT: "left",
                        pygame.K_RIGHT: "right",
                        pygame.K_UP: "up",
                        pygame.K_DOWN: "down"
                    }[event.key]

                    action = player_to_snake_perspective(game.direction, 
                        player_direction)
        if stop:
            break
        
        eaten, score, game_over = game.play_step(action)
        game.pygame_draw()
        clock.tick(speed)
        
        
        if game_over == True:
            break
        
    print('Final Score', score)
        
        
    pygame.quit()