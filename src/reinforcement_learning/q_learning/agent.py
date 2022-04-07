# import torch
from multiprocessing.sharedctypes import Value
import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
import os

import pygame

from collections import deque
from game.snake import SnakeGame, Direction, Point

from .model import Linear_QNet, QTrainer, linear_qnet
from plotter import gamescore_plotter

class Agent:
    def __init__(self, max_memory=100_000, lr=1e-4, batch_size=1000):
        self.n_games = 0
        self.epsilon = 0 # control the randomness
        self.gamma = 0.9 # discount rate (less than 1, around 0.8-0.9)
        self.memory = deque(maxlen=max_memory) # popleft() if it is above memory
        
        self.model = linear_qnet(11, 256, 3)
        self.trainer = QTrainer(self.model, learning_rate=lr, gamma=self.gamma)
        self.batch_size = batch_size

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - game.block_size, head.y)
        point_r = Point(head.x + game.block_size, head.y)
        point_u = Point(head.x, head.y - game.block_size)
        point_d = Point(head.x, head.y + game.block_size)
        
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
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]
        return np.array(state, dtype=int) # zeroes or ones

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample) # TODO check this one

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: exploration/exploitation tradeoff
        self.epsilon = 80 - self.n_games # TODO: play around with this
        final_move = [0, 0, 0] # left, right, forward
        if random.randint(0, 200) < self.epsilon: # explore
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = tf.constant([state], dtype=tf.float32) # create a tensor
            prediction = self.model.predict(state0)
            move = tf.argmax(prediction[0]).numpy() # Get index of maximum value
            final_move[move] = 1
        
        return final_move


    def reward(self, eaten:bool, done:bool) -> int:
        if type(eaten) != bool or type(done) != bool:
            raise ValueError("Reward function only receives booleans")
        
        if eaten:
            return 10

        if done:
            return -10

        return 0

    def save_model(self, file_name='model.pth'): # .pth?
        model_dir_path = './model'
        if not os.path.exists(model_dir_path):
            os.mkdir(model_dir_path)

        file_name = os.path.join(model_dir_path, file_name)
        self.model.save(file_name)

def ai_direction_to_snake(action:list):
    if type(action) != list:
        raise ValueError("Action must be a list")

    if action == [1, 0, 0]:
        return "left"
    
    if action == [0, 1, 0]:
        return "right"

    if action == [0, 0, 1]:
        return "forward"

    raise ValueError("Unknown action: " + str(action))

def train():
    pygame.init()

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    high_score = 0
    agent = Agent(lr=1e-3)
    game = SnakeGame(w=200, h=160)

    speed = 100
    clock = pygame.time.Clock()

    game_frames = 0 
    
    while True:
        # get old state
        state = agent.get_state(game)

        # get move
        action = agent.get_action(state)
        # [0, 0, 0] -> left, right, forward

        # perform move and get new state
        eaten, score, done = game.play_step(ai_direction_to_snake(action))
        
        # show AI training in real-time
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                pygame.quit()
                return
        
        # drawing requires to consume events as in the previous block
        game.pygame_draw() # draw the game
        clock.tick(speed)

        reward = agent.reward(eaten, done)

        # at this point, we have:
        # score, done, reward

        game_frames += 1
        if game_frames > 30*len(game.snake):
            eaten = False
            done = True
            reward = -10
            print("STOPPING DUE TO INFINITE LOOP")

        state_next = agent.get_state(game)
        # train short memory
        agent.train_short_memory(state, action, reward, state_next, done)

        # remember
        agent.remember(state, action, reward, state_next, done)

        if done:
            # train long memory (replay memory, or experience replay)

            if score > high_score:
                high_score = score
                agent.save_model('best_snake.pth')
            
            if high_score > 5:
                game = SnakeGame()
            elif high_score > 3:
                game = SnakeGame(w=400, h=300)
            elif high_score > 1:
                game = SnakeGame(w=320, h=240)
            else:
                game = SnakeGame(w=200, h=160)
            
            game_frames = 0
            agent.n_games += 1
            agent.train_long_memory()

            print('Game', agent.n_games, 'Score', score, 'Record:', high_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            # plot the results
            gamescore_plotter(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()