"""CLI to train/see QLearning in action solving the snake game"""
import argparse
import numpy as np
import pygame

from reinforcement_learning.q_learning import (
    SnakeAgent, ai_direction_to_snake,
    QTrainer, linear_qnet, snake_state_11, snake_reward
)
from game.snake import SnakeGame
from plotter import gamescore_plotter


def buil_arg_parser():
    """Parses the user's arguments"""
    parser = argparse.ArgumentParser(
        description="Use Deep-QLearning in the snake game",
        epilog="Built with <3 by Emmanuel Byrd at 8th Light Ltd.")
    parser.add_argument(
        "--best-models-dir",
        metavar="./model", default="./model", type=str,
        help="Folder to store the increasingly best models"
    )
    parser.add_argument(
        "--score-history",
        metavar="./score_history", default="./score_history", type=str,
        help="Where to store the score history"
    )
    parser.add_argument(
        "--checkpoint-path", metavar="./model/snake_5.pth", type=str,
        help="Path of pre-trained model to start from"
    )
    parser.add_argument(
        "--fps", metavar="100", type=int, default=100,
        help="Frames per second"
    )
    parser.add_argument(
        "--learning-rate", metavar="1e-3", type=float, default=1e-3,
        help="QTrainer learning rate"
    )
    parser.add_argument(
        "--gamma", metavar="0.9", type=float, default=0.9,
        help="QTrainer gamma value"
    )
    parser.add_argument(
        "--hidden-layer-size", metavar="256", type=int, default=256,
        help="Size of the hidden layer"
    )
    parser.add_argument(
        "--max-width", metavar="400", type=int, default=400,
        help="Maximum board width"
    )
    parser.add_argument(
        "--max-height", metavar="320", type=int, default=320,
        help="Maximum board height"
    )
    return parser


def train(args):
    """Execute AI training/game loop"""
    pygame.init()

    score_tracker = ScoreTracker()

    high_score = 0

    agent = SnakeAgent(
        QTrainer(generate_model(args),
                 learning_rate=args.learning_rate,
                 gamma=args.gamma)
    )
    game = SnakeGame(width=200, height=160)

    clock = pygame.time.Clock()

    game_frames = 0

    while True:
        # get old state
        state = snake_state_11(game)

        # get move
        action = agent.get_action(state)
        # [0, 0, 0] -> left, right, forward

        # perform move and get new state
        eaten, score, done = game.play_step(ai_direction_to_snake(action))

        # show AI training in real-time
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                pygame.quit()
                return

        # drawing requires to consume events e.g. pygame.event.get()
        game.pygame_draw()  # draw the game
        clock.tick(args.fps)

        reward = snake_reward(eaten, done)

        game_frames += 1
        if game_frames > 30 * len(game.snake):
            eaten = False
            done = True
            reward = -10
            print("Stopping due to infinite loop strategy")

        state_next = snake_state_11(game)
        # train short memory
        agent.train_short_memory(state, action, reward, state_next, done)

        # remember
        agent.remember(state, action, reward, state_next, done)

        if done:
            if score > high_score:
                high_score = score
                agent.save_model(args.best_models_dir,
                                 f'snake_{high_score}.pth')

            game = scaling_board(high_score,
                                    args.max_width, args.max_height)

            game_frames = 0
            agent.n_games += 1

            # train long memory (replay memory, or experience replay)
            agent.train_long_memory()

            print('Game', agent.n_games, 'Score', score, 'Record:', high_score)

            # show the results
            score_tracker.add_new_score(score)
            score_tracker.show_hist()
            np.save(args.score_history, np.array(score_tracker.get_hist()))


def generate_model(args):
    """Generate a linear neural network of input 11 and output 3"""
    model = linear_qnet(11, args.hidden_layer_size, 3)
    if args.checkpoint_path:
        model.load_weights(args.checkpoint_path)

    return model


def scaling_board(high_score, max_width, max_height):
    """Choose the appropriate size for the next game depending on the score"""
    if high_score > 5:
        return SnakeGame()

    if high_score > 3:
        return SnakeGame(width=max_width, height=max_height)

    if high_score > 1:
        return SnakeGame(width=320, height=240)

    return SnakeGame(width=200, height=160)


class ScoreTracker:
    """State class that keeps updated information on the score"""

    def __init__(self):
        """Initialize analysis variables"""
        self.plot_scores = []
        self.plot_mean_scores = []
        self.total_score = 0

    def add_new_score(self, score):
        """Adds the given score and calculates the average so far"""
        self.plot_scores.append(score)
        self.total_score += score
        self.plot_mean_scores.append(self.total_score / len(self.plot_scores))

    def show_hist(self):
        """Plot all the stored information"""
        gamescore_plotter(self.plot_scores, self.plot_mean_scores)

    def get_hist(self):
        """Returns a list with the scores and mean scores"""
        return [self.plot_scores, self.plot_mean_scores]


def main():
    """Main function"""
    arg_parser = buil_arg_parser()
    args = arg_parser.parse_args()

    train(args)

    print("Finished.")


if __name__ == "__main__":
    main()
