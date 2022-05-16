"""CLI to play the snake game manually"""
import pygame

from game.snake import SnakeGame, player_to_snake_perspective


def play_snake():
    """Initialize and run the game loop"""
    pygame.init()

    game = SnakeGame()

    speed = 20
    clock = pygame.time.Clock()
    stop = False
    # game loop
    while True:

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

                if event.key in [
                        pygame.K_LEFT,
                        pygame.K_RIGHT,
                        pygame.K_UP,
                        pygame.K_DOWN]:  # any other key keeps forward

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

        _, score, game_over = game.play_step(action)
        game.pygame_draw()
        clock.tick(speed)

        if game_over:
            break

    print('Final Score', score)

    pygame.quit()


if __name__ == '__main__':
    play_snake()
