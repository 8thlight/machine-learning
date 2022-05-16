"""Tests the model.py file"""
# pylint: disable=missing-function-docstring
import tensorflow as tf
import numpy as np
from tensorflow import keras

from game.snake import SnakeGame
from ..agent import SnakeAgent, ai_direction_to_snake, snake_state_11, snake_reward
from ..model import QTrainer, linear_qnet

placeholder_model = linear_qnet(11, 3, 3)


def test_shapes_linear_model():
    model = linear_qnet(11, 256, 3)
    model.compile()

    assert len(model.layers) == 3

    assert isinstance(model.layers[0], keras.layers.InputLayer)
    assert isinstance(model.layers[1], keras.layers.Dense)
    assert isinstance(model.layers[2], keras.layers.Dense)

    assert model.layers[0].output_shape == [(None, 11)]
    assert model.layers[1].output_shape == (None, 256)
    assert model.layers[2].output_shape == (None, 3)


def test_qtrainer_defaults():
    trainer = QTrainer(placeholder_model)

    assert trainer.model == placeholder_model
    assert trainer.gamma == 0.9

    assert isinstance(trainer.optimizer, keras.optimizers.Adam)
    assert (trainer.optimizer.learning_rate.numpy()) == np.float32(1e-4)

    assert isinstance(trainer.loss_object, keras.losses.MeanSquaredError)


def test_parameters_updated():
    agent = SnakeAgent(QTrainer(linear_qnet(11, 256, 3)))
    game = SnakeGame()

    state = snake_state_11(game)
    action = agent.get_action(state)
    snake_action = ai_direction_to_snake(action)

    eaten, _, done = game.play_step(snake_action)
    reward = snake_reward(eaten, done)

    state_next = snake_state_11(game)

    weights = []
    for variable in agent.trainer.model.trainable_variables:
        weights.append(tf.identity(variable))

    agent.train_short_memory(state, action, reward, state_next, done)

    for i, new_weight in enumerate(agent.trainer.model.trainable_variables):
        # at least one element changed in every layer
        assert not tf.math.reduce_all(weights[i] == new_weight).numpy()
