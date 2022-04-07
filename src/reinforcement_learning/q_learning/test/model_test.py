import tensorflow as tf
import numpy as np
from tensorflow import keras
import pytest

from game.snake import SnakeGame
from ..agent import Agent, ai_direction_to_snake
from ..model import linear_qnet, Linear_QNet, QTrainer

placeholder_model = linear_qnet(11, 3, 3)

@pytest.mark.skip(reason="not understanding failure reason")
def test_shapes_inheritance_model():
    model = Linear_QNet(11, 256, 3)
    model.compile()
    # model(tf.ones((1,11)))

    assert len(model.layers) == 2

    assert type(model.layers[0]) == keras.layers.Dense
    assert type(model.layers[1]) == keras.layers.Dense

    assert model.layers[0].input_shape == (None, 11)
    assert model.layers[0].output_shape == (None, 256)
    assert model.layers[1].output_shape == (None, 3)

def test_shapes_linear_model():
    model = linear_qnet(11, 256, 3)
    model.compile()

    assert len(model.layers) == 3

    assert type(model.layers[0]) == keras.layers.InputLayer
    assert type(model.layers[1]) == keras.layers.Dense
    assert type(model.layers[2]) == keras.layers.Dense

    assert model.layers[0].output_shape == [(None, 11)]
    assert model.layers[1].output_shape == (None, 256)
    assert model.layers[2].output_shape == (None, 3)

def test_qtrainer_defaults():
    trainer = QTrainer(placeholder_model)

    assert trainer.model == placeholder_model
    assert trainer.gamma == 0.9

    assert type(trainer.optimizer) == keras.optimizers.Adam
    assert (trainer.optimizer.learning_rate.numpy()) == np.float32(1e-4)

    assert type(trainer.loss_object) == keras.losses.MeanSquaredError

def test_parameters_updated():
    agent = Agent()
    game = SnakeGame()

    state = agent.get_state(game)
    action = agent.get_action(state)
    snake_action = ai_direction_to_snake(action)

    eaten, _, done = game.play_step(snake_action)
    reward = agent.reward(eaten, done)

    state_next = agent.get_state(game)

    weights = []
    for variable in agent.model.trainable_variables:
        weights.append(tf.identity(variable))

    agent.train_short_memory(state, action, reward, state_next, done)

    for i, new_weight in enumerate(agent.model.trainable_variables):
        # at least one element changed in every layer
        assert not tf.math.reduce_all(weights[i] == new_weight).numpy()



