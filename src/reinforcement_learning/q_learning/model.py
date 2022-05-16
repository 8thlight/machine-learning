"""Holds the NN model and its parameter update functionality"""
# pylint: disable=too-many-arguments,too-few-public-methods
import tensorflow as tf
from tensorflow import keras


def linear_qnet(input_size: int, hidden_size: int,
                output_size: int) -> keras.Model:
    """Creates a 1-hidden-layer dense neural network"""
    inputs = keras.layers.Input(shape=input_size, name="Input")

    layer1 = keras.layers.Dense(
        hidden_size,
        activation="relu",
        name="Dense_1")(inputs)
    action = keras.layers.Dense(output_size, name="Dense_2")(layer1)

    return keras.Model(inputs=inputs, outputs=action)


class QTrainer():
    """Trains the given model according to an optimizer and loss function"""

    def __init__(self, model, learning_rate=1e-4, gamma=0.9):
        """Uses the ADAM optimizer and MeanSquaredError loss function"""
        self.model = model
        self.gamma = gamma

        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_object = keras.losses.MeanSquaredError()


    def train_step(self, states, actions, rewards, next_states, dones):
        """
        Updates the model's parameters by calculating their derivatives
        with respect to the loss function
        """
        future_rewards = self.model.predict(next_states)
        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards + self.gamma * tf.reduce_max(
            future_rewards, axis=1
        )

        updated_q_values = updated_q_values * (1 - dones)

        masks = actions

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            # train the model on the states and updated Q-values
            q_values = self.model(states)  # similar to action_probs

            # apply the masks to the Q-values to get the Q-value for the action
            # taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # calculate loss between new Q-value and old Q-value
            loss = self.loss_object(updated_q_values, q_action)

            # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
