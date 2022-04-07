import tensorflow as tf
from tensorflow import keras
import numpy as np

def linear_qnet(input_size:int, hidden_size:int, output_size:int) -> keras.Model:
    inputs = keras.layers.Input(shape=input_size, name="Input")

    layer1 = keras.layers.Dense(hidden_size, activation="relu", name="Dense_1")(inputs)
    action = keras.layers.Dense(output_size, name="Dense_2")(layer1)

    return keras.Model(inputs=inputs, outputs=action)

class Linear_QNet(keras.Model):
    def __init__(self, input_size:int, hidden_size:int, output_size:int):
        super().__init__()
        self.linear1 = keras.layers.Dense(hidden_size, input_shape=(None, input_size), activation="relu")
        self.linear2 = keras.layers.Dense(output_size)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.linear1(inputs)
        x = self.linear2(x)
        return x

class QTrainer():
    def __init__(self, model, learning_rate=1e-4, gamma=0.9):
        self.model = model
        self.gamma = gamma

        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_object = keras.losses.MeanSquaredError()

    # @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        state = tf.constant(states, dtype=tf.float32)
        action = tf.constant(actions, dtype=tf.float32)
        reward = tf.constant(rewards, dtype=tf.float32)
        next_state = tf.constant(next_states, dtype=tf.float32)
        done = tf.constant(dones, dtype=tf.float32)
        
        if len(state.shape) == 1:
            # we want a shape (1, x), but have a single element
            state = tf.expand_dims(state, axis=0) # convert to shape (1,)
            action = tf.expand_dims(action, axis=0)
            reward = tf.expand_dims(reward, axis=0)
            next_state = tf.expand_dims(next_state, axis=0)
            done = tf.expand_dims(done, axis=0)

        future_rewards = self.model.predict(next_state)
        # Q value = reward + discount factor * expected future reward
        updated_q_values = reward + self.gamma * tf.reduce_max(
            future_rewards, axis=1
        )
        
        updated_q_values = updated_q_values * (1 - done)

        masks = action

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            # train the model on the states and updated Q-values
            q_values = self.model(state) # similar to action_probs

            # apply the masks to the Q-values to get the Q-value for the action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # calculate loss between new Q-value and old Q-value
            loss = self.loss_object(updated_q_values, q_action)

            print(loss)

            # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
