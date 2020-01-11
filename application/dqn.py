import tensorflow as tf
import numpy as np


class DQNCore:
    def __init__(self, learning_rate, discount):
        self.discount = discount

        self.__common_state_size = 26

        self.__number_of_field_level_actions_parameters = 6
        self.__field_level_actions_count = 9

        self.__number_of_method_level_actions_parameters = 8
        self.__method_level_actions_count = 13

        self.__number_of_class_level_actions_parameters = 10
        self.__class_level_actions_count = 7

        self.__models = []
        self.__define_models()
        for model in self.__models:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss=tf.losses.Huber(),
                          metrics=[tf.metrics.Accuracy()])
            model.summary()

    def __define_models(self):
        field_level_actions_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=32,
                                  activation=tf.keras.activations.relu,
                                  kernel_initializer=tf.keras.initializers.he_normal(),
                                  input_shape=(
                                      self.__common_state_size + self.__number_of_field_level_actions_parameters,)),
            tf.keras.layers.Dense(units=16,
                                  activation=tf.keras.activations.relu,
                                  kernel_initializer=tf.keras.initializers.he_normal()),
            tf.keras.layers.Dense(units=self.__field_level_actions_count)
        ])

        method_level_actions_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=32,
                                  activation=tf.keras.activations.relu,
                                  kernel_initializer=tf.keras.initializers.he_normal(),
                                  input_shape=(
                                      self.__common_state_size + self.__number_of_method_level_actions_parameters,)),
            tf.keras.layers.Dense(units=16,
                                  activation=tf.keras.activations.relu,
                                  kernel_initializer=tf.keras.initializers.he_normal()),
            tf.keras.layers.Dense(units=self.__method_level_actions_count)
        ])

        class_level_actions_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=32,
                                  activation=tf.keras.activations.relu,
                                  kernel_initializer=tf.keras.initializers.he_normal(),
                                  input_shape=(
                                      self.__common_state_size + self.__number_of_class_level_actions_parameters,)),
            tf.keras.layers.Dense(units=16,
                                  activation=tf.keras.activations.relu,
                                  kernel_initializer=tf.keras.initializers.he_normal()),
            tf.keras.layers.Dense(units=self.__class_level_actions_count)
        ])

        self.__models.append(field_level_actions_model)
        self.__models.append(method_level_actions_model)
        self.__models.append(class_level_actions_model)

    # def get_field_level_actions_model(self):
    #     return self.__models[0]
    #
    # def get_method_level_actions_model(self):
    #     return self.__models[1]
    #
    # def get_class_level_actions_model(self):
    #     return self.__models[2]

    def get_q_values(self, action_type, state):
        return self.__models[action_type].predict(x=[state], batch_size=1)[0].tolist()

    def train(self, action_type, state, q_values):
        self.__models[action_type].fit(x=[state, ], y=[q_values, ], epochs=1)

    def save_models(self, name):
        self.__models[0].save(name.format('_field_base_actions.h5'))
        self.__models[1].save(name.format('_action_base_actions.h5'))
        self.__models[2].save(name.format('_class_base_actions.h5'))

    def load_models(self, name):
        self.__models[0] = tf.keras.models.load_model(name.format('_field_base_actions.h5'))
        self.__models[1] = tf.keras.models.load_model(name.format('_action_base_actions.h5'))
        self.__models[2] = tf.keras.models.load_model(name.format('_class_base_actions.h5'))

    def get_common_state_size(self):
        return self.__common_state_size
    #
    # def get_output_size(self):
    #     return self.__field_level_actions_count
    #
    # def get_q_values(self, state):
    #     return self.model.predict(x=[state], batch_size=1)[0].tolist()
    #
    # def train(self, state, q_values):
    #     self.model.fit(x=[state, ], y=[q_values, ], epochs=1)
    #
    # def load_model(self, path):
    #     self.model = tf.keras.models.load_model(path)


# todo: old_state_q_values[action] = reward + self.discount * np.amax(new_state_q_values)

class DQN:
    __network = DQNCore(learning_rate=0.01, discount=0.95)
    __possible_actions = []

    @staticmethod
    def set_possible_actions(possible_actions):
        possible_actions = sorted(possible_actions, key=lambda t: t[0])
        possible_actions = sorted(possible_actions, key=lambda t: t[1])
        last = object()
        for t in possible_actions:
            if t == last:
                continue
            last = t
            DQN.__possible_actions.append(t)

    @staticmethod
    def get_q_values(action_type, common_state, action_parameters):
        state = common_state + action_parameters
        return DQN.__network.get_q_values(action_type=action_type, state=state)

    @staticmethod
    def get_experience(action_type, action, old_common_state, new_common_state, action_parameters, reward):
        old_state = old_common_state + action_parameters
        old_state_q_values = DQN.__network.get_q_values(action_type, old_state)
        old_state_q_values[action] = reward + \
                                     DQN.__network.discount * DQN.__get_max_q_value_of_a_common_state(new_common_state)
        DQN.__network.train(action_type=action_type, state=old_state, q_values=old_state_q_values)

    @staticmethod
    def __get_max_q_value_of_a_common_state(common_state):
        max_values = []
        for action_type, action_params in DQN.__possible_actions:
            max_values.append(np.amax(DQN.__network.get_q_values(action_type=action_type,
                                                                   state=common_state + action_params)))
        return np.amax(max_values)

    @staticmethod
    def get_common_state_regular_len():
        return DQN.__network.get_common_state_size()

    @staticmethod
    def save_model(name):
        DQN.__network.save_models('./trained_models/{}'.format(name) + '{}')

    @staticmethod
    def load_model(name):
        DQN.__network.load_models('./trained_models/{}'.format(name) + '{}')
