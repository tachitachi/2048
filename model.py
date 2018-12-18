import tensorflow as tf
import numpy as np

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

# Simple deep Q learning for 2048

class Model(object):
    def __init__(self, obs_space, action_space, gamma=0.99, tau=0.001):
        self.obs_space = obs_space
        self.action_space = action_space
        self.gamma = gamma
        self.tau = tau

        graph = tf.Graph()
        with graph.as_default():

            self.x = tf.placeholder(tf.float32, [None] + list(self.obs_space))
            self.input_action = tf.placeholder(tf.float32, [None, self.action_space])
            self.targets = tf.placeholder(tf.float32, [None])

            def network(x, scope, reuse=False):
                with tf.variable_scope(scope, reuse=reuse):
                    net = flatten(x)
                    net = tf.nn.relu(tf.layers.dense(net, 256))

                    # value
                    net = tf.layers.dense(net, self.action_space)
                    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

                    return net, params

            self.value, network_params = network(self.x, scope='value')
            self.target_value, target_params = network(self.x, scope='target_value')

            update_targets = []
            for p0, p1 in zip(network_params, target_params):
                update_targets.append(p1.assign(p0 * self.tau + p1 * (1 - self.tau)))
            self.update_targets = tf.group(update_targets)

            copy_targets = []
            for p0, p1 in zip(network_params, target_params):
                copy_targets.append(p1.assign(p0))
            self.copy_targets = tf.group(copy_targets)



            loss = tf.reduce_mean((tf.reduce_sum(self.value * self.input_action, axis=1) - self.targets)**2)

            opt = tf.train.AdamOptimizer(1e-3)
            train_tensor = opt.minimize(loss)
            with tf.control_dependencies([train_tensor]):
                self.train_op = tf.identity(loss, name='train_op')

            self.init_fn = tf.global_variables_initializer()



            self.sess = tf.Session(graph=graph)

    def init(self):
        self.sess.run(self.init_fn)
        self.sess.run(self.copy_targets)

    #def train(self, x0, action, value, update_target=False):
    def train(self, x0, action, reward, x1, update_target=False):
        target_value = self.sess.run(self.target_value, {self.x: x1})
        #return target_value

        action = one_hot(np.asarray(action), self.action_space)

        targets = np.asarray(reward) + self.gamma * np.max(target_value, axis=1)

        loss = self.sess.run(self.train_op, {self.x: x0, self.input_action: action, self.targets: targets})
        self.sess.run(self.update_targets)

        return loss

    def predict(self, x):
        values = self.sess.run(self.value, {self.x: x})
        return np.argmax(values, axis=1)


def one_hot(x, size):
    b = np.zeros((x.shape[0], size))
    b[np.arange(x.shape[0]), x] = 1
    return b

class ReplayBuffer(object):
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size

    def add(self, inputs):
        self.buffer.append(inputs)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, count):
        indices = np.random.choice(count, (len(self.buffer,)))

        items = []
        for idx in indices:
            items.append(self.buffer[idx])

        return tuple(zip(*items))

    def __len__(self):
        return len(self.buffer)

if __name__ == '__main__':
    obs_space = (4, 4, 14)
    action_space = 4
    m = Model(obs_space, action_space)
    m.init()

    x = np.random.random([5] + list(obs_space))
    action = one_hot(np.random.choice(action_space, 5), action_space)
    print(action)
    targets = np.random.random((5,)) * 100
    print(m.train(x, action, targets))
    print(m.predict(x))