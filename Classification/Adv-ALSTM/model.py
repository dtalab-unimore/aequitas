"""
Adv-ALSTM (Feng et al., 2019) in TensorFlow 2.

Port of the original TensorFlow 1 graph (tf.Session, tf.contrib, tf.layers),
which does not run on TensorFlow 2 / Keras 3. Plain tf.Variable and
GradientTape are used, no Keras, and every piece replicates the TF1 one:
- in_fc: dense layer with tanh, Glorot-uniform kernel, zero bias;
- BasicLSTMCell: one kernel [input + units, 4 * units], gate order i, j, f, o,
  forget bias 1.0;
- temporal attention over the LSTM outputs (att_W, att_h, att_u);
- pre_fc prediction layer, hinge loss (or sigmoid + log loss);
- adversarial loss on the latent features, L2 on pre_fc only;
- Adam with the TF1 update rule (epsilon 1e-8).
"""
import numpy as np
import tensorflow as tf


def glorot_uniform(rng, shape):
    # TF1 fans: a 1-D shape uses its size for both
    fan_in, fan_out = (shape[0], shape[0]) if len(shape) == 1 else (shape[0], shape[1])
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return tf.Variable(rng.uniform(-limit, limit, size=shape).astype(np.float32))


def zeros(shape):
    return tf.Variable(np.zeros(shape, dtype=np.float32))


def hinge_loss(labels, logits):
    # tf.losses.hinge_loss: labels {0, 1} -> {-1, 1}, mean over elements
    signs = 2.0 * labels - 1.0
    return tf.reduce_mean(tf.nn.relu(1.0 - signs * logits))


def log_loss(labels, predictions, epsilon=1e-7):
    # tf.losses.log_loss
    return tf.reduce_mean(-labels * tf.math.log(predictions + epsilon)
                          - (1.0 - labels) * tf.math.log(1.0 - predictions + epsilon))


class AdvALSTM(tf.Module):
    def __init__(self, fea_dim, seq, unit, alp, bet, eps, lr, att=True, hinge=True, adv=True, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        tf.random.set_seed(seed)

        self.seq, self.unit = seq, unit
        self.alp, self.bet, self.eps, self.lr = alp, bet, eps, lr
        self.att, self.hinge, self.adv = att, hinge, adv

        self.in_W = glorot_uniform(rng, [fea_dim, fea_dim])
        self.in_b = zeros([fea_dim])
        self.lstm_W = glorot_uniform(rng, [fea_dim + unit, 4 * unit])
        self.lstm_b = zeros([4 * unit])
        if att:
            self.att_W = glorot_uniform(rng, [unit, unit])
            self.att_h = zeros([unit])
            self.att_u = glorot_uniform(rng, [unit])
        self.fc_W = glorot_uniform(rng, [unit * 2 if att else unit, 1])
        self.fc_b = zeros([1])

        self.variables_list = [v for v in self.trainable_variables]
        # Adam slots (TF1 AdamOptimizer)
        self.step = tf.Variable(0.0)
        self.m = [tf.Variable(tf.zeros_like(v)) for v in self.variables_list]
        self.v = [tf.Variable(tf.zeros_like(v)) for v in self.variables_list]

    # ---- forward ------------------------------------------------------
    def features(self, pv):
        x = tf.tanh(tf.tensordot(pv, self.in_W, axes=1) + self.in_b)
        batch = tf.shape(x)[0]
        h = tf.zeros([batch, self.unit])
        c = tf.zeros([batch, self.unit])
        outputs = []
        for t in range(self.seq):
            z = tf.matmul(tf.concat([x[:, t, :], h], axis=1), self.lstm_W) + self.lstm_b
            i, j, f, o = tf.split(z, 4, axis=1)
            c = c * tf.sigmoid(f + 1.0) + tf.sigmoid(i) * tf.tanh(j)
            h = tf.tanh(c) * tf.sigmoid(o)
            outputs.append(h)
        outputs = tf.stack(outputs, axis=1)

        if not self.att:
            return outputs[:, -1, :]

        a_laten = tf.tanh(tf.tensordot(outputs, self.att_W, axes=1) + self.att_h)
        a_alphas = tf.nn.softmax(tf.tensordot(a_laten, self.att_u, axes=1))
        a_con = tf.reduce_sum(outputs * tf.expand_dims(a_alphas, -1), axis=1)
        return tf.concat([outputs[:, -1, :], a_con], axis=1)

    def head(self, features):
        out = tf.matmul(features, self.fc_W) + self.fc_b
        return out if self.hinge else tf.sigmoid(out)

    def loss_fn(self, labels, pred):
        return hinge_loss(labels, pred) if self.hinge else log_loss(labels, pred)

    @tf.function
    def predict(self, pv):
        return self.head(self.features(pv))

    # ---- training -----------------------------------------------------
    @tf.function
    def train_step(self, pv, gt):
        with tf.GradientTape() as tape:
            with tf.GradientTape() as inner:
                fea = self.features(pv)
                inner.watch(fea)
                pred = self.head(fea)
                loss = self.loss_fn(gt, pred)

            adv_loss = 0.0 * loss
            if self.adv:
                # The TF1 graph called tf.stop_gradient on this gradient but
                # discarded the result, so gradients flowed through it: kept as is.
                delta = tf.nn.l2_normalize(inner.gradient(loss, fea), axis=1)
                adv_loss = self.loss_fn(gt, self.head(fea + self.eps * delta))

            l2 = tf.nn.l2_loss(self.fc_W) + tf.nn.l2_loss(self.fc_b)
            obj = loss + self.alp * l2 + self.bet * adv_loss

        grads = tape.gradient(obj, self.variables_list)
        self._adam(grads)
        return obj, loss, l2, adv_loss

    def _adam(self, grads, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.step.assign_add(1.0)
        lr_t = self.lr * tf.sqrt(1.0 - beta2 ** self.step) / (1.0 - beta1 ** self.step)
        for var, g, m, v in zip(self.variables_list, grads, self.m, self.v):
            m.assign(beta1 * m + (1.0 - beta1) * g)
            v.assign(beta2 * v + (1.0 - beta2) * tf.square(g))
            var.assign_sub(lr_t * m / (tf.sqrt(v) + epsilon))

    # ---- persistence --------------------------------------------------
    def save(self, path):
        np.savez(path, *[v.numpy() for v in self.variables_list])

    def load(self, path):
        with np.load(path) as data:
            for i, var in enumerate(self.variables_list):
                var.assign(data[f"arr_{i}"])
