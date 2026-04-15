"""TensorFlow v1-style public PURS model, adapted to run via tf.compat.v1."""

import tensorflow as tf


tf1 = tf.compat.v1


class Model(object):
    """Public PURS architecture with minimal compatibility updates."""

    def __init__(
        self,
        user_count,
        item_count,
        batch_size,
        hidden_size=128,
        long_memory_window=10,
        short_memory_window=3,
    ):
        self.batch_size = int(batch_size)
        self.long_memory_window = int(long_memory_window)
        self.short_memory_window = int(short_memory_window)

        self.u = tf1.placeholder(tf.int32, [self.batch_size])
        self.i = tf1.placeholder(tf.int32, [self.batch_size])
        self.y = tf1.placeholder(tf.float32, [self.batch_size])
        self.hist = tf1.placeholder(tf.int32, [self.batch_size, self.long_memory_window])
        self.lr = tf1.placeholder(tf.float64, [])

        user_emb_w = tf1.get_variable("user_emb_w", [user_count, hidden_size // 2])
        item_emb_w = tf1.get_variable("item_emb_w", [item_count, hidden_size // 2])
        user_b = tf1.get_variable(
            "user_b", [user_count], initializer=tf.constant_initializer(0.0)
        )
        item_b = tf1.get_variable(
            "item_b", [item_count], initializer=tf.constant_initializer(0.0)
        )

        item_emb = tf.concat(
            [
                tf.nn.embedding_lookup(item_emb_w, self.i),
                tf.nn.embedding_lookup(user_emb_w, self.u),
            ],
            axis=1,
        )
        item_b = tf.gather(item_b, self.i)
        user_b = tf.gather(user_b, self.u)

        h_emb = tf.concat(
            [
                tf.nn.embedding_lookup(
                    item_emb_w,
                    tf.slice(self.hist, [0, 0], [self.batch_size, self.long_memory_window]),
                ),
                tf.tile(
                    tf.expand_dims(tf.nn.embedding_lookup(user_emb_w, self.u), 1),
                    [1, self.long_memory_window, 1],
                ),
            ],
            axis=2,
        )

        unexp_emb = tf.concat(
            [
                tf.nn.embedding_lookup(
                    item_emb_w,
                    tf.slice(
                        self.hist,
                        [0, self.long_memory_window - self.short_memory_window],
                        [self.batch_size, self.short_memory_window],
                    ),
                ),
                tf.tile(
                    tf.expand_dims(tf.nn.embedding_lookup(user_emb_w, self.u), 1),
                    [1, self.short_memory_window, 1],
                ),
            ],
            axis=2,
        )

        h_long_emb = tf.nn.embedding_lookup(
            item_emb_w,
            tf.slice(self.hist, [0, 0], [self.batch_size, self.long_memory_window]),
        )

        h_short_emb = tf.nn.embedding_lookup(
            item_emb_w,
            tf.slice(
                self.hist,
                [0, self.long_memory_window - self.short_memory_window],
                [self.batch_size, self.short_memory_window],
            ),
        )

        # Use Keras GRU for forward compatibility with modern TensorFlow builds.
        long_output = tf.keras.layers.GRU(
            hidden_size,
            return_sequences=True,
            name="long_gru",
        )(h_emb)

        long_preference, _ = self.seq_attention(
            long_output, hidden_size, self.long_memory_window
        )
        # Keep public behavior: tf1 dropout arg is keep_prob.
        long_preference = tf1.nn.dropout(long_preference, keep_prob=0.1)

        concat = tf.concat([long_preference, item_emb], axis=1)
        concat = tf.keras.layers.BatchNormalization(name="concat_bn")(concat, training=False)
        concat = tf.keras.layers.Dense(80, activation=tf.nn.sigmoid, name="f1")(concat)
        concat = tf.keras.layers.Dense(40, activation=tf.nn.sigmoid, name="f2")(concat)
        concat = tf.keras.layers.Dense(1, activation=None, name="f3")(concat)
        concat = tf.reshape(concat, [-1])

        unexp_factor = self.unexp_attention(
            item_emb, unexp_emb, [self.long_memory_window] * self.batch_size
        )
        unexp_factor = tf.keras.layers.BatchNormalization(name="unexp_bn")(
            unexp_factor, training=False
        )
        unexp_factor = tf.reshape(unexp_factor, [-1, hidden_size])
        unexp_factor = tf.keras.layers.Dense(hidden_size, name="unexp_f1")(unexp_factor)
        unexp_factor = tf.keras.layers.Dense(1, activation=None, name="unexp_f2")(unexp_factor)
        unexp_factor = tf.reshape(unexp_factor, [-1])

        self.center = self.mean_shift(h_long_emb)
        unexp = tf.reduce_mean(self.center, axis=1)
        unexp = tf.norm(unexp - tf.nn.embedding_lookup(item_emb_w, self.i), ord="euclidean", axis=1)
        self.unexp = unexp
        unexp = tf.exp(-1.0 * unexp) * unexp
        unexp = tf.stop_gradient(unexp)

        relevance = tf.reduce_mean(h_long_emb, axis=1)
        relevance = tf.norm(
            relevance - tf.nn.embedding_lookup(item_emb_w, self.i),
            ord="euclidean",
            axis=1,
        )
        _ = relevance

        annoyance = tf.reduce_mean(h_short_emb, axis=1)
        annoyance = tf.norm(
            annoyance - tf.nn.embedding_lookup(item_emb_w, self.i),
            ord="euclidean",
            axis=1,
        )
        _ = annoyance

        self.logits = item_b + concat + user_b + unexp_factor * unexp
        self.score = tf.sigmoid(self.logits)

        self.global_step = tf1.Variable(0, trainable=False, name="global_step")
        self.global_epoch_step = tf1.Variable(0, trainable=False, name="global_epoch_step")
        self.global_epoch_step_op = tf1.assign(
            self.global_epoch_step, self.global_epoch_step + 1
        )

        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        )
        trainable_params = tf1.trainable_variables()
        self.opt = tf1.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 1)
        self.train_op = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step
        )

    def train(self, sess, uij, lr):
        loss, _ = sess.run(
            [self.loss, self.train_op],
            feed_dict={
                self.u: uij[0],
                self.hist: uij[1],
                self.i: uij[2],
                self.y: uij[3],
                self.lr: lr,
            },
        )
        return loss

    def test(self, sess, uij):
        score, unexp = sess.run(
            [self.score, self.unexp],
            feed_dict={
                self.u: uij[0],
                self.hist: uij[1],
                self.i: uij[2],
                self.y: uij[3],
            },
        )
        return score, uij[3], uij[0], uij[2], unexp

    def save(self, sess, path):
        saver = tf1.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf1.train.Saver()
        saver.restore(sess, save_path=path)

    def extract_axis_1(self, data, ind):
        batch_range = tf.range(tf.shape(data)[0])
        indices = tf.stack([batch_range, ind], axis=1)
        res = tf.gather_nd(data, indices)
        return res

    def seq_attention(self, inputs, hidden_size, attention_size):
        w_omega = tf.Variable(tf.random.normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random.normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random.normal([attention_size], stddev=0.1))
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name="vu")
        alphas = tf.nn.softmax(vu, name="alphas")
        output = tf.reduce_sum(
            inputs * tf.tile(tf.expand_dims(alphas, -1), [1, 1, hidden_size]),
            1,
            name="attention_embedding",
        )
        return output, alphas

    def unexp_attention(self, querys, keys, keys_id):
        _ = keys_id
        querys = tf.expand_dims(querys, 1)
        keys_length = tf.shape(keys)[1]
        embedding_size = querys.get_shape().as_list()[-1]
        keys = tf.reshape(keys, shape=[-1, keys_length, embedding_size])
        querys = tf.reshape(
            tf.tile(querys, [1, keys_length, 1]), shape=[-1, keys_length, embedding_size]
        )

        net = tf.concat([keys, keys - querys, querys, keys * querys], axis=-1)
        for units in [32, 16]:
            net = tf.keras.layers.Dense(units=units, activation=tf.nn.relu)(net)
        att_wgt = tf.keras.layers.Dense(units=1, activation=tf.sigmoid)(net)
        outputs = tf.reshape(att_wgt, shape=[-1, 1, keys_length], name="weight")
        scores = outputs
        scores = scores / (embedding_size ** 0.5)
        scores = tf.nn.softmax(scores)
        outputs = tf.matmul(scores, keys)
        outputs = tf.reduce_sum(outputs, 1, name="unexp_embedding")
        return outputs

    def mean_shift(self, input_X, window_radius=0.2):
        X1 = tf.expand_dims(tf.transpose(input_X, perm=[0, 2, 1]), 1)
        X2 = tf.expand_dims(input_X, 1)
        C = input_X

        def _mean_shift_step(C):
            C4 = tf.expand_dims(C, 3)
            Y = tf.reduce_sum(tf.pow((C4 - X1) / window_radius, 2), axis=2)
            gY = tf.exp(-Y)
            num = tf.reduce_sum(tf.expand_dims(gY, 3) * X2, axis=2)
            denom = tf.reduce_sum(gY, axis=2, keepdims=True)
            return num / denom

        def _mean_shift(i, C, max_diff):
            new_C = _mean_shift_step(C)
            max_diff = tf.reshape(
                tf.reduce_max(tf.sqrt(tf.reduce_sum(tf.pow(new_C - C, 2), axis=1))), []
            )
            return i + 1, new_C, max_diff

        def _cond(i, C, max_diff):
            _ = (i, C)
            return max_diff > 1e-5

        _, C, _ = tf.while_loop(
            cond=_cond,
            body=_mean_shift,
            loop_vars=(tf.constant(0), C, tf.constant(1e10)),
        )
        return C
