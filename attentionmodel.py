from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf

class PIModel(object):
    def __init__(self, config, pretrained_embeddings):
        self.config = config
        self.embeddings = tf.Variable(pretrained_embeddings, trainable=self.config.retrain_embeddings)
        self.add_placeholders()
        self.add_embeddings()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_train_op()

    def add_placeholders(self):
        self.prem_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_prem_len))
        self.prem_len_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.hyp_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_hyp_len))
        self.hyp_len_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.label_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())
        self.learning_rate_placeholder = tf.placeholder(tf.float32, shape=(1,))

    def create_feed_dict(self, prem_batch, prem_len, hyp_batch, hyp_len, dropout, learning_rate = None, label_batch=None):
        feed_dict = {
            self.prem_placeholder: prem_batch,
            self.prem_len_placeholder: prem_len,
            self.hyp_placeholder: hyp_batch,
            self.hyp_len_placeholder: hyp_len,
            self.dropout_placeholder: dropout
        }
        if label_batch is not None:
            feed_dict[self.label_placeholder] = label_batch
        if learning_rate is not None:
            feed_dict[self.learning_rate_placeholder] = learning_rate
        else:
            learning_rate = 0
        return feed_dict

    def add_embeddings(self):
        self.embed_prems = tf.nn.embedding_lookup(self.embeddings, self.prem_placeholder)
        self.embed_hyps = tf.nn.embedding_lookup(self.embeddings, self.hyp_placeholder)

    def add_prediction_op(self):
         
        initer = tf.contrib.layers.xavier_initializer()

        # Recurrent cells
        if False:
            with tf.variable_scope("premhyp"):
                prem_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.config.state_size), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                _, outputs = tf.nn.dynamic_rnn(prem_cell, tf.concat([tf.boolean_mask(self.embed_prems, tf.sequence_mask(self.prem_len_placeholder, 20)), self.embed_hyps], 1),\
                              sequence_length=self.prem_len_placeholder + self.hyp_len_placeholder, dtype=tf.float32)
        if True:
            with tf.variable_scope("prem"):
                prem_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.config.state_size), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                new_prems, prem_out = tf.nn.dynamic_rnn(prem_cell, self.embed_prems,\
                              sequence_length=self.prem_len_placeholder, dtype=tf.float32)
            with tf.variable_scope("hyp"):
                hyp_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.config.state_size), output_keep_prob = self.dropout_placeholder,state_keep_prob = self.dropout_placeholder)
                _, outputs = tf.nn.dynamic_rnn(hyp_cell, self.embed_hyps,\
                             sequence_length=self.hyp_len_placeholder, initial_state=prem_out)
        h = outputs
        if False:
            Wy = tf.Variable(initer([1,1,self.config.state_size, self.config.state_size]))
            Wh = tf.Variable(initer([self.config.state_size, self.config.state_size]))
            w =  tf.Variable(initer([1,1,self.config.state_size]))
            M = tf.tanh(tf.reduce_sum(tf.multiply(Wy, tf.expand_dims(new_prems,3)), 3) + tf.expand_dims(tf.matmul(outputs, Wh), 1))
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(w, M), 2), dim = 1)
            r = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, 2), new_prems), 1)
            Wp = tf.Variable(initer([self.config.state_size, self.config.state_size]))
            Wx= tf.Variable(initer([self.config.state_size, self.config.state_size]))
            h = tf.tanh(tf.matmul(r, Wp) + tf.matmul(outputs, Wx))

        # softmax
        
        Ws = tf.Variable(initer([self.config.state_size,3]))
        bs = tf.Variable(tf.zeros([1,3]) + 1e-3)
        self.logits = tf.matmul(h, Ws) + bs


    def add_loss_op(self):
        beta = self.config.l2
        reg = 0
        for v in tf.trainable_variables():
            reg = reg + tf.nn.l2_loss(v)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_placeholder, logits=self.logits) + beta*reg) 

    def add_train_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder[0])
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config.max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def optimize(self, sess, train_x, train_y, lr):
        prem_batch, prem_len, hyp_batch, hyp_len, dropout = train_x
        label_batch = train_y
        input_feed = self.create_feed_dict(prem_batch, prem_len, hyp_batch, hyp_len, dropout, lr, label_batch )
        output_feed = [self.train_op, self.logits, self.loss]
        _, logits, loss = sess.run(output_feed, input_feed)
        return np.argmax(logits, axis=1), loss

    def validate(self, sess, valid_x, valid_y):
        prem_batch, prem_len, hyp_batch, hyp_len = valid_x
        label_batch = valid_y
        input_feed = self.create_feed_dict(prem_batch, prem_len, hyp_batch, hyp_len,1, [0], label_batch)
        output_feed = [self.logits, self.loss]
        logits, loss = sess.run(output_feed, input_feed)
        return np.argmax(logits, axis=1), loss

    def predict(self, sess, test_x):
        prem_batch, prem_len, hyp_batch, hyp_len = test_x
        input_feed = self.create_feed_dict(prem_batch, prem_len, hyp_batch, hyp_len, 1)
        output_feed = [self.logits]
        logits = sess.run(output_feed, input_feed)
        return np.argmax(logits[0], axis=1)

    def run_train_epoch(self, sess, dataset, lr):
        print(np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()]))
        preds = []
        labels = []
        losses = 0.
        x = 0
        for prem, prem_len, hyp, hyp_len, label in dataset:
            pred, loss = self.optimize(sess, (prem, prem_len, hyp, hyp_len, self.config.dropout), label, lr)
            preds.extend(pred)
            labels.extend(label)
            losses += loss * len(label)
            x += 1
        return preds, labels, losses / len(labels)

    def run_val_epoch(self, sess, dataset):
        preds = []
        labels = []
        losses = 0.
        for prem, prem_len, hyp, hyp_len, label in dataset:
            pred, loss = self.validate(sess, (prem, prem_len, hyp, hyp_len), label)
            preds.extend(pred)
            labels.extend(label)
            losses += loss * len(label)
        return preds, labels, losses / len(labels)

    def run_test_epoch(self, sess, dataset):
        preds = []
        labels = []
        losses = 0.
        for prem, prem_len, hyp, hyp_len, label in dataset:
            pred = self.predict(sess, (prem, prem_len, hyp, hyp_len))
            preds.extend(pred)
            labels.extend(label)
        return preds, labels
