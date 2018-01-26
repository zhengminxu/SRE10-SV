import tensorflow as tf

def model_input():
	input_ = tf.placeholder(tf.float32, [None, None,100]) # number of speakers x number of recordings x lda dimension
	return input_

def model(input_, alpha, beta):
	dense1 = tf.layers.dense(inputs = input_, units = 100, activation=tf.nn.tanh) # [None, 100]
	out = tf.layers.dense(inputs = dense1, units = 100, activation=None)
	
	id_dim = 75
	hs_i = tf.slice(dense1, [0,0,0], [-1,-1,id_dim]) # None, 4, 75
	
	mean_hs_i = tf.reduce_mean(hs_i, axis = 1) # None, 75
	all_mean_hs = tf.reduce_mean(tf.reshape(hs_i, [-1,75]), axis = 0) # 1, 75
	
	speaker_id_loss = tf.reduce_mean(tf.reduce_mean(tf.norm(hs_i - tf.reshape(mean_hs_i,[-1,1,id_dim]), axis = 2), axis = 1))
	reconstruct_loss = tf.losses.mean_squared_error(input_, out)
	internal_disp_loss = tf.reduce_mean(tf.norm(mean_hs_i - all_mean_hs, axis = 1))

	reg_term = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.01), tf.trainable_variables())
	loss = reconstruct_loss + alpha * (beta * speaker_id_loss - (1-beta) * internal_disp_loss) + reg_term
	
	return hs_i, loss


def model_opt(loss, learning_rate):
	opt = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
	return opt

class Autoencoder:
    def __init__(self, alpha=0.001, beta=0.2, learning_rate=0.01):
        tf.reset_default_graph()
        self.input_ = model_input()
        self.encoded, self.loss = model(self.input_, alpha, beta)
        self.opt = model_opt(self.loss, learning_rate)
        