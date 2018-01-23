#coding:utf-8
'''
@author:liaoliwei

'''

import tensorflow as tf
import numpy as np
import lenet5_inference
import pre
import os
import lenet5_train

def pic(path):
	with tf.Graph().as_default() as g:
		x = tf.placeholder(tf.float32,[
			1,
			lenet5_inference.IMAGE_SIZE,
			lenet5_inference.IMAGE_SIZE,
			lenet5_inference.NUM_CHANNELS])
		xs = pre.pre(path)
		reshaped_xs = np.reshape(xs, (
                1,
                lenet5_inference.IMAGE_SIZE,
                lenet5_inference.IMAGE_SIZE,
                lenet5_inference.NUM_CHANNELS))
		pic_feed = {x:reshaped_xs}
		regularizer = tf.contrib.layers.l2_regularizer(lenet5_train.REGULARIZATION_RATE)
		y = lenet5_inference.inference(x, False, regularizer)
		variable_averages = tf.train.ExponentialMovingAverage(lenet5_train.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)
		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(lenet5_train.MODEL_SAVE_PATH)
			print('loading model...\n\n')
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				yy = sess.run(y, feed_dict=pic_feed)
				res = sess.run(tf.argmax(yy,1))
				print('预测结果是：',res)
			else:
				print("No checkpoint file found!")
				return
def main(argv=None):
	path = input("input path:")
	pic(path)

if __name__=="__main__":
	main()
