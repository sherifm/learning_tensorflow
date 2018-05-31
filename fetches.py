#!/usr/bin/env python

#Import tensorflow
import tensorflow as tf

#Create nondes with constant values
a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

#Create nodes with arithmetic operations
d = tf.multiply(a,b)
e = tf.add(c,b)
f = tf.subtract(d,e)

'''
Run session in 'with' block.Opening a session using the 
with clause will ensure the session is automatically closed
once all computations are done.
'''
with tf.Session() as sess:
	fetches = [a,b,c,d,e,f]
	outs = sess.run(fetches)#Run multple nodes (fetches)
	#Closing session not explicitly needed in 'with' block
print("outs = {}".format(outs))
print(type(outs[0]))