#!/usr/bin/env python

import tensorflow as tf

'''Compare dimenstions'''
A = tf.constant([ [1,2,3],
 [4,5,6] ])
print(A.get_shape())

x = tf.constant([1,0,1])
print(x.get_shape())

'''Dimension of x needs to expand from 1D vector 
to 2D  one-column matrix'''

x = tf.expand_dims(x,1) #Expand dimension of x
print(x.get_shape())
b = tf.matmul(A,x) #Multply matrices
sess = tf.InteractiveSession()
print('matmul result:\n {}'.format(b.eval()))
sess.close()

#Tip: Use tf.transpose() to flip/transpose a matrix.


