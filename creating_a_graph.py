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

#Launch the computational graph in a tf.Session
sess = tf.Session()

#Execute the computational graph in with .run() method
outs = sess.run(f)

#Close the session to free resources
sess.close()

#Print results of executing the graph
print("outs = {}".format(outs))
