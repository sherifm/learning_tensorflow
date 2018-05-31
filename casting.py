#!/usr/bin/env python

import tensorflow as tf

#Specify name and data type attributes
x = tf.constant([1,2,3],name='x',dtype=tf.float32)
print(x.dtype)

#Change a datatype using tf.cast()
x = tf.cast(x,tf.int64)
print(x.dtype)


