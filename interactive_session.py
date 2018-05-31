#!/usr/bin/env python

import tensorflow as tf

'''
Use InteractiveSession() and .eval() to  get a
full look at the values without the need to constantly 
refer to the session object.

tf.InteractiveSession() allows you to replace the usual tf.Ses
sion(), so that you don’t need a variable holding the session for
running ops. This can be useful in interactive Python environ‐
ments, like when writing IPython notebooks, for instance.
'''

sess = tf.InteractiveSession()
c = tf.linspace(0.0, 4.0,5)
print("The content of 'c':\n {}\n".format(c.eval()))
sess.close()