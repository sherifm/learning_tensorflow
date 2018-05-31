#!/usr/bin/env python

import tensorflow as tf

g1 = tf.get_default_graph() #Store default graph in g1
g2 = tf.Graph() #Create additional graph in g2

print(g1 is tf.get_default_graph()) #Check if g1 is default graph: True

#Define g2 as new default graph for the with statmenets
with g2.as_default(): 
	print(g1 is tf.get_default_graph())#Check if g1 is default graph: False

#Check if g1 is again the default after with block: True
print(g1 is tf.get_default_graph())

