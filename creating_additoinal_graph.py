#!/usr/bin/env python

import tensorflow as tf


print(tf.get_default_graph()) #Print default graph
g = tf.Graph() #Create an additional graph
print(g) #Print additiona graph

#Creat a node with constant
a = tf.constant(5)#Default graph is used

#Check if node a uses graph g
print(a.graph is g)

#Check if node a uses default graph
print(a.graph is tf.get_default_graph())