import os
import sys
import math
import networkx as nx
import numpy as np
from itertools import product
from multiprocessing import Process, sharedctypes

parallelism = 8


class DnnInferenceEngine(object):
    def __init__(self, graph):
        self.g = graph

    def run(self, tin):
        self.g.in_node.set_input(tin)
        out = {}
        currents = [self.g.in_node]
        done = set()
        counter = 0
        while (len(currents) != 0):
            nexts = []
            for current in currents:
                skip_current = False
                predecessors = self.g.G.predecessors(current)
                for predecessor in predecessors:
                    if predecessor not in done:
                        nexts.append(predecessor)
                        skip_current = True
                if skip_current:
                    continue
                current.run(counter)
                if not isinstance(current, Input):
                    counter += 1
                if self.g.is_out_node(current):
                    out = current.result
                done.add(current)
                for successor in self.g.G.successors(current):
                    nexts.append(successor)
            currents = nexts
        return out


class DnnGraphBuilder(object):
    def __init__(self):
        self.G = nx.DiGraph()
        self.name_num = {"conv2d": 0,
                         "bias_add": 0,
                         "max_pool2d": 0,
                         "batch_norm": 0,
                         "leaky_relu": 0,
                         "input": 0}
        self.in_node = None
        self.out_node = None

    def set_in_node(self, node):
        self.in_node = node

    def set_out_node(self, node):
        self.out_node = node

    def is_out_node(self, node):
        if self.out_node is node:
            return True
        else:
            return False

    def get_name(self, layer_name):
        name = layer_name + "_" + str(self.name_num[layer_name])
        self.name_num[layer_name] += 1
        return name

    def create_conv2d(self, in_node, kernel, strides, padding):
        out_node = Conv2D(self.get_name("conv2d"), in_node,
                          kernel, strides, padding)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_bias_add(self, in_node, biases):
        out_node = BiasAdd(self.get_name("bias_add"), in_node, biases)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_max_pool2d(self, in_node, ksize, strides, padding):
        out_node = MaxPool2D(self.get_name("max_pool2d"),
                             in_node, ksize, strides, padding)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_batch_norm(self, in_node, mean, variance, gamma, epsilon):
        out_node = BatchNorm(self.get_name("batch_norm"),
                             in_node, mean, variance, gamma, epsilon)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_leaky_relu(self, in_node):
        out_node = LeakyReLU(self.get_name("leaky_relu"), in_node)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_input(self, in_shape):
        out_node = Input(self.get_name("input"), in_shape)
        self.G.add_node(out_node)
        self.set_in_node(out_node)  # Assume there's only one input
        return out_node


class DnnNode(object):
    def __init__(self):
        pass

    def run(self, counter):
        self.result = None

#
# Implement the classes below.
#


class Conv2D(DnnNode):
    def __init__(self, name, in_node, kernel, strides, padding):
        raise NotImplementedError('Conv2d is not implemented yet')

    def run(self, counter):
        pass


class BiasAdd(DnnNode):
    def __init__(self, name, in_node, biases):
        raise NotImplementedError('BiasAdd is not implemented yet')

    def run(self, counter):
        pass


class MaxPool2D(DnnNode):
    def __init__(self, name, in_node, ksize, strides, padding):
        raise NotImplementedError('MaxPool2D is not implemented yet')

    def run(self, counter):
        pass


class BatchNorm(DnnNode):
    def __init__(self, name, in_node, mean, variance, gamma, epsilon):
        raise NotImplementedError('BatchNorm is not implemented yet')

    def run(self, counter):
        pass


class LeakyReLU(DnnNode):
    def __init__(self, name, in_node):
        raise NotImplementedError('LeakyReLU is not implemented yet')

    def run(self, counter):
        pass


class Input(DnnNode):
    def __init__(self, name, in_shape):
        self.name = name
        self.in_shape = in_shape
        self.result = np.ndarray(self.in_shape)

    def set_input(self, tensor):
        assert tuple(self.in_shape) == tuple(tensor.shape)
        self.result = tensor

    def run(self, counter):
        pass
