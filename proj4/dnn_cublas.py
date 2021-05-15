import os
import sys
import math
import networkx as nx
import numpy as np
import multiprocessing as mp
import time
from ctypes import *

mylib = cdll.LoadLibrary('./lib_dnn_cublas.so')

class DnnInferenceEngine(object):
    def __init__(self, graph, debug):
        self.g = graph
        self.debug = debug

    def run(self, tin):
        self.g.in_node.set_input(tin)
        out = {}
        currents = [self.g.in_node]
        done = set()
        i = 0
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
                current.run()
                if self.g.is_out_node(current):
                    out = current.result
                done.add(current)
                for successor in self.g.G.successors(current):
                    nexts.append(successor)
                if self.debug:
                    np.save("intermediate/layer_" + str(i) + ".npy", current.result)
                    i += 1
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
        return self.out_node is node

    def get_name(self, layer_name):
        name = layer_name + "_" + str(self.name_num[layer_name])
        self.name_num[layer_name] += 1
        return name

    def create_conv2d(self, in_node, kernel, strides, padding):
        out_node = Conv2D(self.get_name("conv2d"), in_node, kernel, strides, padding)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_bias_add(self, in_node, biases):
        out_node = BiasAdd(self.get_name("bias_add"), in_node, biases)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_max_pool2d(self, in_node, ksize, strides, padding):
        out_node = MaxPool2D(self.get_name("max_pool2d"), in_node, ksize, strides, padding)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_batch_norm(self, in_node, mean, variance, gamma, epsilon):
        out_node = BatchNorm(self.get_name("batch_norm"), in_node, mean, variance, gamma, epsilon)
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

    def run(self):
        self.result = None 

#
# Complete below classes.
#

class Conv2D(DnnNode):
    def __init__(self, name, in_node, kernel, strides, padding):
        if in_node.result.shape[3] != kernel.shape[2]:
            raise ValueError
        self.name = name
        self.in_node = in_node
        self.kernel = kernel
        self.strides = strides
        self.padding = padding
        self.n_in, self.h_in, self.w_in, self.c_in = self.in_node.result.shape
        self.h_ker, self.w_ker, self.c_ker, self.n_ker = self.kernel.shape
        self.h_stride, self.w_stride = self.strides[1:3]

        if self.padding == 'SAME':
            self.h_out = math.ceil(float(self.h_in)/float(self.h_stride))
            self.w_out = math.ceil(float(self.w_in)/float(self.w_stride))
            self.h_pad = (self.h_out-1)*self.h_stride+self.h_ker-self.h_in
            self.w_pad = (self.w_out-1)*self.w_stride+self.w_ker-self.w_in
        elif self.padding == 'VALID':
            self.h_pad = 0
            self.w_pad = 0
            self.h_out = math.ceil(float(self.h_in - self.h_ker + 1)/float(self.h_stride))
            self.w_out = math.ceil(float(self.w_in - self.w_ker + 1)/float(self.w_stride))
        else:
            raise ValueError

        self.result = np.zeros((self.n_in, self.h_out, self.w_out, self.n_ker))

        print(self.name)

    def multiprocess_worker(self, n, start_h, end_h):
        res = np.ndarray((end_h - start_h, self.w_out, self.n_ker))
        for h in range(start_h, end_h):
            for w in range(self.w_out):
                for n_k in range(self.n_ker):
                    res[h-start_h, w, n_k] = np.sum(np.multiply(self.img_pad[n, h*self.h_stride : h*self.h_stride+self.h_ker, w*self.w_stride:w*self.w_stride+self.w_ker,:], self.kernel[:,:,:,n_k]))
        return res

    def run(self):
        top_pad = self.h_pad//2
        bot_pad = self.h_pad - top_pad

        left_pad = self.w_pad//2
        right_pad = self.w_pad - left_pad

        self.img_pad = np.pad(self.in_node.result, ((0,0),(top_pad, bot_pad), (left_pad, right_pad),(0,0)), 'constant', constant_values=(0))
        #num_process = 8

        #for n in range(self.n_in):
        #    loopCnt = math.ceil(self.h_out / float(num_process))
        #    with mp.Pool(processes=num_process, initializer=None, initargs=None) as pool:
        #        res = pool.starmap(self.multiprocess_worker, [[n, i*num_process, min((i+1)*num_process, self.h_out)] for i in range(loopCnt)])
        #        self.result[n] = np.vstack(res)

        tik = time.time()
        x = np.zeros([self.n_in, self.h_out, self.w_out, self.h_ker, self.w_ker, self.c_in])
        for i in range(self.h_out):
            ih = i * self.h_stride
            for j in range(self.w_out):
                jw = j * self.w_stride
                x[:, i, j, :, :, :] = self.img_pad[:, ih:ih + self.h_ker, jw:jw + self.w_ker, :]

        x = x.reshape([self.n_in * self.h_out * self.w_out, -1])
        if x.flags['C_CONTIGUOUS']:
            x = np.asfortranarray(x)

        w = self.kernel.reshape([-1, self.n_ker])
        if w.flags['C_CONTIGUOUS']:
            w = np.asfortranarray(w)

        self.result = np.zeros((x.shape[0], w.shape[1]), dtype=np.float32, order='F')

        #self.result = x.dot(w)
        mylib.cublas_mul_float(
            x.astype(np.float32).ctypes.data_as(POINTER(c_float)),
            w.astype(np.float32).ctypes.data_as(POINTER(c_float)),
            self.result.ctypes.data_as(POINTER(c_float)),
            x.shape[0],
            x.shape[1],
            w.shape[1])

        self.result = np.ascontiguousarray(self.result)
        self.result = self.result.reshape([self.n_in, self.h_out, self.w_out, self.n_ker])
        tok = time.time() - tik
        print("conv 2d : %.3f" % tok)
      
class BiasAdd(DnnNode):
    def __init__(self, name, in_node, biases):
        if not in_node.result.shape[3] == biases.shape[0]:
            raise ValueError

        self.name = name
        self.in_node = in_node
        self.biases = biases
        self.result = in_node.result
        print(self.name)

    def run(self):
        self.result = self.in_node.result + self.biases

class MaxPool2D(DnnNode):
    def __init__(self, name, in_node, ksize, strides, padding):
        self.name = name
        self.in_node = in_node
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

        self.n_in, self.h_in, self.w_in, self.c_in = self.in_node.result.shape
        self.h_ker, self.w_ker = self.ksize[1:3]
        self.h_stride, self.w_stride = self.strides[1:3]
        if self.padding == 'SAME':
            self.h_out = math.ceil(float(self.h_in)/float(self.h_stride))
            self.w_out = math.ceil(float(self.w_in)/float(self.w_stride))
            self.h_pad = (self.h_out-1)*self.h_stride+self.h_ker-self.h_in
            self.w_pad = (self.w_out-1)*self.w_stride+self.w_ker-self.w_in
        elif self.padding == 'VALID':
            self.h_pad = 0
            self.w_pad = 0
            self.h_out = math.ceil(float(self.h_in - self.h_ker + 1)/float(self.h_stride))
            self.w_out = math.ceil(float(self.w_in - self.w_ker + 1)/float(self.w_stride))
        else:
            raise ValueError

        self.result = np.zeros((self.n_in, self.h_out, self.w_out, self.c_in))

        print(self.name)

    def run(self):
        top_pad = self.h_pad//2
        bot_pad = self.h_pad - top_pad

        left_pad = self.w_pad//2
        right_pad = self.w_pad - left_pad

        img_pad = np.pad(self.in_node.result, ((0,0),(top_pad, bot_pad), (left_pad, right_pad),(0,0)), 'edge')
    
        tik = time.time()
        #for n in range(self.n_in):
        #    for h in range(self.h_out):
        #        for w in range(self.w_out):
        #            for c in range(self.c_in):
        #                self.result[n, h, w, c] = np.max(img_pad[n, h*self.h_stride : h*self.h_stride + self.h_ker, w*self.w_stride : w*self.w_stride + self.w_ker, c])
        x = np.zeros([self.n_in, self.h_out, self.w_out, self.h_ker, self.w_ker, self.c_in])
        for i in range(self.h_out):
            ih = i * self.h_stride
            for j in range(self.w_out):
                jw = j * self.w_stride
                x[:, i, j, :, :, :] = img_pad[:, ih:ih + self.h_ker, jw:jw + self.w_ker, :]

        x_shape = x.shape
        x = x.reshape([x_shape[0] * x_shape[1] * x_shape[2], x_shape[3] * x_shape[4], -1])

        #self.result = np.max(x, axis=1)
        self.result = np.zeros((x.shape[0], x.shape[2]), dtype=np.float32)
        mylib.cublas_max_pool_float(
            x.astype(np.float32).ctypes.data_as(POINTER(c_float)),
            self.result.ctypes.data_as(POINTER(c_float)),
            x.shape[0],
            x.shape[1],
            x.shape[2])

        self.result = self.result.reshape([x_shape[0], x_shape[1], x_shape[2], -1])
        tok = time.time() - tik
        print("max pool 2d : %.3f" % tok)
        

class BatchNorm(DnnNode):
    def __init__(self, name, in_node, mean, variance, gamma, epsilon):
        dim_ch = in_node.result.shape[3]
        cond_mean = dim_ch == mean.shape[0] 
        cond_variance = dim_ch == variance.shape[0]
        cond_gamma = dim_ch == gamma.shape[0]

        if not cond_mean or not cond_variance or not cond_gamma:
            raise ValueError

        self.name = name
        self.in_node = in_node
        self.mean = mean
        self.variance = variance
        self.gamma = gamma
        self.epsilon = epsilon
        self.result = in_node.result
        print(self.name)

    def run(self):
        tik = time.time()
        #n, h, w, c = self.in_node.result.shape
        #bn = lambda x, m, v, g: g * ((x - m) / np.sqrt(v + self.epsilon))
        #for i in range(h):
        #    for j in range(w):
        #        self.result[0, i, j] = np.vectorize(bn)(self.in_node.result[0, i, j], self.mean, self.variance, self.gamma)
        in_shape = self.in_node.result.shape
        x = self.in_node.result.reshape([-1, in_shape[3]])
        self.result = np.zeros(x.shape, dtype=np.float32)

        mylib.cublas_norm_float(
            x.astype(np.float32).ctypes.data_as(POINTER(c_float)),
            self.result.ctypes.data_as(POINTER(c_float)),
            x.shape[0],
            x.shape[1],
            self.gamma.astype(np.float32).ctypes.data_as(POINTER(c_float)),
            self.mean.astype(np.float32).ctypes.data_as(POINTER(c_float)),
            self.variance.astype(np.float32).ctypes.data_as(POINTER(c_float)),
            c_float(self.epsilon))

        self.result = self.result.reshape(in_shape)
        tok = time.time() - tik
        print("batch norm : %.3f" % tok)

class LeakyReLU(DnnNode):
    def __init__(self, name, in_node):
        self.name = name
        self.in_node = in_node
        self.result = in_node.result
        print(self.name)

    def run(self):
        tik = time.time()
        #leaky_relu = lambda x: max(0.1*x, x)
        #leaky_relu_layer = np.vectorize(leaky_relu)
        #self.result = leaky_relu_layer(self.in_node.result)
        self.result = np.zeros(self.in_node.result.shape, dtype=np.float32)

        mylib.cublas_leaky_relu_float(
            self.in_node.result.astype(np.float32).ctypes.data_as(POINTER(c_float)),
            self.result.ctypes.data_as(POINTER(c_float)),
            self.result.size)

        tok = time.time() - tik
        print("leaky relu : %.3f" % tok)


# Do not modify below
class Input(DnnNode):
    def __init__(self, name, in_shape):
        self.name = name
        self.in_shape = in_shape 
        self.result = np.ndarray(self.in_shape)

    def set_input(self, tensor):
        assert tuple(self.in_shape) == tuple(tensor.shape)
        self.result = tensor 

    def run(self):
        pass

