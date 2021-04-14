import os
import sys
import math
import networkx as nx
import numpy as np
from itertools import product
from multiprocessing import Process, sharedctypes

#parallelism = 8
parallelism = os.cpu_count() - 1


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
        #raise NotImplementedError('Conv2d is not implemented yet')
        assert in_node.shape[3] == kernel.shape[2]
        self.name = name
        self.in_node = in_node
        self.kernel = kernel
        self.strides = self.get_strides(strides)
        self.is_pad_same = self.is_pad_same(padding)
        self.shape = (in_node.shape[0], in_node.shape[1], in_node.shape[2], kernel.shape[3])
        print(self.name)

    def get_strides(self, strides):
        num = len(strides)
        if num == 4:
            return [strides[1], strides[2]]
        elif num == 2:
            return strides
        elif num == 1:
            return [strides[0], strides[0]]
        else:
            raise Exception('The length of strides option should be 1 or 2 or 4')

    def is_pad_same(self, padding):
        if padding == 'SAME':
            return True
        elif padding == 'VALID':
            return False
        else:
            raise NotImplementedError('The padding option should be SAME or VALID')

    def get_output_size(self, in_size, kernel_size, stride_size):
        if self.is_pad_same:
            return np.ceil(in_size / stride_size)
        else:
            return np.ceil((in_size - kernel_size + 1) / stride_size)

    def get_padding_size(self, in_size, kernel_size, stride_size, out_size):
        if self.is_pad_same:
            return max((out_size - 1) * stride_size + kernel_size - in_size, 0)
        else:
            return 0

    def mul(self, x, w, start, step):
        arr = np.frombuffer(self.arr.get_obj())
        c = arr.reshape(x.shape[0], w.shape[1])
        for i in range(start, start + step):
            for j in range(w.shape[1]):
                for k in range(x.shape[1]):
                    c[i, j] += x[i, k] * w[k, j]

    def run(self, counter):
        in_n, in_h, in_w, in_c = self.in_node.shape
        kernel_h, kernel_w, kernel_i, kernel_o = self.kernel.shape
        stride_h, stride_w = self.strides

        out_h = int(self.get_output_size(in_h, kernel_h, stride_h))
        out_w = int(self.get_output_size(in_w, kernel_w, stride_w))

        pad_h = int(self.get_padding_size(in_h, kernel_h, stride_h, out_h))
        pad_w = int(self.get_padding_size(in_w, kernel_w, stride_w, out_w))

        # UPPER_SAME
        pad_h_upper = int(np.floor(pad_h / 2))
        pad_h_lower = int(np.ceil(pad_h / 2))
        pad_w_upper = int(np.floor(pad_w / 2))
        pad_w_lower = int(np.ceil(pad_w / 2))
        pad = ((0, 0), (pad_h_upper, pad_h_lower), (pad_w_upper, pad_w_lower), (0, 0))

        in_pad = np.pad(self.in_node.result, pad)

        x = np.zeros([in_n, out_h, out_w, kernel_h, kernel_w, in_c])
        for i in range(out_h):
            ih = i * stride_h
            for j in range(out_w):
                jw = j * stride_w
                x[:, i, j, :, :, :] = in_pad[:, ih:ih + kernel_h, jw:jw + kernel_w, :]

        x_shape = x.shape
        x = x.reshape([x_shape[0] * x_shape[1] * x_shape[2], -1])

        w = self.kernel.reshape([kernel_h * kernel_w * kernel_i, kernel_o])

        #self.result = x.dot(w)

        #self.result = np.zeros([x.shape[0], w.shape[1]])
        #for i in range(x.shape[0]):
        #    for j in range(w.shape[1]):
        #        for k in range(x.shape[1]):
        #            self.result[i, j] += x[i, k] * w[k, j]

        self.arr = sharedctypes.Array('d', x.shape[0] * w.shape[1])
        step = int(x.shape[0] / parallelism)
        procs =[]

        for i in range(0, parallelism):
            proc = Process(target=self.mul, args=(x, w, step * i, step))
            procs.append(proc)
            proc.start()

        remain = x.shape[0] % parallelism
        if remain != 0:
            proc = Process(target=self.mul, args=(x, w, step * parallelism, remain))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        self.result = np.frombuffer(self.arr.get_obj())
        self.result = self.result.reshape([x_shape[0], x_shape[1], x_shape[2], -1])


class BiasAdd(DnnNode):
    def __init__(self, name, in_node, biases):
        #raise NotImplementedError('BiasAdd is not implemented yet')
        assert all((m == n) or (m == 1) or (n == 1) for m, n in zip(in_node.shape[::-1], biases.shape[::-1]))
        self.name = name
        self.in_node = in_node
        self.biases = biases
        self.shape = in_node.shape
        print(self.name)

    def add(self, x, b, start, step):
        arr = np.frombuffer(self.arr.get_obj())
        c = arr.reshape(x.shape)
        for i in range(start, start + step):
            for j in range(x.shape[1]):
                c[i, j] = x[i, j] + b[j]

    def run(self, counter):
        #self.result = self.in_node.result + self.biases
        in_shape = self.in_node.shape
        x = self.in_node.result.reshape([-1, in_shape[3]])

        #self.result = np.zeros(x.shape)
        #for i in range(x.shape[0]):
        #    for j in range(x.shape[1]):
        #        self.result[i, j] = x[i, j] + self.biases[j]

        self.arr = sharedctypes.Array('d', x.shape[0] * x.shape[1])
        step = int(x.shape[0] / parallelism)
        procs =[]

        for i in range(0, parallelism):
            proc = Process(target=self.add, args=(x, self.biases, step * i, step))
            procs.append(proc)
            proc.start()

        remain = x.shape[0] % parallelism
        if remain != 0:
            proc = Process(target=self.add, args=(x, self.biases, step * parallelism, remain))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        self.result = np.frombuffer(self.arr.get_obj())
        self.result = self.result.reshape(in_shape)


class MaxPool2D(DnnNode):
    def __init__(self, name, in_node, ksize, strides, padding):
        #raise NotImplementedError('MaxPool2D is not implemented yet')
        assert in_node.shape[0] > 0
        self.name = name
        self.in_node = in_node
        self.ksize = self.get_ksize(ksize)
        self.strides = self.get_strides(strides)
        self.is_pad_same = self.is_pad_same(padding)
        self.shape = (in_node.shape[0], int(np.ceil(in_node.shape[1]/self.strides[0])), int(np.ceil(in_node.shape[2]/self.strides[1])), in_node.shape[3])
        print(self.name)

    def get_ksize(self, ksize):
        num = len(ksize)
        if num == 4:
            return [ksize[1], ksize[2]]
        elif num == 2:
            return ksize
        elif num == 1:
            return [ksize[0], ksize[0]]
        else:
            raise Exception('The length of ksize option should be 1 or 2 or 4')

    def get_strides(self, strides):
        num = len(strides)
        if num == 4:
            return [strides[1], strides[2]]
        elif num == 2:
            return strides
        elif num == 1:
            return [strides[0], strides[0]]
        else:
            raise Exception('The length of strides option should be 1 or 2 or 4')

    def is_pad_same(self, padding):
        if padding == 'SAME':
            return True
        elif padding == 'VALID':
            return False
        else:
            raise NotImplementedError('The padding option should be SAME or VALID')

    def get_output_size(self, in_size, kernel_size, stride_size):
        if stride_size == 1:
            return in_size
        else:
            return 1 + (in_size - kernel_size) / stride_size

    def get_padding_size(self, in_size, kernel_size, stride_size, out_size):
        if self.is_pad_same:
            return max((out_size - 1) * stride_size + kernel_size - in_size, 0)
        else:
            return 0

    def max(self, x, start, step):
        arr = np.frombuffer(self.arr.get_obj())
        c = arr.reshape(x.shape[0], x.shape[2])
        for i in range(start, start + step):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    if c[i, k] == 0:
                        c[i, k] = x[i, j, k]
                    else:
                        c[i, k] = max(c[i, k], x[i, j, k])

    def run(self, counter):
        in_n, in_h, in_w, in_c = self.in_node.shape
        kernel_h, kernel_w = self.ksize
        stride_h, stride_w = self.strides

        out_h = int(self.get_output_size(in_h, kernel_h, stride_h))
        out_w = int(self.get_output_size(in_w, kernel_w, stride_w))

        pad_h = int(self.get_padding_size(in_h, kernel_h, stride_h, out_h))
        pad_w = int(self.get_padding_size(in_w, kernel_w, stride_w, out_w))

        # UPPER_SAME
        pad_h_upper = int(np.floor(pad_h / 2))
        pad_h_lower = int(np.ceil(pad_h / 2))
        pad_w_upper = int(np.floor(pad_w / 2))
        pad_w_lower = int(np.ceil(pad_w / 2))
        pad = ((0, 0), (pad_h_upper, pad_h_lower), (pad_w_upper, pad_w_lower), (0, 0))

        in_pad = np.pad(self.in_node.result, pad)

        x = np.zeros([in_n, out_h, out_w, kernel_h, kernel_w, in_c])
        for i in range(out_h):
            ih = i * stride_h
            for j in range(out_w):
                jw = j * stride_w
                x[:, i, j, :, :, :] = in_pad[:, ih:ih + kernel_h, jw:jw + kernel_w, :]

        x_shape = x.shape
        x = x.reshape([x_shape[0] * x_shape[1] * x_shape[2], x_shape[3] * x_shape[4], -1])

        #self.result = np.max(x, axis=1)

        #self.result = np.zeros([x.shape[0], x.shape[2]])
        #for i in range(x.shape[0]):
        #    for j in range(x.shape[1]):
        #        for k in range(x.shape[2]):
        #            if self.result[i, k] == 0:
        #                self.result[i, k] = x[i, j, k]
        #            else:
        #                self.result[i, k] = max(self.result[i, k], x[i, j, k])

        self.arr = sharedctypes.Array('d', x.shape[0] * x.shape[2])
        step = int(x.shape[0] / parallelism)
        procs =[]

        for i in range(0, parallelism):
            proc = Process(target=self.max, args=(x, step * i, step))
            procs.append(proc)
            proc.start()

        remain = x.shape[0] % parallelism
        if remain != 0:
            proc = Process(target=self.max, args=(x, step * parallelism, remain))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        self.result = np.frombuffer(self.arr.get_obj())
        self.result = self.result.reshape([x_shape[0], x_shape[1], x_shape[2], -1])


class BatchNorm(DnnNode):
    def __init__(self, name, in_node, mean, variance, gamma, epsilon):
        #raise NotImplementedError('BatchNorm is not implemented yet')
        assert in_node.shape[3] == len(mean) == len(variance) == len(gamma)
        self.name = name
        self.in_node = in_node
        self.mean = mean
        self.variance = variance
        self.gamma = gamma
        self.epsilon = epsilon
        self.shape = in_node.shape
        print(self.name)

    def norm(self, x, gamma, mean, var, epsilon, start, step):
        arr = np.frombuffer(self.arr.get_obj())
        c = arr.reshape(x.shape[0], x.shape[1])
        for i in range(start, start + step):
            for j in range(x.shape[1]):
                c[i, j] = gamma[j] * (x[i, j] - mean[j]) / np.sqrt(var[j] + epsilon)

    def run(self, counter):
        in_shape = self.in_node.shape
        x = self.in_node.result.reshape([-1, in_shape[3]])

        #self.result = np.zeros(x.shape)
        #for i in range(x.shape[0]):
        #    for j in range(x.shape[1]):
        #        self.result[i, j] = self.gamma[j] * (x[i, j] - self.mean[j]) / np.sqrt(self.variance[j] + self.epsilon)

        self.arr = sharedctypes.Array('d', x.shape[0] * x.shape[1])
        step = int(x.shape[0] / parallelism)
        procs =[]

        for i in range(0, parallelism):
            proc = Process(target=self.norm, args=(x, self.gamma, self.mean, self.variance, self.epsilon, step * i, step))
            procs.append(proc)
            proc.start()

        remain = x.shape[0] % parallelism
        if remain != 0:
            proc = Process(target=self.norm, args=(x, self.gamma, self.mean, self.variance, self.epsilon, step * parallelism, remain))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        self.result = np.frombuffer(self.arr.get_obj())
        self.result = self.result.reshape(in_shape)


class LeakyReLU(DnnNode):
    def __init__(self, name, in_node):
        #raise NotImplementedError('LeakyReLU is not implemented yet')
        assert in_node.shape[0] > 0
        self.name = name
        self.in_node = in_node
        self.alpha = 0.10000000149011612
        self.shape = in_node.shape
        print(self.name)

    def leaky_relu(self, x, alpha, start, step):
        arr = np.frombuffer(self.arr.get_obj())
        c = arr.reshape(x.shape[0])
        for i in range(start, start + step):
            c[i] = max(alpha * x[i], x[i])

    def run(self, counter):
        in_shape = self.in_node.shape
        x = self.in_node.result.reshape(-1)

        #self.result = np.zeros(x.shape)
        #for i in range(x.shape[0]):
        #    self.result[i] = max(self.alpha * x[i], x[i])

        self.arr = sharedctypes.Array('d', x.shape[0])
        step = int(x.shape[0] / parallelism)
        procs =[]

        for i in range(0, parallelism):
            proc = Process(target=self.leaky_relu, args=(x, self.alpha, step * i, step))
            procs.append(proc)
            proc.start()

        remain = x.shape[0] % parallelism
        if remain != 0:
            proc = Process(target=self.leaky_relu, args=(x, self.alpha, step * parallelism, remain))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        self.result = np.frombuffer(self.arr.get_obj())
        self.result = self.result.reshape(in_shape)


class Input(DnnNode):
    def __init__(self, name, in_shape):
        self.name = name
        self.in_shape = in_shape
        self.result = np.ndarray(self.in_shape)
        self.shape = self.in_shape
        print(self.name)

    def set_input(self, tensor):
        assert tuple(self.in_shape) == tuple(tensor.shape)
        self.result = tensor

    def run(self, counter):
        pass
