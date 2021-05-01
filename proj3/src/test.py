import unittest
import tensorflow as tf
import numpy as np
import os

from dnn import (
    Conv2D,
    BiasAdd,
    MaxPool2D,
    BatchNorm,
    LeakyReLU,
    DnnGraphBuilder,
    DnnInferenceEngine,
)


class TestConv2D(unittest.TestCase):
    def setUp(self):
        self.tf_g = tf.Graph()
        self.tf_sess = tf.compat.v1.Session(graph=self.tf_g)

        self.impl_g = DnnGraphBuilder()
        self.impl_sess = DnnInferenceEngine(self.impl_g, False)

        in_shape = (1, 416, 416, 3)
        with self.tf_g.as_default():
            self.tf_inp = tf.compat.v1.placeholder(
                tf.float32, shape=in_shape, name="input"
            )
        self.impl_inp = self.impl_g.create_input(in_shape)

        self.im = np.random.rand(1, 416, 416, 3)
        self.filter = np.random.rand(3, 3, 3, 16)

    def test_padding_same(self):
        with self.tf_g.as_default():
            tf_conv2d = tf.nn.conv2d(
                self.tf_inp, self.filter, strides=[1, 1, 1, 1], padding="SAME"
            )

        impl_conv2d = self.impl_g.create_conv2d(
            self.impl_inp, self.filter, strides=[1, 1, 1, 1], padding="SAME"
        )
        self.impl_g.set_out_node(impl_conv2d)

        feed_dict = {self.tf_inp: self.im}
        tf_out = self.tf_sess.run([self.tf_inp, tf_conv2d], feed_dict)[-1]
        impl_out = self.impl_sess.run(self.im)

        self.assertEqual(tf_out.shape, impl_out.shape)
        diff_mean = np.absolute(tf_out - impl_out).mean()
        self.assertTrue((diff_mean < 0.01).all())

    def test_padding_valid(self):
        with self.tf_g.as_default():
            tf_conv2d = tf.nn.conv2d(
                self.tf_inp, self.filter, strides=[1, 1, 1, 1], padding="VALID"
            )

        impl_conv2d = self.impl_g.create_conv2d(
            self.impl_inp, self.filter, strides=[1, 1, 1, 1], padding="VALID"
        )
        self.impl_g.set_out_node(impl_conv2d)

        feed_dict = {self.tf_inp: self.im}
        tf_out = self.tf_sess.run([self.tf_inp, tf_conv2d], feed_dict)[-1]
        impl_out = self.impl_sess.run(self.im)

        self.assertEqual(tf_out.shape, impl_out.shape)
        diff_mean = np.absolute(tf_out - impl_out).mean()
        self.assertTrue((diff_mean < 0.01).all())

    @unittest.expectedFailure
    def test_invalid_pad(self):
        impl_conv2d = self.impl_g.create_conv2d(
            self.impl_inp, self.filter, strides=[1, 1, 1, 1], padding="FAIL"
        )

    @unittest.expectedFailure
    def test_invalid_dim(self):
        impl_conv2d = self.impl_g.create_conv2d(
            self.impl_inp,
            np.ones(
                47,
            ),
            strides=[1, 1, 1, 1],
            padding="VALID",
        )


class TestMaxpool(unittest.TestCase):
    def setUp(self):
        self.tf_g = tf.Graph()
        self.tf_sess = tf.compat.v1.Session(graph=self.tf_g)

        self.impl_g = DnnGraphBuilder()
        self.impl_sess = DnnInferenceEngine(self.impl_g, False)

        in_shape = (1, 416, 416, 3)
        with self.tf_g.as_default():
            self.tf_inp = tf.compat.v1.placeholder(
                tf.float32, shape=in_shape, name="input"
            )
        self.impl_inp = self.impl_g.create_input(in_shape)

        self.im = np.random.rand(1, 416, 416, 3)

    def test_padding_same(self):
        with self.tf_g.as_default():
            tf_maxpool = tf.nn.max_pool2d(
                self.tf_inp, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
            )

        impl_maxpool = self.impl_g.create_max_pool2d(
            self.impl_inp, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )
        self.impl_g.set_out_node(impl_maxpool)

        feed_dict = {self.tf_inp: self.im}
        tf_out = self.tf_sess.run([self.tf_inp, tf_maxpool], feed_dict)[-1]
        impl_out = self.impl_sess.run(self.im)

        self.assertEqual(tf_out.shape, impl_out.shape)
        diff_mean = np.absolute(tf_out - impl_out).mean()
        self.assertTrue((diff_mean < 0.01).all())

    def test_padding_valid(self):
        with self.tf_g.as_default():
            tf_maxpool = tf.nn.max_pool2d(
                self.tf_inp, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID"
            )

        impl_maxpool = self.impl_g.create_max_pool2d(
            self.impl_inp, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID"
        )
        self.impl_g.set_out_node(impl_maxpool)

        feed_dict = {self.tf_inp: self.im}
        tf_out = self.tf_sess.run([self.tf_inp, tf_maxpool], feed_dict)[-1]
        impl_out = self.impl_sess.run(self.im)

        self.assertEqual(tf_out.shape, impl_out.shape)
        diff_mean = np.absolute(tf_out - impl_out).mean()
        self.assertTrue((diff_mean < 0.01).all())


class TestMaxpool(unittest.TestCase):
    def setUp(self):
        self.tf_g = tf.Graph()
        self.tf_sess = tf.compat.v1.Session(graph=self.tf_g)

        self.impl_g = DnnGraphBuilder()
        self.impl_sess = DnnInferenceEngine(self.impl_g, False)

        in_shape = (1, 416, 416, 16)
        with self.tf_g.as_default():
            self.tf_inp = tf.compat.v1.placeholder(
                tf.float32, shape=in_shape, name="input"
            )
        self.impl_inp = self.impl_g.create_input(in_shape)

        self.im = np.random.rand(*in_shape)
        self.mean = np.random.rand(16,)
        self.var = np.random.rand(16,)
        self.gamma = np.random.rand(16,)

    def test_simple(self):
        with self.tf_g.as_default():
            tf_bn = tf.nn.batch_normalization(
                self.tf_inp,
                mean=self.mean,
                variance=self.var,
                offset=0,
                scale=self.gamma,
                variance_epsilon=1e-5,
            )

        impl_bn = self.impl_g.create_batch_norm(
            self.impl_inp, self.mean, self.var, self.gamma, 1e-5
        )
        self.impl_g.set_out_node(impl_bn)

        feed_dict = {self.tf_inp: self.im}
        tf_out = self.tf_sess.run([self.tf_inp, tf_bn], feed_dict)[-1]
        impl_out = self.impl_sess.run(self.im)

        self.assertEqual(tf_out.shape, impl_out.shape)
        diff_mean = np.absolute(tf_out - impl_out).mean()
        self.assertTrue((diff_mean < 0.01).all())


class TestBiasAdd(unittest.TestCase):
    def setUp(self):
        self.tf_g = tf.Graph()
        self.tf_sess = tf.compat.v1.Session(graph=self.tf_g)

        self.impl_g = DnnGraphBuilder()
        self.impl_sess = DnnInferenceEngine(self.impl_g, False)

        in_shape = (1, 416, 416, 16)
        with self.tf_g.as_default():
            self.tf_inp = tf.compat.v1.placeholder(
                tf.float32, shape=in_shape, name="input"
            )
        self.impl_inp = self.impl_g.create_input(in_shape)

        self.im = np.random.rand(*in_shape)
        self.bias = np.random.rand(
            16,
        )

    def test_simple(self):
        with self.tf_g.as_default():
            tf_ba = tf.nn.bias_add(self.tf_inp, self.bias)

        impl_ba = self.impl_g.create_bias_add(self.impl_inp, self.bias)
        self.impl_g.set_out_node(impl_ba)

        feed_dict = {self.tf_inp: self.im}
        tf_out = self.tf_sess.run([self.tf_inp, tf_ba], feed_dict)[-1]
        impl_out = self.impl_sess.run(self.im)

        self.assertEqual(tf_out.shape, impl_out.shape)
        diff_mean = np.absolute(tf_out - impl_out).mean()
        self.assertTrue((diff_mean < 0.01).all())


if __name__ == "__main__":
    unittest.main()
