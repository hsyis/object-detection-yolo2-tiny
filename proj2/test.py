import unittest
import tensorflow as tf
import numpy as np

from dnn import Conv2D, BiasAdd, MaxPool2D, BatchNorm, LeakyReLU, \
    DnnGraphBuilder, DnnInferenceEngine


class TestConv2D(unittest.TestCase):
    def setUp(self):
        self.tf_g = tf.Graph()
        self.tf_sess = tf.compat.v1.Session(graph=self.tf_g)

        self.impl_g = DnnGraphBuilder()
        self.impl_sess = DnnInferenceEngine(self.impl_g)

        in_shape = (1, 416, 416, 3)
        with self.tf_g.as_default():
            self.tf_inp = tf.compat.v1.placeholder(
                tf.float32, shape=in_shape, name="input")
        self.impl_inp = self.impl_g.create_input(in_shape)

        self.im = np.random.rand(1, 416, 416, 3)
        self.filter = np.random.rand(3, 3, 3, 16)

    def test_padding_same(self):
        with self.tf_g.as_default():
            tf_conv2d = tf.nn.conv2d(
                self.tf_inp,
                self.filter,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )

        impl_conv2d = self.impl_g.create_conv2d(
            self.impl_inp,
            self.filter,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        self.impl_g.set_out_node(impl_conv2d)

        feed_dict = {self.tf_inp: self.im}
        tf_out = self.tf_sess.run([self.tf_inp, tf_conv2d], feed_dict)[-1]
        impl_out = self.impl_sess.run(self.im)

        self.assertEqual(tf_out.shape, impl_out.shape)
        diff_mean = np.absolute(tf_out - impl_out).mean()
        self.assertTrue((diff_mean < 0.001).all())

    def test_padding_valid(self):
        with self.tf_g.as_default():
            tf_conv2d = tf.nn.conv2d(
                self.tf_inp,
                self.filter,
                strides=[1, 1, 1, 1],
                padding='VALID'
            )

        impl_conv2d = self.impl_g.create_conv2d(
            self.impl_inp,
            self.filter,
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        self.impl_g.set_out_node(impl_conv2d)

        feed_dict = {self.tf_inp: self.im}
        tf_out = self.tf_sess.run([self.tf_inp, tf_conv2d], feed_dict)[-1]
        impl_out = self.impl_sess.run(self.im)

        self.assertEqual(tf_out.shape, impl_out.shape)
        diff_mean = np.absolute(tf_out - impl_out).mean()
        self.assertTrue((diff_mean < 0.001).all())

    @unittest.expectedFailure
    def test_fail(self):
        impl_conv2d = self.impl_g.create_conv2d(
            self.impl_inp,
            self.filter,
            strides=[1, 1, 1, 1],
            padding='FAIL'
        )

    @unittest.expectedFailure
    def test_dim_fail(self):
        invalid_dim_conv2d = self.impl_g.create_conv2d(
            self.impl_inp,
            np.random.rand(3, 3, 1, 16),
            strides=[1, 1, 1, 1],
            padding='SAME'
        )


if __name__ == '__main__':
    unittest.main()
