from ctypes import *

mylib = cdll.LoadLibrary('./libtestctypes.so')

a = c_float(24)
b = c_float(4.5)
c = c_float(0)

mylib.add_float_p(a, b, byref(c))
print('(1) {} + {} = {}'.format(a.value, b.value, c.value))

# Specify argtypes
mylib.add_float_p.argtypes = [c_float, c_float, POINTER(c_float)]
mylib.add_float_p(a, b, c)
print('(2) {} + {} = {}'.format(a.value, b.value, c.value))
