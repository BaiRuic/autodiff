from . import tensor_ops as ops

def add(a, b):
    return ops.EWiseAdd()(a, b)

def add_scalar(a, scalar):
    return ops.AddScalar(scalar)(a)

def multiply(a, b):
    return ops.EWiseMul()(a, b)


def mul_scalar(a, scalar):
    return ops.MulScalar(scalar)(a)


def power_scalar(a, scalar):
    return ops.PowerScalar(scalar)(a)


def divide(a, b):
    return ops.EWiseDiv()(a, b)


def divide_scalar(a, scalar):
    return ops.DivScalar(scalar)(a)


def transpose(a, axis=None):
    return ops.Transpose(axis)(a)


def reshape(a, shape):
    return ops.Reshape(shape)(a)


def broadcast_to(a, shape):
    return ops.BroadcastTo(shape)(a)


def summation(a, axis=None):
    return ops.Summation(axis)(a)

def matmul(a, b):
    return ops.MatMul()(a, b)


def negate(a):
    return ops.Negative()(a)

def log(a):
    return ops.Log()(a)


def exp(a):
    return ops.Exp()(a)



def relu(a):
    return ops.ReLU()(a)