from typing import List, Tuple, Union, Optional
import numpy as np


class Op:
    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args: List[np.ndarray]) -> np.ndarray:
        """计算前向运算
        Parameter:
            *args: 参与前向运算计算的数组

        Return:
            output: 前向传播计算好的结果
        """
        raise NotImplementedError()

    def gradient(
        self, out_grad: "Tensor", node: "Tensor"
    ) -> Union["Tensor", Tuple["Tensor"]]:
        """计算反向传播梯度计算
        Parameters:
            out_grad: 从后面节点传到当前节点的梯度
            node: 前向传播的节点, 因为使用扩展计算图, 所以需要原节点

        Return:
            对节点node的输入节点的偏梯度
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: "Tensor", node: "Tensor") -> Tuple["Tensor"]:
        """为了后续统一, 将返回的梯度都转为元组类型"""
        output = self.gradient(out_grad, node)
        if isinstance(output, Tuple):
            return output
        else:
            return (output,)


class TensorOp(Op):
    def __call__(self, *args):
        """保证了直接调用运算操作的时候, 是创建一个新的Tensor对象，且该对象里面包括了具体操作"""
        from .tensor import Tensor

        return Tensor.make_from_op(self, args)


class EWiseAdd(TensorOp):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a + b

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        return out_grad, out_grad


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: np.ndarray):
        return a + self.scalar

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        return out_grad


class EWiseMul(TensorOp):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a * b

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        lhs, rhs = node.inputs
        return EWiseMul()(out_grad, rhs), EWiseMul()(out_grad, lhs)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: np.ndarray):
        return a * self.scalar

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        return MulScalar(self.scalar)(out_grad)


class PowerScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: np.ndarray):
        return np.power(a, self.scalar)

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        return EWiseMul()(
            out_grad,
            MulScalar(self.scalar)(PowerScalar(self.scalar - 1)(node.inputs[0])),
        )


class Negative(TensorOp):
    def compute(self, a: np.ndarray):
        return -a

    def gradient(self, out_grad: "Tensor", node: "Tnesor"):
        return Negative()(out_grad)


class EWiseDiv(TensorOp):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a / b

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        lhs, rhs = node.inputs
        return EWiseDiv()(out_grad, rhs), Negative()(
            EWiseDiv()(EWiseMul()(out_grad, lhs), PowerScalar(2)(rhs))
        )


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: np.ndarray):
        return a / self.scalar

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        return DivScalar(self.scalar)(out_grad)


class Exp(TensorOp):
    def compute(self, a: np.ndarray):
        return np.exp(a)

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        return EWiseMul()(out_grad, Exp()(node.inputs[0]))


class Transpose(TensorOp):
    def __init__(self, axis: Optional[Tuple[int]] = None):
        if axis is not None:
            assert len(axis) == 2, ValueError("The axis must be a tuple of length 2!")
        self.axis = axis

    def compute(self, a: np.ndarray):
        axis_ = [i for i in range(a.ndim)]
        if self.axis is None:
            axis_[-1], axis_[-2] = axis_[-2], axis_[-1]
        else:
            axis_[self.axis[0]], axis_[self.axis[1]] = (
                axis_[self.axis[1]],
                axis_[self.axis[0]],
            )

        return np.transpose(a, axis_)

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        return Transpose(self.axis)(out_grad)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: np.ndarray):
        return np.reshape(a, self.shape)

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        ori_shape = node.inputs[0].shape
        return Reshape(ori_shape)(out_grad)


class Summation(TensorOp):
    def __init__(self, axis: Optional[Tuple[int]] = None) -> None:
        self.axis = axis

    def compute(self, a: np.ndarray):
        return np.sum(a, axis=self.axis)

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> "Tensor":
        ori_shape = node.inputs[0].shape
        if self.axis: 
            temp_shape = list(ori_shape)
            for i in self.axis:
                temp_shape[i] = 1
        else:
            temp_shape = [1] * len(ori_shape)

        out_grad = Reshape(temp_shape)(out_grad)
        return BroadcastTo(ori_shape)(out_grad)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: np.ndarray) -> np.ndarray:
        return np.broadcast_to(a, self.shape)

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> "Tensor":
        ori_shape = node.inputs[0].shape
        # 找到是如何boradcaste的，然后对扩张的轴求sum，使之压缩
        # 这是ndim的broadcast
        axis_ = list(range(len(self.shape) - len(ori_shape)))
        # 然后求shape的扩张
        for i in range(-1, -len(ori_shape) - 1, -1):
            if ori_shape[i] != self.shape[i]:
                assert ori_shape[i] == 1
                axis_.append(i)

        return Reshape(ori_shape)(Summation(axis=tuple(axis_))(out_grad))


class MatMul(TensorOp):
    def compute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b)

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> Tuple["Tensor"]:
        lhs, rhs = node.inputs
        l_grad = MatMul()(out_grad, Transpose()(rhs))
        r_grad = MatMul()(Transpose()(lhs), out_grad)

        if l_grad.ndim > lhs.ndim:
            l_grad = Summation(axis=tuple(list(range(l_grad.ndim - lhs.ndim))))(
                l_grad
            )

        if r_grad.ndim > rhs.ndim:
            r_grad = Summation(axis=tuple(list(range(r_grad.ndim - rhs.ndim))))(
                r_grad
            )

        return l_grad, r_grad


class Log(TensorOp):
    def compute(self, a: np.ndarray) -> np.ndarray:
        return np.log(a)

    def gradient(self, out_grad: "Tensor", node: "Tensor") -> "Tensor":
        return EWiseDiv()(out_grad, node.inputs[0])


class ReLU(TensorOp):
    def compute(self, a: np.ndarray) -> np.ndarray:
        self.mask = (a > 0).astype("int8")
        return np.maximum(a, 0)

    def gradient(self, out_grad: "Tensor", node: "Tensor"):
        return MulScalar(self.mask)(out_grad)
