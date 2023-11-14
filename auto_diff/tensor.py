import numpy as np
from typing import List, Optional, Union
from .tensor_ops import Op
from . import tensor_ops as ops
from .autograd import compute_gradient_of_variables


class Tensor:
    op: Optional[Op]
    inputs: Optional[List["Tensor"]]
    cached_data: Optional[np.ndarray]
    requires_grad: Optional[bool]=False
    grad: Optional["Tensor"] = None

    def __init__(self, array, *, dtype="float64", requires_grad=True, **kwargs):
        if isinstance(array, Tensor):
            if dtype is None:
                dtype = array.dtype
            cached_data = Tensor._array_from_numpy(array.cached_data, dtype=dtype)
        elif isinstance(array, (np.ndarray, int, float)):
            if dtype is None:
                dtype = array.dtype if isinstance(array, np.ndarray) else type(array)
            cached_data = Tensor._array_from_numpy(array, dtype=dtype)
        elif isinstance(array, list):
            cached_data = Tensor._array_from_numpy(array)
        else:
            raise ValueError("The array is unvalied!")

        self._init(
            op=None, inputs=[], cached_data=cached_data, requires_grad=requires_grad
        )

    def _init(
        self,
        op: Optional[Op],
        inputs: Optional[List["Tensor"]],
        *,
        num_outputs: int = 1,
        cached_data: Optional[np.ndarray] = None,
        requires_grad: Optional[bool] = None,
    ):
        if requires_grad is None and inputs is not None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs if inputs is not None else []
        self.num_outputs = num_outputs
        self.cached_data = cached_data if cached_data is not None else None
        self.requires_grad = requires_grad

    def realized_cached_data(self):
        if self.cached_data is not None:
            return self.cached_data
        elif self.inputs is not None:
            self.cached_data = self.op.compute(
                *[x.realized_cached_data() for x in self.inputs]
            )
        else:
            raise ValueError("The tensor is not valid!")
        
        return self.cached_data

    @staticmethod
    def _array_from_numpy(numpy_array, dtype="float64"):
        return np.array(numpy_array, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Tensor"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op=op, inputs=inputs)
        tensor.realized_cached_data()

        return tensor

    @staticmethod
    def make_const(data: Union[np.ndarray, "Tensor"], requires_grad: bool = False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            op=None,
            inputs=[],
            cached_data=data
            if isinstance(data, np.ndarray)
            else data.realized_cached_data(),
            requires_grad=requires_grad,
        )

    def detach(self):
        return Tensor.make_const(self.realized_cached_data())

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, array: Union[np.ndarray, "Tensor"]):
        if isinstance(array, np.ndarray):
            self.cached_data = array
        elif isinstance(self, Tensor):
            self.cached_data = array.realized_cached_data()
        else:
            raise ValueError("The array type is not valid!")
        
    def numpy(self):
        data = self.realized_cached_data()
        return data

    @property
    def shape(self):
        return self.realized_cached_data().shape

    @property
    def dtype(self):
        return self.realized_cached_data().dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def is_leaf(self):
        return self.op is None

    def __repr__(self) -> str:
        return f"Tensor({self.realized_cached_data()})"

    def __str__(self) -> str:
        return self.realized_cached_data().__str__()

    def backward(self, out_grad: Optional["Tensor"] = None):
        out_grad = out_grad if out_grad else Tensor(np.ones(self.shape))
        compute_gradient_of_variables(self, out_grad)

    def __add__(self, other):
        if isinstance(other, Tensor):
            return ops.EWiseAdd()(self, other)
        elif isinstance(other, (int, float)):
            return ops.AddScalar(other)(self)
        
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return ops.EWiseMul()(self, other)
        else:
            return ops.MulScalar(other)(self)
    
    def __pow__(self, other):
        return ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return ops.EWiseAdd()(self, ops.Negative()(other))
        else:
            return ops.AddScalar(-other)(self)
        
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return ops.EWiseDiv()(self, other)
        else:
            return ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return ops.MatMul()(self, other)

    def matmul(self, other):
        return ops.MatMul()(self, other)

    def sum(self, axes=None):
        return ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return ops.Reshape(shape)(self)

    def __neg__(self):
        return ops.Negative()(self)

    def transpose(self, axes=None):
        return ops.Transpose(axes)(self)
    
    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__