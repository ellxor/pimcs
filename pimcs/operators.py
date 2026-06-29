from __future__ import annotations
from dataclasses import dataclass
from typing import Union
from collections import defaultdict
from collections.abc import Callable
from enum import Enum, auto
from numpy import complex64, isclose


class PIOperatorKind(Enum):
    Jz = auto()
    Jp = auto()
    Jm = auto()
    A  = auto()
    Ad = auto()
    Ap = auto()
    As = auto()

class BinaryOperatorKind(Enum):
    Add = auto()
    Mul = auto()


class PIExpression:
    def __add__(self, other):  return BinOp(BinaryOperatorKind.Add, self,  other)
    def __mul__(self, other):  return BinOp(BinaryOperatorKind.Mul, self,  other)
    def __sub__(self, other):  return BinOp(BinaryOperatorKind.Add, self, -other)
    def __neg__(self):         return BinOp(BinaryOperatorKind.Mul, self, coeff(-1))
    def __rmul__(self, other): return BinOp(BinaryOperatorKind.Mul, self, TimeDependent(other, False) if callable(other) else coeff(other))

    def __pow__(self, n: int):
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"Exponent must be a non-negative integer, got {n!r}")

        return Leaf(1) if n == 0 else self * (self ** (n - 1))


    def dag(self):
        match self:
            case Leaf(value = PIOperatorKind()): 
                match self.value:
                    case PIOperatorKind.Jz: return self
                    case PIOperatorKind.Jp: return Leaf(PIOperatorKind.Jm, self.dim)
                    case PIOperatorKind.Jm: return Leaf(PIOperatorKind.Jp, self.dim)
                    case PIOperatorKind.A:  return Leaf(PIOperatorKind.Ad, self.dim)
                    case PIOperatorKind.Ad: return Leaf(PIOperatorKind.A, self.dim)
            case Leaf():
                return coeff(self.value.conjugate())
            case TimeDependent():
                return TimeDependent(self.func, not self.conj)
            case BinOp():
                return BinOp(self.kind, self.right.dag(), self.left.dag())


    def is_herm(self):
        return all(isclose(coeff, 0) for coeff, *_ in to_sum_of_products(self - self.dag(), 0))


    def displace(self):
        match self:
            case Leaf(PIOperatorKind.A):  return BinOp(BinaryOperatorKind.Add, self, Leaf(PIOperatorKind.Ap, None))
            case Leaf(PIOperatorKind.Ad): return BinOp(BinaryOperatorKind.Add, self, Leaf(PIOperatorKind.As, None))
            case BinOp(): return BinOp(self.kind, self.left.displace(), self.right.displace())
            case _:       return self


    _OP_NAMES = {
        PIOperatorKind.Jz: 'Jz',
        PIOperatorKind.Jp: 'J+',
        PIOperatorKind.Jm: 'J-',
        PIOperatorKind.A:  'a',
        PIOperatorKind.Ad: 'a†',
    }

    def __str__(self):
        match self:
            case Leaf(value=PIOperatorKind() as o):
                return self._OP_NAMES[o]
            case Leaf():
                return str(self.value)
            case BinOp(kind = BinaryOperatorKind.Add):
                return f"({str(self.left)} + {str(self.right)})"
            case BinOp(kind = BinaryOperatorKind.Mul):
                return f"({str(self.left)} * {str(self.right)})"
            case TimeDependent():
                return f"conjf({self.func})" if self.conj else f"({self.func})"

@dataclass
class Leaf(PIExpression):
    value: complex | PIOperatorKind
    dim: int | None

@dataclass
class BinOp(PIExpression):
    kind:  BinaryOperatorKind
    left:  Expr
    right: Expr

@dataclass
class TimeDependent(PIExpression):
    func: Callable
    conj: bool

def coeff(value) -> Leaf:
    return Leaf(complex64(value), None)

PIQobj = Union[Leaf, BinOp]


# Constructor functions: TODO: add number to these operators

def destroy(truncation: int) -> PIQobj:
    return Leaf(PIOperatorKind.A, truncation)


def jspin(N: int, op = None) -> tuple[PIQobj, ...]:
    valid = "xyz+-"

    if op is not None and (not isinstance(op, str) or op not in valid):
        list_of_options = ", ".join(f"'{v}'" for v in valid)
        raise ValueError(f"Operator must be one of: {{{list_of_options}}}")

    jz = Leaf(PIOperatorKind.Jz, N)
    jp = Leaf(PIOperatorKind.Jp, N)
    jm = Leaf(PIOperatorKind.Jm, N)

    jx =  0.5  * (jp + jm)
    jy = -0.5j * (jp - jm)

    match op:
        case None: return jx, jy, jz
        case 'x':  return jx
        case 'y':  return jy
        case 'z':  return jz
        case '+':  return jp
        case '-':  return jm


def validate_pair(a: int | None, b: int | None) -> int | None:
    if a is None: return b
    if b is None: return a

    if a != b:
        raise ValueError(f"Dimensions do not match, got {a} and {b}")

    return a


def validate_dimension(expr: PIQobj) -> tuple[int, int]:
    match expr:
        case Leaf():
            match expr.value:
                case PIOperatorKind.Jz | PIOperatorKind.Jp | PIOperatorKind.Jm:
                    return expr.dim, None
                case PIOperatorKind.A | PIOperatorKind.Ad:
                    return None, expr.dim
                case _:
                    return None, None

        case BinOp():
            ls, lb = validate_dimension(expr.left)
            rs, rb = validate_dimension(expr.right)
            ls, lb = validate_dimension(expr.left)
            return validate_pair(ls, rs), validate_pair(lb, rb)

        case TimeDependent():
            return None, None


def expand(expr: PIQobj) -> list[list]:
    match expr:
        case Leaf(): return [[expr.value]]
        case BinOp(kind = BinaryOperatorKind.Add): return expand(expr.left) + expand(expr.right)
        case BinOp(kind = BinaryOperatorKind.Mul): return [l + r for l in expand(expr.left) for r in expand(expr.right)]
        case TimeDependent(): return [[expr]]



def to_sum_of_products(expr: PIQobj, tlist: [float]):
    terms = expand(expr)
    groups = defaultdict(complex)

    for term in terms:
        coeff = 1
        spins = []
        bosons = []
        tfunc = complex(1)

        for value in term:
            match value:
                case PIOperatorKind.Jz | PIOperatorKind.Jp | PIOperatorKind.Jm:
                    spins.append(value)
                case PIOperatorKind.A | PIOperatorKind.Ad | PIOperatorKind.Ap | PIOperatorKind.As:
                    bosons.append(value)
                case TimeDependent():
                    arr = value.func(tlist)
                    if value.conj: arr = arr.conj()
                    tfunc *= arr
                case _:
                    coeff *= value

        key = (tuple(spins), tuple(bosons), (tfunc,) if isinstance(tfunc, complex) else tuple(tfunc))
        groups[key] += coeff

    return [(c, spins, bosons, tfunc)
        for (spins, bosons, tfunc), c in groups.items()
        if not isclose(abs(c), 0)
    ]

