from __future__ import annotations
from dataclasses import dataclass
from typing import Union
from collections import defaultdict
from enum import Enum, auto
from math import isclose


class PIOperatorKind(Enum):
    Jz = auto()
    Jp = auto()
    Jm = auto()
    A  = auto()
    Ad = auto()
    
    # pseudo-operators to represent alpha (alpha * Identity)
    Alpha = auto()
    AlphaStar = auto()


class BinaryOperatorKind(Enum):
    Add = auto()
    Mul = auto()


class PIExpression:
    def __add__(self, other):  return BinOp(BinaryOperatorKind.Add, self,  other)
    def __mul__(self, other):  return BinOp(BinaryOperatorKind.Mul, self,  other)
    def __sub__(self, other):  return BinOp(BinaryOperatorKind.Add, self, -other)
    def __neg__(self):         return BinOp(BinaryOperatorKind.Mul, self, -1)
    def __rmul__(self, other): return BinOp(BinaryOperatorKind.Mul, self, coeff(other))

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
            case BinOp():
                return BinOp(self.kind, self.left.dag(), self.right.dag())



    def displace(self):
        match self:
            case Leaf(value = PIOperatorKind.A):
                return BinOp(BinaryOperatorKind.Add, self, coeff(PIOperatorKind.Alpha))
            case Leaf(value = PIOperatorKind.Ad):
                return BinOp(BinaryOperatorKind.Add, self, coeff(PIOperatorKind.AlphaStar))
            case Leaf():
                return self
            case BinOp():
                return BinOp(self.kind, self.left.displace(), self.right.displace())

    _OP_NAMES = {
        PIOperatorKind.Jz: 'Jz',
        PIOperatorKind.Jp: 'J+',
        PIOperatorKind.Jm: 'J-',
        PIOperatorKind.A:  'a',
        PIOperatorKind.Ad: 'a†',
        PIOperatorKind.Alpha: 'α',
        PIOperatorKind.AlphaStar: 'α*',
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

@dataclass
class Leaf(PIExpression):
    value: complex | PIOperatorKind
    dim: int | None

@dataclass
class BinOp(PIExpression):
    kind:  BinaryOperatorKind
    left:  Expr
    right: Expr

def coeff(value) -> Leaf:
    return Leaf(value, None)

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
            return validate_pair(ls, rs), validate_pair(lb, rb)


def expand(expr: PIQobj) -> list[list]:
    match expr:
        case Leaf(): return [[expr.value]]
        case BinOp(kind = BinaryOperatorKind.Add): return expand(expr.left) + expand(expr.right)
        case BinOp(kind = BinaryOperatorKind.Mul): return [l + r for l in expand(expr.left) for r in expand(expr.right)]


def to_sum_of_products(expr: PIQobj):
    terms = expand(expr)
    groups = defaultdict(complex)

    for term in terms:
        coeff = 1
        spins = []
        bosons = []

        for value in term:
            match value:
                case PIOperatorKind.Jz | PIOperatorKind.Jp | PIOperatorKind.Jm:
                    spins.append(value)
                case PIOperatorKind.A | PIOperatorKind.Ad:
                    bosons.append(value)
                case PIOperatorKind.Alpha | PIOperatorKind.AlphaStar:
                    bosons.append(value)
                case _:
                    coeff *= value

        key = (tuple(spins), tuple(bosons))
        groups[key] += coeff

    return [(c, spins, bosons)
        for (spins, bosons), c in groups.items()
        if not isclose(abs(c), 0)
    ]



def ops_to_factor(ops) -> tuple[int, int, str]:
    if len(ops) == 0:
        return 0, 0, "1"

    spin_index = 0
    boson_index = 0
    weights = []

    for op in reversed(ops):
        match op:
            case PIOperatorKind.Jz:
                weights.append(f"(m + {spin_index})")
            case PIOperatorKind.Jp:
                weights.append(f"sqrtf((jpm + 1 - {spin_index}) * (jmm + {spin_index}))")
                spin_index -= 1
            case PIOperatorKind.Jm:
                weights.append(f"sqrtf((jmm + 1 + {spin_index}) * (jpm - {spin_index}))")
                spin_index += 1
            case PIOperatorKind.A:
                weights.append(f"sqrtf(a + {boson_index})")
                boson_index -= 1
            case PIOperatorKind.Ad:
                weights.append(f"sqrtf(a + 1 + {boson_index})")
                boson_index += 1
            case PIOperatorKind.Alpha:
                weights.append(f"state->alpha")
            case PIOperatorKind.AlphaStar:
                weights.append(f"conjf(state->alpha)")

    return spin_index, boson_index, " * ".join(weights)

