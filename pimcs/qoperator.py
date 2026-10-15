from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Iterable

import numpy as np


class QOperator(Enum):
	Jp = auto()
	Jz = auto()
	Jm = auto()
	Create = auto()
	Destroy = auto()
	Alpha = auto()
	AlphaConj = auto()


class QBinaryOperator(Enum):
	Add = auto()
	Mul = auto()


Term = tuple[int, list[QOperator]]


REWRITES = {
	(QOperator.Jz, QOperator.Jp): [
		(1, [QOperator.Jp, QOperator.Jz]),
		(1, [QOperator.Jp]),
	],
	(QOperator.Jm, QOperator.Jz): [
		(1, [QOperator.Jz, QOperator.Jm]),
		(1, [QOperator.Jm]),
	],
	(QOperator.Jm, QOperator.Jp): [
		(1, [QOperator.Jp, QOperator.Jm]),
		(-2, [QOperator.Jz]),
	],
	(QOperator.Destroy, QOperator.Create): [
		(1, [QOperator.Create, QOperator.Destroy]),
		(1, []),
	],
}


def to_normal_ordering(ops: list[QOperator], coefficient: int = 1) -> list[Term]:
	for i, pair in enumerate(zip(ops, ops[1:])):
		if pair in REWRITES:
			prefix, suffix = ops[:i], ops[i+2:]

			return [
				t
				for c, middle in REWRITES[pair]
				for t in to_normal_ordering(prefix + middle + suffix, coefficient * c)
			]

	return [(coefficient, ops)]


def wrap_qobj(object):
	if isinstance(object, QExpr):
		return object
	if callable(object):
		return QFunc(object)
	return QLeafExpr(object)


class QExpr:
	def __neg__(self):
		return QBinaryExpr(QBinaryOperator.Mul, self, QLeafExpr(-1))

	def __add__(self, other: QExpr):
		return QBinaryExpr(QBinaryOperator.Add, self, wrap_qobj(other))

	def __mul__(self, other: QExpr):
		return QBinaryExpr(QBinaryOperator.Mul, self, wrap_qobj(other))

	def __sub__(self, other: QExpr):
		return QBinaryExpr(QBinaryOperator.Add, self, -wrap_qobj(other))

	def __rmul__(self, other: complex | Callable):
		leaf = QFunc(other) if callable(other) else complex(other)
		return QBinaryExpr(QBinaryOperator.Mul, self, QLeafExpr(leaf))

	def __pow__(self, n: int):
		if not isinstance(n, int) or n < 0:
			raise ValueError(f"Exponent of QExpr must be a non-negative integer, got {n!r}")

		return QLeafExpr(1) if n == 0 else self * (self ** (n - 1))

	def dag(self):
		match self:
			case QLeafExpr(value):
				if isinstance(value, (int, float, complex, np.number)):
					return QLeafExpr(complex(value).conjugate())

				match value:
					case QOperator.Jz:
						return QLeafExpr(QOperator.Jz)
					case QOperator.Jp:
						return QLeafExpr(QOperator.Jm)
					case QOperator.Jm:
						return QLeafExpr(QOperator.Jp)
					case QOperator.Destroy:
						return QLeafExpr(QOperator.Create)
					case QOperator.Create:
						return QLeafExpr(QOperator.Destroy)
					case QOperator.Alpha:
						return QLeafExpr(QOperator.AlphaConj)
					case QOperator.AlphaConj:
						return QLeafExpr(QOperator.Alpha)
					case _:
						raise AssertionError("unreachable code path reached!!!")

			case QBinaryExpr(kind, lhs, rhs):
				return QBinaryExpr(kind, rhs.dag(), lhs.dag())

			case _:
				raise AssertionError("unreachable code path reached!!!")

	def displace(self):
		match self:
			case QLeafExpr(QOperator.Destroy):
				return QBinaryExpr(QBinaryOperator.Add, QLeafExpr(QOperator.Destroy), QLeafExpr(QOperator.Alpha))
			case QLeafExpr(QOperator.Create):
				return QBinaryExpr(QBinaryOperator.Add, QLeafExpr(QOperator.Create), QLeafExpr(QOperator.AlphaConj))
			case QBinaryExpr(kind, lhs, rhs):
				return QBinaryExpr(kind, lhs.displace(), rhs.displace())
			case _:
				return self

	def is_hermitian(self):
		return len((self - self.dag()).to_sum_of_products()) == 0

	def expand(self):
		match self:
			case QFunc():
				return [[self]]
			case QLeafExpr(value):
				return [[value]]
			case QBinaryExpr(kind=QBinaryOperator.Add, lhs=lhs, rhs=rhs):
				return lhs.expand() + rhs.expand()
			case QBinaryExpr(kind=QBinaryOperator.Mul, lhs=lhs, rhs=rhs):
				return [l + r for l in lhs.expand() for r in rhs.expand()]
			case _:
				raise AssertionError("unreachable code path reached!!!")

	def to_sum_of_products(self):
		terms = defaultdict(complex)

		for term in self.expand():
			coefficient = 1
			spin_operators = []
			boson_operators = []
			shift_factors = []
			functions = []

			for value in term:
				match value:
					case QOperator.Jz | QOperator.Jp | QOperator.Jm:
						spin_operators.append(value)
					case QOperator.Create | QOperator.Destroy:
						boson_operators.append(value)
					case QOperator.Alpha | QOperator.AlphaConj:
						shift_factors.append(value)
					case QFunc(f):
						functions.append(f)
					case _:
						coefficient *= value

			spin_terms = to_normal_ordering(spin_operators)
			boson_terms = to_normal_ordering(boson_operators)

			shift_factors.sort(key=lambda op: op.value)
			functions.sort(key=lambda f: id(f))

			for c1, ops1 in spin_terms:
				for c2, ops2 in boson_terms:
					key = tuple(ops1 + ops2 + shift_factors), tuple(functions)
					terms[key] += coefficient * c1 * c2

		return [
			(coefficient, ops, fs) for (ops, fs), coefficient in terms.items()
			if not np.isclose(np.abs(coefficient), 0)
		]


@dataclass
class QFunc(QExpr):
	f: Callable

	def __hash__(self):
		return hash(id(self.f))

@dataclass
class QLeafExpr(QExpr):
	value: complex | QOperator | QFunc


@dataclass
class QBinaryExpr(QExpr):
	kind: QBinaryOperator
	lhs: QExpr
	rhs: QExpr


def jspin(ops: str = "xyz") -> Iterable[QExpr]:
	jp = QLeafExpr(QOperator.Jp)
	jm = QLeafExpr(QOperator.Jm)
	jz = QLeafExpr(QOperator.Jz)

	jx = 0.5 * (jp + jm)
	jy = 0.5j * (jp - jm)

	valid_operators = {
		"+": jp,
		"-": jm,
		"x": jx,
		"y": jy,
		"z": jz,
	}

	for op in ops:
		if op not in valid_operators:
			raise ValueError(f"Invalid operator: {op!r}")

	operators = tuple(valid_operators[op] for op in ops)
	return operators[0] if len(operators) == 1 else operators


def destroy() -> QExpr:
	return QLeafExpr(QOperator.Destroy)


def commutator(a: QExpr, b: QExpr) -> QExpr:
	result = a * b - b * a

	if len(result.to_sum_of_products()) == 0:
		return 0

	return result
