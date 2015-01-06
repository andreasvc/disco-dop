"""Functions for working with bitvectors.

NB: most functions are in bit.pxd to facilitate function inlining."""
from __future__ import print_function


def pyintbitcount(a):
	"""Return number of set bits (1s) in a Python integer.

	>>> pyintbitcount(0b0011101)
	4"""
	count = 0
	while a:
		a &= a - 1
		count += 1
	return count


cpdef int bitcount(uint64_t vec):
	"""Return number of set bits (1s).

	>>> bitcount(0b0011101)
	4"""
	return bit_popcount(vec)


cpdef int pyintnextset(a, int pos):
	"""Return index of first set bit, starting from pos.

	>>> pyintnextset(0b001101, 1)
	2"""
	cdef uint64_t mask = -1
	a >>= pos
	if a == 0:
		return -1
	while a & mask == 0:
		a >>= (8 * sizeof(uint64_t))
		pos += (8 * sizeof(uint64_t))
	return pos + bit_ctz(a & mask)


cpdef int fanout(arg):
	"""Return number of contiguous components in bit vector (gaps plus one).

	>>> fanout(0b011011011)
	3"""
	cdef uint32_t result = 0
	cdef uint64_t vec
	if arg < ((1UL << (8UL * sizeof(uint64_t) - 1UL)) - 1UL):
		vec = arg
		while vec:
			vec >>= bit_ctz(vec)
			vec >>= bit_ctz(~vec)
			result += 1
	else:
		# when the argument does not fit in unsigned long long, resort to plain
		# Python code to use Python's bigints. This algorithm is based
		# on a bit counting algorihm. It goes through as many iterations as
		# there are set bits (while the other algorithm jumps over contiguous
		# sequences of set bits in one go).
		prev = arg
		result = 0
		while arg:
			arg &= arg - 1
			if ((prev - arg) << 1) & prev == 0:
				result += 1
			prev = arg
	return result


def test():
	cdef uint64_t ulongvec[2]
	bigpyint = 0b11100110101111001101011111100110101001100110
	assert nextset(0b001100110, 3) == 5
	assert nextset(0b001100110, 7) == -1
	assert nextunset(0b001100110, 1) == 3
	assert nextunset(~0UL, 0) == 64
	assert pyintnextset(0b001100110, 3) == 5
	assert pyintnextset(0b001100110, 7) == -1
	assert pyintnextset(bigpyint, 3) == 5
	assert pyintnextset(bigpyint, 7) == 9
	assert bitcount(0b001100110) == 4
	assert bitlength(0b001100110) == 7
	assert testbit(0b001100110, 1)
	assert not testbit(0b001100110, 3)
	assert fanout(0b0111100) == 1
	assert fanout(0b1000001) == 2
	assert fanout(0b011011011) == 3
	ulongvec[0] = 1UL << (sizeof(uint64_t) * 8 - 1)
	ulongvec[1] = 1
	assert anextset(ulongvec, 0, 2) == sizeof(uint64_t) * 8 - 1, (
			anextset(ulongvec, 0, 2), sizeof(uint64_t) * 8 - 1)
	assert anextset(ulongvec, sizeof(uint64_t) * 8, 2) == sizeof(uint64_t) * 8, (
		anextset(ulongvec, sizeof(uint64_t) * 8, 2), sizeof(uint64_t) * 8)
	assert anextunset(ulongvec, 0, 2) == 0, (
		anextunset(ulongvec, 0, 2), 0)
	assert anextunset(ulongvec, sizeof(uint64_t) * 8 - 1, 2) == (
			sizeof(uint64_t) * 8 + 1), (anextunset(ulongvec,
			sizeof(uint64_t) * 8 - 1, 2), sizeof(uint64_t) * 8 + 1)
	ulongvec[0] = 0
	assert anextset(ulongvec, 0, 2) == sizeof(uint64_t) * 8, (
		anextset(ulongvec, 0, 2), sizeof(uint64_t) * 8)
	ulongvec[1] = 0
	assert anextset(ulongvec, 0, 2) == -1, (
		anextset(ulongvec, 0, 2), -1)
	ulongvec[0] = ~0UL
	assert anextunset(ulongvec, 0, 2) == sizeof(uint64_t) * 8, (
		anextunset(ulongvec, 0, 2), sizeof(uint64_t) * 8)
	print('it worked')

__all__ = ['bitcount', 'fanout', 'pyintbitcount', 'pyintnextset']
