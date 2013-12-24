from discodop.containers cimport ULLong, ULong, UInt, UShort, UChar
from libc.string cimport memcpy

# See: http://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html
cdef extern:
	int __builtin_ffsll(ULLong)
	int __builtin_ctzll(ULLong)
	int __builtin_clzll(ULLong)
	int __builtin_ctzl(ULong)
	int __builtin_popcountl(ULong)
	int __builtin_popcountll(ULLong)

# ctz equivalents:
# - visual studio 2005:
#   unsigned long _BitScanForward(&result, unsigned __int64 val)
# - llvm: llvm.ctlz(i64 val, 1)
# - POSIX (no 64 bit): ffs(int val)
# Cf. http://en.wikipedia.org/wiki/Find_first_set

cdef extern from "macros.h":
	int BITSIZE
	int BITSLOT(int b)
	ULong BITMASK(int b)
	ULong TESTBIT(ULong a[], int b)
	void CLEARBIT(ULong a[], int b)


ctypedef fused unsigned_fused:
	UChar
	UShort
	UInt
	ULong
	ULLong


# cpdef functions defined in bit.pyx
#on python integers
cpdef int fanout(arg)
cpdef int pyintnextset(a, int pos)
#on C integers
cpdef int bitcount(ULLong vec)

# cdef inline functions defined here:
#on C integers
#cdef inline bint testbit(unsigned_fused vec, UInt pos)
#on ULLongs
#cdef inline int nextset(ULLong vec, UInt pos)
#cdef inline int nextunset(ULLong vec, UInt pos)
#cdef inline int bitlength(ULLong vec)
#on arrays of unsigned long
#cdef inline int abitcount(ULong *vec, short slots)
#cdef inline int anextset(ULong *vec, UInt pos, short slots)
#cdef inline int anextunset(ULong *vec, UInt pos, short slots)
#cdef inline bint subset(ULong *vec1, ULong *vec2, short slots)
#cdef inline void setunion(ULong *dest, ULong *src, short slots)
#cdef inline void ulongset(ULong *dest, ULong value, short slots)
#cdef inline void ulongcpy(ULong *dest, ULong *src, short slots)


cdef inline bint testbit(unsigned_fused vec, UInt pos):
	"""Mask a particular bit, return nonzero if set.

	>>> testbit(0b0011101, 0)
	True
	>>> testbit(0b0011101, 1)
	False
	>>> testbit(0b100000000000000000000000000000000, 32) != 0
	True
	"""
	if (unsigned_fused is UChar
			or unsigned_fused is UShort
			or unsigned_fused is UInt):
		return vec & (1U << pos)
	else:
		return vec & ((<unsigned_fused>1U) << pos) != 0


cdef inline int nextset(ULLong vec, UInt pos):
	""" Return next set bit starting from pos, -1 if there is none.

	>>> nextset(0b001101, 1)
	2
	"""
	#return (pos + __builtin_ctzll(vec >> pos)) if (vec >> pos) else -1
	# mask instead of shift:
	return __builtin_ffsll(vec & (~0ULL << pos)) - 1
	#ULLong x = vec & ~((1 << pos) - 1)
	#ULLong x = (vec >> pos) << pos
	#return x ? __builtin_ctzll(x) : -1
	#return  __builtin_ffsll(x) - 1


cdef inline int nextunset(ULLong vec, UInt pos):
	""" Return next unset bit starting from pos.

	>> nextunset(0b001101, 2)
	4
	>> nextunset((1<<64)-1, 0)
	64
	"""
	cdef ULLong x = ~vec & (~0ULL << pos)
	return __builtin_ctzll(x) if x else (sizeof(ULLong) * 8)


cdef inline int bitlength(ULLong vec):
	"""Return number of bits needed to represent vector.

	(equivalently: index of most significant set bit, plus one)

	>>> bitlength(0b0011101)
	5"""
	return sizeof(vec) * 8 - __builtin_clzll(vec)


cdef inline int abitcount(ULong *vec, short slots):
	""" Return number of set bits in variable length bitvector """
	cdef short a
	cdef int result = 0
	for a in range(slots):
		result += __builtin_popcountl(vec[a])
	return result


cdef inline int abitlength(ULong *vec, short slots):
	"""Return number of bits needed to represent vector.

	(equivalently: index of most significant set bit, plus one)."""
	cdef short a = slots - 1
	while a and not vec[a]:
		a -= 1
	return (a + 1) * sizeof(ULong) * 8 - __builtin_clzll(vec[a])


cdef inline int anextset(ULong *vec, UInt pos, short slots):
	""" Return next set bit starting from pos, -1 if there is none. """
	cdef short a = BITSLOT(pos)
	cdef ULong x
	if a >= slots:
		return -1
	x = vec[a] & (~0UL << (pos % BITSIZE))
	while x == 0UL:
		a += 1
		if a == slots:
			return -1
		x = vec[a]
	return a * BITSIZE + __builtin_ctzl(x)


cdef inline int anextunset(ULong *vec, UInt pos, short slots):
	""" Return next unset bit starting from pos. """
	cdef short a = BITSLOT(pos)
	cdef ULong x
	if a >= slots:
		return a * BITSIZE
	x = vec[a] | (BITMASK(pos) - 1)
	while x == ~0UL:
		a += 1
		if a == slots:
			return a * BITSIZE
		x = vec[a]
	return a * BITSIZE + __builtin_ctzl(~x)


cdef inline short iteratesetbits(ULong *vec, short slots,
		ULong *cur, short *idx):
	"""Iterate over set bits in an array of unsigned long.

	:param slots: number of elements in unsigned long array ``vec``.
	:param cur and idx: pointers to variables to maintain state,
		``idx`` should be initialized to 0, and ``cur`` to the first element of
		the bit array ``vec``, i.e., ``cur = vec[idx]``.
	:returns: the index of a set bit, or -1 if there are no more set
		bits. The result of calling stopped iterator is undefined.

	e.g.::

		ULong vec[4] = {0, 0, 0, 0b10001}, cur = vec[0]
		short idx = 0
		iteratesetbits(vec, 0, 4, &cur, &idx) # returns 0
		iteratesetbits(vec, 0, 4, &cur, &idx) # returns 4
		iteratesetbits(vec, 0, 4, &cur, &idx) # returns -1
	"""
	cdef short tmp
	while not cur[0]:
		idx[0] += 1
		if idx[0] >= slots:
			return -1
		cur[0] = vec[idx[0]]
	tmp = __builtin_ctzl(cur[0])  # index of bit in current slot
	CLEARBIT(cur, tmp)
	return idx[0] * BITSIZE + tmp


cdef inline short iterateunsetbits(ULong *vec, short slots,
		ULong *cur, short *idx):
	""" Like ``iteratesetbits``, but return indices of zero bits. """
	cdef short tmp
	while not ~cur[0]:
		idx[0] += 1
		if idx[0] >= slots:
			return -1
		cur[0] = vec[idx[0]]
	tmp = __builtin_ctzl(~cur[0])  # index of bit in current slot
	CLEARBIT(cur, tmp)
	return idx[0] * BITSIZE + tmp


cdef inline void ulongset(ULong *dest, ULong value, short slots):
	"""Like memset, but set one ULong at a time.

	Should be faster for small arrays."""
	cdef short a
	for a in range(slots):
		dest[a] = value

#cdef inline void ulongcpy(ULong *dest, ULong *src, short slots):
#	"""Like memcpy, but copy one ULong at a time.
#
#	Should be faster for small arrays."""
#	cdef short a
#	for a in range(slots):
#		dest[a] = src[a]

cdef inline void ulongcpy(ULong *dest, ULong *src, short slots):
	"""memcpy wrapper for unsigned long arrays."""
	memcpy(<char *>dest, <char *>src, slots * sizeof(ULong))


cdef inline void setintersectinplace(ULong *dest, ULong *src, short slots):
	"""dest gets the intersection of dest and src.

	both operands must have at least `slots' slots."""
	cdef short a
	for a in range(slots):
		dest[a] &= src[a]


cdef inline void setunioninplace(ULong *dest, ULong *src, short slots):
	"""dest gets the union of dest and src.

	Both operands must have at least ``slots`` slots."""
	cdef short a
	for a in range(slots):
		dest[a] |= src[a]


cdef inline void setunion(ULong *dest, ULong *src1, ULong *src2, short slots):
	"""dest gets the union of src1 and src2.

	operands must have at least ``slots`` slots."""
	cdef short a
	for a in range(slots):
		dest[a] = src1[a] | src2[a]


cdef inline bint subset(ULong *vec1, ULong *vec2, short slots):
	"""Test whether vec1 is a subset of vec2.

	i.e., all set bits of vec1 should be set in vec2."""
	cdef short a
	for a in range(slots):
		if (vec1[a] & vec2[a]) != vec1[a]:
			return False
	return True
