from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t

cdef extern from "macros.h":
	int BITSIZE
	int BITSLOT(int b)
	uint64_t BITMASK(int b)
	uint64_t TESTBIT(uint64_t a[], int b)
	void CLEARBIT(uint64_t a[], int b)
	void SETBIT(uint64_t a[], int b)


cdef extern from "bitcount.h":
	unsigned int bit_clz(uint64_t)
	unsigned int bit_ctz(uint64_t)
	unsigned int bit_popcount(uint64_t)


ctypedef fused unsigned_fused:
	uint8_t
	uint16_t
	uint32_t
	uint64_t


# cpdef functions defined in bit.pyx
# on python integers
cpdef int fanout(arg)
cpdef int pyintnextset(a, int pos)
# on C integers
cpdef int bitcount(uint64_t vec)

# cdef inline functions defined here:
# ===================================
# - on variously sized C integers
# cdef inline bint testbit(unsigned_fused vec, uint32_t pos)
# - on uint64_t
# cdef inline int nextset(uint64_t vec, uint32_t pos)
# cdef inline int nextunset(uint64_t vec, uint32_t pos)
# cdef inline int bitlength(uint64_t vec)
# - on uint64_t arrays
# cdef inline int abitcount(uint64_t *vec, int slots)
# cdef inline int anextset(uint64_t *vec, uint32_t pos, int slots)
# cdef inline int anextunset(uint64_t *vec, uint32_t pos, int slots)
# cdef inline bint subset(uint64_t *vec1, uint64_t *vec2, int slots)
# cdef inline void setunioninplace(uint64_t *dest, uint64_t *src,
# 		int slots)
# cdef inline void setintersectinplace(uint64_t *dest, uint64_t *src,
# 		int slots)
# cdef inline void setunion(uint64_t *dest, uint64_t *src1, uint64_t *src2,
# 		int slots)
# cdef inline void setintersect(uint64_t *dest, uint64_t *src1, uint64_t *src2,
# 		int slots)
# cdef inline int iteratesetbits(uint64_t *vec, int slots,
# 		uint64_t *cur, int *idx)
# cdef inline int iterateunsetbits(uint64_t *vec, int slots,
# 		uint64_t *cur, int *idx)
# cdef inline int reviteratesetbits(uint64_t *vec, uint64_t *cur, int *idx)


cdef inline bint testbit(unsigned_fused vec, uint32_t pos):
	"""Mask a particular bit, return nonzero if set.

	>>> testbit(0b0011101, 0)
	True
	>>> testbit(0b0011101, 1)
	False
	>>> testbit(0b100000000000000000000000000000000, 32) != 0
	True
	"""
	if (unsigned_fused is uint8_t
			or unsigned_fused is uint16_t
			or unsigned_fused is uint32_t):
		return vec & (1U << pos)
	else:
		return vec & ((<unsigned_fused>1U) << pos) != 0


cdef inline int nextset(uint64_t vec, uint32_t pos):
	""" Return next set bit starting from pos, -1 if there is none.

	>>> nextset(0b001101, 1)
	2
	"""
	return (pos + bit_ctz(vec >> pos)) if (vec >> pos) else -1
	# mask instead of shift:
	# return __builtin_ffsl(vec & (~0UL << pos)) - 1
	# uint64_t x = vec & ~((1 << pos) - 1)
	# uint64_t x = (vec >> pos) << pos
	# return x ? bit_ctz(x) : -1
	# return  __builtin_ffsl(x) - 1


cdef inline int nextunset(uint64_t vec, uint32_t pos):
	""" Return next unset bit starting from pos.

	>> nextunset(0b001101, 2)
	4
	>> nextunset((1<<64)-1, 0)
	64
	"""
	cdef uint64_t x = ~vec & (~0UL << pos)
	return bit_ctz(x) if x else (sizeof(uint64_t) * 8)


cdef inline int bitlength(uint64_t vec):
	"""Return number of bits needed to represent vector.

	(equivalently: index of most significant set bit, plus one)

	>>> bitlength(0b0011101)
	5"""
	return sizeof(vec) * 8 - bit_clz(vec) if vec else 0


cdef inline int abitcount(uint64_t *vec, int slots):
	""" Return number of set bits in variable length bitvector """
	cdef int a
	cdef int result = 0
	for a in range(slots):
		result += bit_popcount(vec[a])
	return result


cdef inline int abitlength(uint64_t *vec, int slots):
	"""Return number of bits needed to represent vector.

	(equivalently: index of most significant set bit, plus one)."""
	cdef int a = slots - 1
	while not vec[a]:
		a -= 1
		if a < 0:
			return 0
	return (a + 1) * sizeof(uint64_t) * 8 - bit_clz(vec[a])


cdef inline int anextset(uint64_t *vec, uint32_t pos, int slots):
	""" Return next set bit starting from pos, -1 if there is none. """
	cdef int a = BITSLOT(pos)
	cdef uint64_t x
	if a >= slots:
		return -1
	x = vec[a] & (~0UL << (pos % BITSIZE))
	while x == 0UL:
		a += 1
		if a == slots:
			return -1
		x = vec[a]
	return a * BITSIZE + bit_ctz(x)


cdef inline int anextunset(uint64_t *vec, uint32_t pos, int slots):
	""" Return next unset bit starting from pos. """
	cdef int a = BITSLOT(pos)
	cdef uint64_t x
	if a >= slots:
		return a * BITSIZE
	x = vec[a] | (BITMASK(pos) - 1)
	while x == ~0UL:
		a += 1
		if a == slots:
			return a * BITSIZE
		x = vec[a]
	return a * BITSIZE + bit_ctz(~x)


cdef inline int iteratesetbits(uint64_t *vec, int slots,
		uint64_t *cur, int *idx):
	"""Iterate over set bits in an array of unsigned long.

	:param slots: number of elements in unsigned long array ``vec``.
	:param cur and idx: pointers to variables to maintain state,
		``idx`` should be initialized to 0,
		and ``cur`` to the first element of
		the bit array ``vec``, i.e., ``cur = vec[idx]``.
	:returns: the index of a set bit, or -1 if there are no more set
		bits. The result of calling a stopped iterator is undefined.

	e.g.::

		int idx = 0
		uint64_t vec[4] = {0, 0, 0, 0b10001}, cur = vec[idx]
		iteratesetbits(vec, 4, &cur, &idx)  # returns 0
		iteratesetbits(vec, 4, &cur, &idx)  # returns 4
		iteratesetbits(vec, 4, &cur, &idx)  # returns -1
	"""
	cdef int tmp
	while not cur[0]:
		idx[0] += 1
		if idx[0] >= slots:
			return -1
		cur[0] = vec[idx[0]]
	tmp = bit_ctz(cur[0])  # index of bit in current slot
	cur[0] ^= 1UL << tmp  # TOGGLEBIT(cur, tmp)
	return idx[0] * BITSIZE + tmp


cdef inline int iterateunsetbits(uint64_t *vec, int slots,
		uint64_t *cur, int *idx):
	"""Like ``iteratesetbits``, but return indices of zero bits.

	:param cur: should be initialized as: ``cur = ~vec[idx]``."""
	cdef int tmp
	while not cur[0]:
		idx[0] += 1
		if idx[0] >= slots:
			return -1
		cur[0] = ~vec[idx[0]]
	tmp = bit_ctz(cur[0])  # index of bit in current slot
	cur[0] ^= 1UL << tmp  # TOGGLEBIT(cur, tmp)
	return idx[0] * BITSIZE + tmp


cdef inline void setintersectinplace(uint64_t *dest, uint64_t *src, int slots):
	"""dest gets the intersection of dest and src.

	both operands must have at least `slots' slots."""
	cdef int a
	for a in range(slots):
		dest[a] &= src[a]


cdef inline void setunioninplace(uint64_t *dest, uint64_t *src, int slots):
	"""dest gets the union of dest and src.

	Both operands must have at least ``slots`` slots."""
	cdef int a
	for a in range(slots):
		dest[a] |= src[a]


cdef inline void setintersect(uint64_t *dest, uint64_t *src1, uint64_t *src2,
		int slots):
	"""dest gets the intersection of src1 and src2.

	operands must have at least ``slots`` slots."""
	cdef int a
	for a in range(slots):
		dest[a] = src1[a] & src2[a]


cdef inline void setunion(uint64_t *dest, uint64_t *src1, uint64_t *src2,
		int slots):
	"""dest gets the union of src1 and src2.

	operands must have at least ``slots`` slots."""
	cdef int a
	for a in range(slots):
		dest[a] = src1[a] | src2[a]


cdef inline bint subset(uint64_t *vec1, uint64_t *vec2, int slots):
	"""Test whether vec1 is a subset of vec2.

	i.e., all set bits of vec1 should be set in vec2."""
	cdef int a
	for a in range(slots):
		if (vec1[a] & vec2[a]) != vec1[a]:
			return False
	return True
