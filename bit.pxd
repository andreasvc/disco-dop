from containers cimport ULLong, ULong, UInt, UChar

cdef extern:
	int __builtin_ffsll (ULLong)
	int __builtin_ctzll (ULLong)
	int __builtin_clzll (ULLong)
	int __builtin_ctzl (ULong)
	int __builtin_popcountl (ULong)
	int __builtin_popcountll (ULLong)

cdef extern from "macros.h":
	int BITSIZE
	int BITSLOT(int b)
	ULong BITMASK(int b)
	ULong TESTBIT(ULong a[], int b)

# cpdef functions defined in bit.pyx
#on python integers
cpdef int fanout(arg)
cpdef int pyintnextset(a, int pos)

#on ULLongs
cpdef int bitcount(ULLong vec)
cpdef bint bitminmax(ULLong a, ULLong b)
cpdef bint testbitc(UChar arg, UInt pos)
cpdef bint testbitshort(unsigned short arg, UInt pos)

cdef binrepr(ULong *vec, int slots)

# cdef inline functions defined here:
#on ULLongs
#cdef inline int nextset(ULLong vec, UInt pos)
#cdef inline int nextunset(ULLong vec, UInt pos)
#cdef inline bint testbit(ULLong vec, UInt pos)
#cdef inline bint testbitint(UInt arg, UInt pos)
#cdef inline int bitlength(ULLong vec)

#on arrays of unsigned long
#cdef inline int abitcount(ULong *vec, UInt slots)
#cdef inline int anextset(ULong *vec, UInt pos, UInt slots)
#cdef inline int anextunset(ULong *vec, UInt pos, UInt slots)
#cdef inline bint subset(ULong *vec1, ULong *vec2, UInt slots)
#cdef inline void setunion(ULong *dest, ULong *src, UInt slots)
#cdef inline void ulongset(ULong *dest, ULong value, UInt slots)
#cdef inline void ulongcpy(ULong *dest, ULong *src, UInt slots)


# See: http://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html
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

# todo: see if this can be turned into a macro
cdef inline bint testbit(ULLong vec, UInt pos):
	""" Mask a particular bit, return nonzero if set
	>>> testbit(0b0011101, 0)
	True
	>>> testbit(0b0011101, 1)
	False
	>>> testbit(0b100000000000000000000000000000000, 32) != 0
	True
	"""
	return vec & (1ULL << pos) != 0

cdef inline bint testbitint(UInt arg, UInt pos):
	""" Mask a particular bit, return nonzero if set
	>>> testbit(0b0011101, 0)
	1
	>>> testbit(0b0011101, 1)
	0"""
	return arg & (1ULL << pos)

cdef inline int bitlength(ULLong vec):
	""" number of bits needed to represent vector
	(equivalently: index of most significant set bit, plus one)
	>>> bitlength(0b0011101)
	5"""
	return sizeof(vec) * 8 - __builtin_clzll(vec)

cdef inline int abitcount(ULong *vec, UInt slots):
	""" number of set bits in variable length bitvector """
	cdef int a, result = __builtin_popcountl(vec[0])
	for a from 1 <= a < slots:
		result += __builtin_popcountl(vec[a])
	return result

cdef inline int abitlength(ULong *vec, UInt slots):
	""" number of bits needed to represent vector
	(equivalently: index of most significant set bit, plus one)"""
	cdef int a = slots - 1
	while a and not vec[a]: a -= 1
	return (a + 1) * sizeof(ULong) * 8 - __builtin_clzll(vec[a])

cdef inline int anextset(ULong *vec, UInt pos, UInt slots):
	""" return next set bit starting from pos, -1 if there is none. """
	cdef UInt a = BITSLOT(pos)
	cdef ULong x
	if a >= slots: return -1
	x = vec[a] & (~0UL << (pos % BITSIZE))
	while x == 0:
		a += 1
		if a == slots: return -1
		x = vec[a]
	return a * BITSIZE + __builtin_ctzl(x)

cdef inline int anextunset(ULong *vec, UInt pos, UInt slots):
	""" return next unset bit starting from pos. """
	cdef UInt a = BITSLOT(pos)
	cdef ULong x
	if a >= slots: return a * BITSIZE
	x = vec[a] | (BITMASK(pos) - 1)
	while x == ~0UL:
		a += 1
		if a == slots: return a * BITSIZE
		x = vec[a]
	return a * BITSIZE + __builtin_ctzl(~x)

cdef inline void ulongset(ULong *dest, ULong value, UInt slots):
	""" Like memset, but set one ULong at a time; should be faster
	for small arrays. """
	cdef int a
	for a in range(slots): dest[a] = value

cdef inline void ulongcpy(ULong *dest, ULong *src, UInt slots):
	""" Like memcpy, but copy one ULong at a time; should be faster
	for small arrays. """
	cdef int a
	for a in range(slots): dest[a] = src[a]

cdef inline void setunioninplace(ULong *dest, ULong *src, UInt slots):
	""" dest gets the union of dest and src; both operands must have at least
	`slots' slots. """
	cdef int a
	for a in range(slots): dest[a] |= src[a]

cdef inline void setunion(ULong *dest, ULong *src1, ULong *src2, UInt slots):
	""" dest gets the union of src1 and src2; operands must have at least
	`slots' slots. """
	cdef int a
	for a in range(slots): dest[a] = src1[a] | src2[a]

cdef inline bint subset(ULong *vec1, ULong *vec2, UInt slots):
	""" test whether vec1 is a subset of vec2; i.e., all bits of vec1 should be
	in vec2. """
	cdef int a
	for a in range(slots):
		if (vec1[a] & vec2[a]) != vec1[a]: return False
	return True

