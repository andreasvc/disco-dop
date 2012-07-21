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
	ULong TESTBIT(ULong a[], int b)

# cpdef functions defined in bit.pyx
#on python integers
cpdef inline int fanout(arg)
cpdef inline int pyintnextset(a, int pos)

#on ULLongs
cpdef inline int bitcount(ULLong vec)
cpdef inline bint bitminmax(ULLong a, ULLong b)
cpdef inline bint testbitc(UChar arg, UInt pos)
cpdef inline bint testbitshort(unsigned short arg, UInt pos)

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
	1
	>>> testbit(0b0011101, 1)
	0"""
	return (vec >> pos) & 1

cdef inline bint testbitint(UInt arg, UInt pos):
	""" Mask a particular bit, return nonzero if set
	>>> testbit(0b0011101, 0)
	1
	>>> testbit(0b0011101, 1)
	0"""
	return (arg >> pos) & 1

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

cdef inline int anextset(ULong *vec, UInt pos,
	UInt slots):
	""" return next set bit starting from pos, -1 if there is none. """
	cdef UInt a = BITSLOT(pos), offset = pos % BITSIZE
	if vec[a] >> offset: return pos + __builtin_ctzl(vec[a] >> offset)
	for a in range(a + 1, slots):
		if vec[a]: return a * BITSIZE + __builtin_ctzl(vec[a])
	return -1

cdef inline int anextunset(ULong *vec, UInt pos,
	UInt slots):
	""" return next unset bit starting from pos. """
	cdef UInt a = BITSLOT(pos), offset = pos % BITSIZE
	if ~(vec[a] >> offset):
		return pos + __builtin_ctzl(~(vec[a] >> offset))
	a += 1
	while vec[a] == ~0UL: a += 1
	return a * BITSIZE + __builtin_ctzl(~(vec[a]))

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

cdef inline void setunion(ULong *dest, ULong *src, UInt slots):
	""" dest gets the union of dest and src; both operands must have at least
	`slots' slots. """
	cdef int a
	for a in range(slots): dest[a] |= src[a]

cdef inline bint subset(ULong *vec1, ULong *vec2, UInt slots):
	""" test whether vec1 is a subset of vec2; i.e., all bits of vec1 should be
	in vec2. """
	cdef int a
	for a in range(slots):
		if (vec1[a] & vec2[a]) != vec1[a]: return False
	return True

