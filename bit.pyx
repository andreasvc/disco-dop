# See: http://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html

cpdef inline int nextset(ULLong vec, UInt pos):
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

cpdef inline int nextunset(ULLong vec, UInt pos):
	""" Return next unset bit starting from pos.
	>> nextunset(0b001101, 2)
	4
	>> nextunset((1<<64)-1, 0)
	64
	"""
	cdef ULLong x = ~vec & (~0ULL << pos)
	return __builtin_ctzll(x) if x else (sizeof(ULLong) * 8)

cpdef inline int bitcount(ULLong vec):
	""" Number of set bits (1s)
	>>> bitcount(0b0011101)
	4
	"""
	return __builtin_popcountll(vec)

def pyintbitcount(a):
	""" Number of set bits (1s)
	>>> bitcount(0b0011101)
	4
	"""
	count = 0
	while a:
		a &= a - 1
		count += 1
	return count

cpdef inline int pyintnextset(a, int pos):
	""" First set bit, starting from pos
	>>> nextset(0b001101, 1)
	2
	"""
	cdef ULong mask = -1
	a >>= pos
	if a == 0: return -1
	while a & mask == 0:
		a >>= (8*sizeof(ULong))
		pos += (8*sizeof(ULong))
	return pos + __builtin_ctzl(a & mask)

cpdef inline int bitlength(ULLong vec):
	""" number of bits needed to represent vector
	(equivalently: index of most significant set bit, plus one)
	>>> bitlength(0b0011101)
	5"""
	return sizeof(vec) * 8 - __builtin_clzll(vec)

cpdef inline bint bitminmax(ULLong a, ULLong b):
	""" test whether the leftmost bit of b is adjacent to the first component
	of a. """
	return nextset(b, 0) == nextunset(a, nextset(a, 0))

cpdef inline int fanout(arg):
	""" number of contiguous components in bit vector (gaps plus one)
	>>> fanout(0b011011011)
	3"""
	cdef UInt result = 0
	cdef ULLong vec
	if arg < ((1 << 63) - 1):
		vec = arg
		while vec:
			vec >>= __builtin_ctzll(vec)
			vec >>= __builtin_ctzll(~vec)
			result += 1
	else:
		# when the argument does not fit in a 64-bit int, resort to plain
		# Python code because we need Python's bigints. This algorithm is based
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

cpdef inline bint testbit(ULLong vec, UInt pos):
	""" Mask a particular bit, return nonzero if set
	>>> testbit(0b0011101, 0)
	1
	>>> testbit(0b0011101, 1)
	0"""
	return (vec >> pos) & 1

# todo: see if this can be turned into a macro
cpdef inline bint testbitc(UChar arg, UInt pos):
	""" Mask a particular bit, return nonzero if set
	>>> testbit(0b0011101, 0)
	1
	>>> testbit(0b0011101, 1)
	0"""
	return (arg >> pos) & 1

cpdef inline bint testbitshort(unsigned short arg, UInt pos):
	""" Mask a particular bit, return nonzero if set
	>>> testbit(0b0011101, 0)
	1
	>>> testbit(0b0011101, 1)
	0"""
	return (arg >> pos) & 1

cpdef inline bint testbitint(UInt arg, UInt pos):
	""" Mask a particular bit, return nonzero if set
	>>> testbit(0b0011101, 0)
	1
	>>> testbit(0b0011101, 1)
	0"""
	return (arg >> pos) & 1


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

cdef inline void ulongset(ULong *dest, ULong value,
	UInt slots):
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

def main():
	print "8 * sizeof(unsigned long) ", 8 * sizeof(unsigned long)
	print "8 * sizeof(unsigned long long) ", 8 * sizeof(unsigned long long)
	assert nextset(0b001100110, 3) == 5
	assert nextset(0b001100110, 7) == -1
	assert nextunset(0b001100110, 1) == 3
	assert nextunset(~0ULL, 0) == 64
	assert bitcount(0b001100110) == 4
	assert testbit(0b001100110, 1)
	assert not testbit(0b001100110, 3)
	assert fanout(0b0111100) == 1
	assert fanout(0b1000001) == 2
	assert fanout(0b011011011) == 3
	print "it worked"
	from doctest import testmod, NORMALIZE_WHITESPACE, ELLIPSIS
	#from cydoctest import testmod
	# do doctests, but don't be pedantic about whitespace (I suspect it is the
	# militant anti-tab faction who are behind this obnoxious default)
	optionflags=NORMALIZE_WHITESPACE | ELLIPSIS
	fail, attempted = testmod(verbose=False, optionflags=optionflags)
	if attempted and not fail:
		print "%s: %d doctests succeeded!" % (__file__, attempted)
	else: print "attempted", attempted, "fail", fail

if __name__ == '__main__': main()
