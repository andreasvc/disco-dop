# bit operations adapted from http://wiki.python.org/moin/BitManipulation
def nextset(a, pos):
	""" First set bit, starting from pos
	>>> nextset(0b001101, 1)
	2
	"""
	result = pos
	while (not (a >> result) & 1) and a >> result:
		result += 1
	return result if a >> result else -1

def nextunset(a, pos):
	""" First unset bit, starting from pos
	>>> nextunset(0b001101, 2)
	4
	"""
	result = pos
	while (a >> result) & 1:
		result += 1
	return result

def bitcount(a):
	""" Number of set bits (1s)
	>>> bitcount(0b0011101)
	4
	"""
	count = 0
	while a:
		a &= a - 1
		count += 1
	return count

def bitlength(a):
	""" number of bits required to represent a
	alternatively: index of most significant set bit plus one. 	
	>>> bitlength(0b0011101)
	5
	"""
	length = 0
	while a:
		a >>= 1
		length += 1
	return length

def testbit(a, offset):
	""" Mask a particular bit, return nonzero if set 
	>>> testbit(0b0011101, 0)
	1
	>>> testbit(0b0011101, 1)
	0
	"""
	return a & (1 << offset)

testbitint = testbitshort = testbitc = testbit

def bitminmax(a, b):
	"""test if the least and most significant bits of a and b are 
	consecutive. we shift a and b until they meet in the middle (return true)
	or collide (return false)"""
	b = (b & -b)
	while a and b:
		a >>= 1
		b >>= 1
	return b == 1

if __name__ == '__main__':
	assert nextset(0b001100110, 3) == 5
	assert nextunset(0b001100110, 1) == 3
	assert bitcount(0b001100110) == 4
	assert bitminmax(0b000011, 0b001100)
	assert testbit(0b001100110, 1)
	assert not testbit(0b001100110, 3)
	assert not bitminmax(0b000011, 0b111000)
	assert not bitminmax(0b001100, 0b000011)
	from doctest import testmod, NORMALIZE_WHITESPACE, ELLIPSIS
	# do doctests, but don't be pedantic about whitespace (I suspect it is the
	# militant anti-tab faction who are behind this obnoxious default)
	fail, attempted = testmod(verbose=False, optionflags=NORMALIZE_WHITESPACE | ELLIPSIS)
	if attempted and not fail: print "%s: %d doctests succeeded!" % (__file__, attempted)

