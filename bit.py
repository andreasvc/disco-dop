# bit operations adapted from http://wiki.python.org/moin/BitManipulation
def nextset2(a, pos, bitlen):
	result = 0
	bitlen -= pos
	a >>= pos
	a = a & -a
	while a >> result and result < bitlen:
		result += 1
	return pos + result - 1 if result < bitlen else -1

def nextunset2(a, pos, bitlen):
	result = 0
	bitlen -= pos
	a >>= pos
	a = ~a & -~a
	while a >> result and result < bitlen:
		result += 1
	return pos + result - 1

def nextset(a, pos):
	result = pos
	while (not (a >> result) & 1) and a >> result:
		result += 1
	return result if a >> result else -1

def nextunset(a, pos):
	result = pos
	while (a >> result) & 1:
		result += 1
	return result

def testbit(a, offset):
	return a & (1 << offset)

def bitcount(a):
	count = 0
	while a:
		a &= a - 1
		count += 1
	return count

def bitminmax(a, b):
	"""test if the least and most significant bits of a and b are 
	consecutive. we shift a and b until they meet in the middle (return true)
	or collide (return false)"""
	b = (b & -b)
	while a and b:
		a >>= 1
		b >>= 1
	return b == 1
