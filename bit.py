# bit operations adapted from http://wiki.python.org/moin/BitManipulation
def nextset(a, pos):
	""" First set bit, starting from pos """
	result = pos
	while (not (a >> result) & 1) and a >> result:
		result += 1
	return result if a >> result else -1

def nextunset(a, pos):
	""" First unset bit, starting from pos """
	result = pos
	while (a >> result) & 1:
		result += 1
	return result

def bitcount(a):
	""" Number of set bits (1s) """
	count = 0
	while a:
		a &= a - 1
		count += 1
	return count

def testbit(a, offset):
	""" Mask a particular bit, return nonzero if set """
	return a & (1 << offset)

def bitminmax(a, b):
	"""test if the least and most significant bits of a and b are 
	consecutive. we shift a and b until they meet in the middle (return true)
	or collide (return false)"""
	b = (b & -b)
	while a and b:
		a >>= 1
		b >>= 1
	return b == 1

def main():
	assert nextset(0b001100110, 3) == 5
	assert nextunset(0b001100110, 1) == 3
	assert bitcount(0b001100110) == 4
	assert bitminmax(0b000011, 0b001100)
	assert testbit(0b001100110, 1)
	assert not testbit(0b001100110, 3)
	assert not bitminmax(0b000011, 0b111000)
	assert not bitminmax(0b001100, 0b000011)
	print 'it worked'

if __name__ == '__main__': main()
