/* See: http://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html */

// with gcc builtins: (more portable than assembly)
#define NEXTSET(vec, pos)	(((vec) >> (pos)) ? \
	(pos) + __builtin_ctzll((vec) >> (pos)) : -1)
#define NEXTUNSET(vec, pos)	((pos) + __builtin_ctzll(~((vec) >> (pos))))

/* http://c-faq.com/misc/bitsets.html */
#define BITSIZE 			(8*sizeof(long))
#define BITMASK(b) 			(1UL << ((b) % BITSIZE))
#define BITSLOT(b) 			((b) / BITSIZE)
#define SETBIT(a, b) 		((a)[BITSLOT(b)] |= BITMASK(b))
#define CLEARBIT(a, b) 		((a)[BITSLOT(b)] &= ~BITMASK(b))
#define TESTBIT(a, b) 		((a)[BITSLOT(b)] & BITMASK(b))
#define BITNSLOTS(nb) 		((nb + BITSIZE - 1) / BITSIZE)

/* 3D array indexing (implicit third index k = 0) */
#define IDX(i,j,jmax,kmax)		((i * jmax + j) * kmax)




// this is fast with certain CPUs (in others it is microcode)
/*
int nextset2(unsigned long vec, int pos) {
	int lsb;
	asm("bsfl %1,%0" : "=r"(lsb) : "r"(vec >> pos));
	return (vec >> pos) ? lsb + pos : -1;
}

int nextunset2(unsigned long vec, int pos) {
	int lsb;
	asm("bsfl %1,%0" : "=r"(lsb) : "r"(~vec >> pos));
	return lsb + pos;
}
*/

// and with floating point operations
// floor(log(etc)) [...]

// naive way
/*
int nextset1(unsigned long a, int pos, int bitlen) {
	int result = pos;
	while((!((a >> result) & 1)) && result < bitlen)
		result++;
	return result < bitlen ? result : -1;
}

int nextunset1(unsigned long a, int pos, int bitlen) {
	int result = pos;
	while((a >> result) & 1 && result < bitlen)
		result++;
	return result;
}
*/
