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

#define SETUNION(a, b, i, slots)	(for(i=0; i<slots; i++) a[i] |= b[i])

/* 3D array indexing (implicit third index k = 0) */
#define IDX(i,j,jmax,kmax)		((i * jmax + j) * kmax)
