/* See: http://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html */

// with gcc builtins: (more portable than assembly)
inline int nextset(unsigned long long vec, unsigned int pos) {
	// return next set bit starting from pos, -1 if there is none.
	//return ((vec >> pos) > 0) * pos + __builtin_ffsl(vec >> pos) - 1;
	return (vec >> pos) ? pos + __builtin_ctzll(vec >> pos) : -1;
}

inline int nextunset(unsigned long long vec, unsigned int pos) {
	// return next unset bit starting from pos. there is always a next unset
	// bit, so no bounds checking.
	return pos + __builtin_ctzll(~(vec >> pos));
}

inline int bitminmax(unsigned long long a, unsigned long long b) {
	return nextset(b, 0) == nextunset(a, nextset(a, 0));
	//return (64 - __builtin_clzl(a)) == __builtin_ffsl(b);
}

inline int bitcount(unsigned long long vec) {
	/* number of set bits in vector */
	return __builtin_popcountll(vec);
}

inline int bitlength(unsigned long long vec) {
	/* number of bits needed to represent vector
	(equivalently: index of most significant set bit, plus one */
	return sizeof (vec) * 8 - __builtin_clzll(vec);
}

inline int testbit(unsigned long long vec, unsigned int pos) {
	return (vec >> pos) & 1;
}

inline int testbitc(unsigned char arg, unsigned int pos) {
	return (arg >> pos) & 1;
}

inline int testbitshort(unsigned short arg, unsigned int pos) {
	return (arg >> pos) & 1;
}

inline int testbitint(unsigned int arg, unsigned int pos) {
	return (arg >> pos) & 1;
}


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

/* http://c-faq.com/misc/bitsets.html */
#define BITSIZE 			(8*sizeof(long))
#define BITMASK(b) 			(1UL << ((b) % BITSIZE))
#define BITSLOT(b) 			((b) / BITSIZE)
#define SETBIT(a, b) 		((a)[BITSLOT(b)] |= BITMASK(b))
#define CLEARBIT(a, b) 		((a)[BITSLOT(b)] &= ~BITMASK(b))
#define TESTBIT(a, b) 		((a)[BITSLOT(b)] & BITMASK(b))
#define BITNSLOTS(nb) 		((nb + BITSIZE - 1) / BITSIZE)

#define IDX(i,j,jmax,kmax)		((i * jmax + j) * kmax)

inline int abitcount(unsigned long vec[], int slots) {
	/* number of set bits in vector */
	int a, result = __builtin_popcountl(vec[0]);
	for (a=1; a<slots; a++)
		result += __builtin_popcountl(vec[a]);
	return result;
}

inline int anextset(unsigned long vec[], unsigned int pos, int slots) {
	// return next set bit starting from pos, -1 if there is none.
	int a = BITSLOT(pos), offset = pos % BITSIZE;
	if (vec[a] >> offset) return pos + __builtin_ctzl(vec[a] >> offset);
	for (a=a+1; a<slots; a++)
		if (vec[a]) return a * BITSIZE + __builtin_ctzl(vec[a]);
	return -1;
}

inline void setunion(unsigned long vec1[], unsigned long vec2[], int slots) {
	// vec1 gets the union of vec1 and vec2; must have equal number of slots
	int a;
	for(a=0; a<slots; a++) vec1[a] |= vec2[a];
}

inline int subset(unsigned long vec1[], unsigned long vec2[], int slots) {
	// test whether vec1 is a subset of vec2;
	// i.e., all bits of vec1 should be in vec2
	int a;
	for (a=0; a<slots; a++)
		if ((vec1[a] & vec2[a]) != vec1[a]) return 0;
	return 1;
}

