inline int nextset(unsigned long vec, int pos);
inline int nextunset(unsigned long vec, int pos);

// with gcc builtins:(more portable than assembly)
inline int nextset(unsigned long vec, int pos) {
	// return next set bit starting from pos, -1 if there is none.
	//return (vec >> pos) ? pos + __builtin_ctzl(vec >> pos) : -1;
	return ((vec >> pos) > 0) * pos + __builtin_ffsl(vec >> pos) - 1;
}

inline int nextunset(unsigned long vec, int pos) {
	// return next unset bit starting from pos. there is always a next unset
	// bit, so no bounds checking __builtin_ctzl is undefined when input is
	// zero, in this case it means we should always leave the most significant
	// bit unused
	return pos + __builtin_ctzl(~vec >> pos);
}

inline int bitminmax(unsigned long a, unsigned long b) {
	return nextset(b, 0) == nextunset(a, nextset(a, 0));
	//return (64 - __builtin_clzl(a)) == __builtin_ffsl(b);
}

inline int bitcount(unsigned long vec) {
	return __builtin_popcountl(vec);
}

inline int testbit(unsigned long vec, int pos) {
	return vec & (1 << pos);
}

inline int testbitc(unsigned char arg, int pos) {
	return arg & (1 << pos);
}

inline int testbitshort(unsigned short arg, int pos) {
	return arg & (1 << pos);
}

// [we can do it with unsigned long long to scale to 64 bits]:
// __builtin_ffsll()


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

