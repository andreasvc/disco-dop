int nextset(unsigned long vec, int pos);
int nextunset(unsigned long vec, int pos);

// with gcc builtins:(more portable than assembly)
int nextset(unsigned long vec, int pos) {
	return (vec >> pos) ? pos + __builtin_ffsl(vec >> pos) - 1 : -1;
}
int nextunset(unsigned long vec, int pos) {
	return pos + __builtin_ffsl(~(vec >> pos)) - 1;
}

// this is fast with certain CPUs (in others it is microcode)
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
// [we can do it with unsigned long long to scale to 64 bits]:
// __builtin_ffsll()


// and with floating point operations
// [...]

// naive way
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
