int nextset(long vec, int pos);
int nextunset(long vec, int pos);

// this is fast with certain CPUs (in others it is microcode)
int nextset(long vec, int pos) {
    int lsb;
    asm("bsfl %1,%0" : "=r"(lsb) : "r"(vec >> pos));
	return (vec >> pos) ? lsb + pos : -1;
}
int nextunset(long vec, int pos) {
    int lsb;
    asm("bsfl %1,%0" : "=r"(lsb) : "r"(~vec >> pos));
	return lsb + pos;
}

// there is also a way to do it with gcc builtins,
// and with floating point operations

// naive way
int nextset1(long a, int pos, int bitlen) {
	int result = pos;
	while((!((a >> result) & 1)) && result < bitlen)
		result++;
	return result < bitlen ? result : -1;
}

int nextunset1(long a, int pos, int bitlen) {
	int result = pos;
	while((a >> result) & 1 && result < bitlen)
		result++;
	return result;
}
