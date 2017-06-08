# The number of unsigned longs in the bitvector of a FatChartItem.
# The maximum sentence length will be: SLOTS * sizeof(unsigned long) * 8
DEF SLOTS = 2
# These values give a maximum length of 128 bits on 32 and 64 bit systems.
# IF UNAME_MACHINE == 'x86_64':
# 	DEF SLOTS = 2
# ELSE:
# 	DEF SLOTS = 4

# context summary estimates
DEF SX = 1
DEF SXlrgaps = 2

# Any logprob above this is considered as infinity and pruned by plcfrs
DEF MAX_LOGPROB = 300.0

# The maximum length of the path to the root node and any terminal node.
# Prevents unary cycles from causing stack overflows in k-best extraction.
DEF MAX_DEPTH = 200
