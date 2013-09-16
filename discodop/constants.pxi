# The number of unsigned longs in the bitvector of a FatChartItem
# The maximum sentence length will be: SLOTS * sizeof(unsigned long) * 8
DEF SLOTS = 2

# The number of edges allocated at once
# (higher number means less overhead during allocation, but more space wasted)
DEF EDGES_SIZE = 100

# The arity of the heap. A typical heap is binary (2).
# Higher values result in a heap with a smaller depth,
# but increase the number of comparisons between siblings that need to be done.
DEF HEAP_ARITY = 4
