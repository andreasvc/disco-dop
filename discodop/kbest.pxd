from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp.string cimport string
from .containers cimport SmallChartItem, FatChartItem, Prob, Label, ItemNo, \
		Grammar, ProbRule, Chart, Edge, RankedEdge, RankedEdgeAgenda, \
		CFGtoSmallChartItem, CFGtoFatChartItem, sparse_hash_map, \
		RankedEdgeSet
from .pcfg cimport CFGChart, DenseCFGChart, SparseCFGChart
from .plcfrs cimport LCFRSChart, SmallLCFRSChart, FatLCFRSChart

ctypedef sparse_hash_map[ItemNo, RankedEdgeAgenda[Prob]] agendas_type

cdef string getderiv(ItemNo v, RankedEdge ej, Chart chart)
cdef collectitems(ItemNo v, RankedEdge& ej, Chart chart, itemset)
cdef bint derivhasitem(ItemNo v, RankedEdge& ej, Chart chart, ItemNo u)
