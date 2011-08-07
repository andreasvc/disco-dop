from math import exp, log
from array import array
from collections import defaultdict
from heapq import heappush, heappop, heapify

print "plcfrs in shedskin mode"

def parse(sent, grammar, tags, start, exhaustive):
    """ parse sentence, a list of tokens, optionally with gold tags, and
    produce a chart, either exhaustive (n=0) or up until the viterbi parse
    """
    unary = grammar.unary
    lbinary = grammar.lbinary
    rbinary = grammar.rbinary
    lexical = grammar.lexical
    toid = grammar.toid
    tolabel = grammar.tolabel
    if start == -1: start = toid['S']
    goal = ChartItem(start, (1 << len(sent)) - 1)
    m = maxA = 0
    Cx = [{} for _ in toid]
    C = {}
    A = agenda()

    # scan: assign part-of-speech tags
    Epsilon = toid["Epsilon"]
    for i, w in enumerate(sent):
        recognized = False
	for terminal in lexical.get(w, []):
	    if not tags or tags[i] == tolabel[terminal.lhs].split("@")[0]:
		Ih = ChartItem(terminal.lhs, 1 << i)
		I = ChartItem(Epsilon, i)
		z = terminal.prob
		A[Ih] = Edge(z, z, z, I, None)
		C[Ih] = []
		recognized = True
        if not recognized and tags and tags[i] in toid:
            Ih = ChartItem(toid[tags[i]], 1 << i)
            I = ChartItem(Epsilon, i)
            A[Ih] = Edge(0.0, 0.0, 0.0, I, None)
            C[Ih] = []
            recognized = True
            continue
        elif not recognized:
            print "not covered:", tags[i] if tags else w
            return C, None

    # parsing
    while A:
        Ih, edge = A.popitem()
        C[Ih].append(edge)
        Cx[Ih.label][Ih] = edge

        if Ih == goal:
            if exhaustive: continue
            else: break 
        for I1h, edge in deduced_from(Ih, edge.inside, Cx,
                    unary, lbinary, rbinary):
            if I1h not in C and I1h not in A:
                # haven't seen this item before, add to agenda
                A[I1h] = edge
                C[I1h] = []
            elif I1h in A and edge < A[I1h]:
                # either item has lower score, update agenda
                C[I1h].append(A[I1h])
                A[I1h] = edge
            else:
                # or extend chart
                C[I1h].append(edge)

        maxA = max(maxA, len(A))

    print "max agenda size", maxA, "/ chart keys", len(C),
    print "/ values", sum(map(len, C.values()))
    if goal not in C:
        C = {}
        goal = None
    return (C, goal)

def deduced_from(Ih, x, Cx, unary, lbinary, rbinary):
    I, Ir = Ih.label, Ih.vec
    result = []
    for rule in unary[I]:
        result.append((ChartItem(rule.lhs, Ir),
            Edge(x+rule.prob, x+rule.prob, rule.prob, Ih, None)))
    for rule in lbinary[I]:
        for I1h, edge in Cx[rule.rhs2].iteritems():
            if concat(rule, Ir, I1h.vec):
                result.append((ChartItem(rule.lhs, Ir ^ I1h.vec),
                    Edge(x+edge.inside+rule.prob, x+edge.inside+rule.prob,
                            rule.prob, Ih, I1h)))
    for rule in rbinary[I]:
        for I1h, edge in Cx[rule.rhs1].iteritems():
            if concat(rule, I1h.vec, Ir):
                result.append((ChartItem(rule.lhs, I1h.vec ^ Ir),
                    Edge(x+edge.inside+rule.prob, x+edge.inside+rule.prob,
                            rule.prob, I1h, Ih)))
    return result

def concat(rule, lvec, rvec):
    if lvec & rvec: return False
    lpos = nextset(lvec, 0)
    rpos = nextset(rvec, 0)
    #this algorithm taken from rparse FastYFComposer.
    for x in range(len(rule.args)):
        m = rule.lengths[x] - 1
        for n in range(m + 1):
            if testbit(rule.args[x], n):
                # check if there are any bits left, and
                # if any bits on the right should have gone before
                # ones on this side
                if rpos == -1 or (lpos != -1 and lpos <= rpos):
                    return False
                # jump to next gap
                rpos = nextunset(rvec, rpos)
                if lpos != -1 and lpos < rpos:
                    return False
                # there should be a gap if and only if
                # this is the last element of this argument
                if n == m:
                    if testbit(lvec, rpos):
                        return False
                elif not testbit(lvec, rpos):
                    return False
                #jump to next argument
                rpos = nextset(rvec, rpos)
            else: #if bit == 0:
                # vice versa to the above
                if lpos == -1 or (rpos != -1 and rpos <= lpos):
                    return False
                lpos = nextunset(lvec, lpos)
                if rpos != -1 and rpos < lpos:
                    return False
                if n == m:
                    if testbit(rvec, lpos):
                        return False
                elif not testbit(rvec, lpos):
                    return False
                lpos = nextset(lvec, lpos)
            #else: raise ValueError("non-binary element in yieldfunction")
    if lpos != -1 or rpos != -1:
        return False
    # everything looks all right
    return True

def mostprobablederivation(chart, start, tolabel):
    """ produce a string representation of the viterbi parse in bracket
    notation"""
    edge = min(chart[start])
    return getmpd(chart, start, tolabel), edge.inside

def getmpd(chart, start, tolabel):
    edge = min(chart[start])
    if edge.right and edge.right.label: #binary
        return "(%s %s %s)" % (tolabel[start.label],
                    getmpd(chart, edge.left, tolabel),
                    getmpd(chart, edge.right, tolabel))
    else: #unary or terminal
        return "(%s %s)" % (tolabel[start.label],
                    getmpd(chart, edge.left, tolabel)
                        if edge.left.label else str(edge.left.vec))

def binrepr(a, sent):
    return bin(a.vec)[2:].rjust(len(sent), "0")[::-1]

def pprint_chart(chart, sent, tolabel):
    print "chart:"
    for n, a in sorted((bitcount(a.vec), a) for a in chart):
	if not chart[a]: continue
        print "%s[%s] =>" % (tolabel[a.label], binrepr(a, sent))
        for edge in chart[a]:
            print "%g\t%g" % (exp(-edge.inside), exp(-edge.prob)),
            if edge.left.label:
                print "\t%s[%s]" % (tolabel[edge.left.label],
                                    binrepr(edge.left, sent)),
            else:
                print "\t", repr(sent[edge.left.vec]),
            if edge.right:
                print "\t%s[%s]" % (tolabel[edge.right.label],
                                    binrepr(edge.right, sent)),
            print
        print

def do(sent, grammar):
    print "sentence", sent
    chart, start = parse(sent.split(), grammar, None, -1, False)
    pprint_chart(chart, sent.split(), grammar.tolabel)
    if chart:
        #for t, p in mostprobableparse(chart, start, grammar.tolabel,
        #                                                   n=1000).items():
        #   print p, t
        t, p = mostprobablederivation(chart, start, grammar.tolabel)
        print exp(-p), t, '\n'
        return True
    else:
        print "no parse"
        return False

# bit operations
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

class Grammar(object):
    __slots__ = ('unary', 'lbinary', 'rbinary', 'lexical',
                    'bylhs', 'toid', 'tolabel')
    def __init__(self, unary, lbinary, rbinary, lexical, bylhs, toid, tolabel):
        self.unary = unary
        self.lbinary = lbinary
        self.rbinary = rbinary
        self.lexical = lexical
        self.bylhs = bylhs
        self.toid = toid
        self.tolabel = tolabel

class ChartItem:
    __slots__ = ("label", "vec", "_hash")
    def __init__(self, label, vec):
        self.label = label      #the category of this item (NP/PP/VP etc)
        self.vec = vec          #bitvector describing the spans of this item
        self._hash = hash((self.label, self.vec))
    def __hash__(self):
        return self._hash
    def __eq__(self, other):
        if other is None: return False
        return self.label == other.label and self.vec == other.vec
    def __repr__(self):
        return "ChartItem(%s, %s)" % (self.label, bin(self.vec))

class Edge:
    __slots__ = ('score', 'inside', 'prob', 'left', 'right', '_hash')
    def __init__(self, score, inside, prob, left, right):
        self.score = score; self.inside = inside; self.prob = prob
        self.left = left; self.right = right
        self._hash = hash(((inside, prob), (left, right)))
    def __hash__(self):
        return self._hash
    def __lt__(self, other):
        # the ordering only depends on inside probability
        # (or on estimate of outside score when added)
        return self.score < other.score
    def __le__(self, other):
        return self.score <= other.score
    def __ne__(self, other):
        return not self.__eq__(other)
    def __eq__(self, other):
        return (self.inside == other.inside
                and self.prob == other.prob
                and self.left == other.right
                and self.right == other.right)
    def __gt__(self, other):
        return self.score > other.score
    def __ge__(self, other):
        return self.score >= other.score
    def __repr__(self):
        return "Edge(%g, %g, %g, %r, %r)" % (self.score, self.inside,
                    self.prob,
                    self.left,
                    self.right)

class Terminal:
    __slots__ = ('lhs', 'rhs1', 'rhs2', 'word', 'prob')
    def __init__(self, lhs, rhs1, rhs2, word, prob):
        self.lhs = lhs; self.rhs1 = rhs1; self.rhs2 = rhs2
        self.word = word; self.prob = prob

class Rule:
    __slots__ = ('lhs', 'rhs1', 'rhs2', 'prob',
                'args', 'lengths', '_args', 'lengths')
    def __init__(self, lhs, rhs1, rhs2, args, lengths, prob):
        self.lhs = lhs; self.rhs1 = rhs1; self.rhs2 = rhs2
        self.args = args; self.lengths = lengths; self.prob = prob
        self._args = self.args; self._lengths = self.lengths

#the agenda (priority queue)
class Entry(object):
    __slots__ = ('key', 'value', 'count')
    def __init__(self, key, value, count):
        self.key = key          #the `task'
        self.value = value      #the priority
        self.count = count      #unqiue identifier to resolve ties
    def __eq__(self, other):
        return self.count == other.count
    def __lt__(self, other):
        return self.value < other.value or (self.value == other.value
                and self.count < other.count)
    def __le__(self, other):
        return self.value < other.value or (self.value == other.value
                and self.count <= other.count)
    def __hash__(self):
        return hash(((self.key, self.value), self.count))

INVALID = 0
class agenda(object):
    def __init__(self, iterable=None):
        self.heap = []                      # the priority queue list
        self.counter = 1                    # unique sequence count
        self.mapping = {}                   # mapping of keys to entries
    #   if iterable:
    #       self.heap = [Entry(k, v, n + 1)
    #                       for n, (k,v) in enumerate(dict(iterable).items())]
    #       heapify(self.heap)
    #       self.mapping = dict((entry.key, entry) for entry in self.heap)
    #       self.counter += len(self.heap)

    def __setitem__(self, key, value):
        if key in self.mapping:
            oldentry = self.mapping[key]
            entry = Entry(key, value, oldentry.count)
            self.mapping[key] = entry
            heappush(self.heap, entry)
            oldentry.count = INVALID
        else:
            entry = Entry(key, value, self.counter)
            self.counter += 1
            self.mapping[key] = entry
            heappush(self.heap, entry)
        return None

    def __getitem__(self, key):
        return self.mapping[key].value

    def __delitem__(self, key):
        self.mapping.pop(key).count = INVALID

    def __contains__(self, key):
        return key in self.mapping

    def __len__(self):
        return len(self.mapping)

    def popitem(self):
        entry = heappop(self.heap)
        try: del self.mapping[entry.key]
        except KeyError: pass
        while entry.count is INVALID:
            entry = heappop(self.heap)
            try: del self.mapping[entry.key]
            except KeyError: pass
        return entry.key, entry.value

def main():
    '''
    grammar = splitgrammar(
        [((('S','VP2','VMFIN'),    ((0,1,0),)),  0.0),
        ((('VP2','VP2','VAINF'),  ((0,),(0,1))), log(0.5)),
        ((('VP2','PROAV','VVPP'), ((0,),(1,))), log(0.5)),
        ((('PROAV', 'Epsilon'), ('Daruber', ())), 0.0),
        ((('VAINF', 'Epsilon'), ('werden', ())), 0.0),
        ((('VMFIN', 'Epsilon'), ('muss', ())), 0.0),
        ((('VVPP', 'Epsilon'), ('nachgedacht', ())), 0.0)])
    '''
    # output of splitgrammar:
    unary=[[], [], [], [], [], [],
        [Rule(6, 6, 0, array('I', [3L]), array('H', [2]), -log(0.1))], []]
    lbinary=[[], [], [Rule(6, 2, 7, array('I', [0L, 1L]), array('H', [1, 1]),
        -log(0.5))], [], [], [], [Rule(3, 6, 5, array('I', [2L]), array('H',
        [3]), 0.0), Rule(6, 6, 4, array('I', [0L, 2L]), array('H', [1, 2]),
        -log(0.5))], []]
    rbinary=[[], [], [], [], [Rule(6, 6, 4, array('I', [0L, 2L]), array('H',
        [1, 2]), -log(0.5))], [Rule(3, 6, 5, array('I', [2L]), array('H', [3]),
        0.0)], [], [Rule(6, 2, 7, array('I', [0L, 1L]), array('H', [1, 1]),
        -log(0.5))]]
    lexical={'muss': [Terminal(5, 0, 0, 'muss', 0.0)],
            'werden': [Terminal(4, 0, 0, 'werden', 0.0)],
            'Daruber': [Terminal(2, 0, 0, 'Daruber', 0.0)],
            'nachgedacht': [Terminal(7, 0, 0, 'nachgedacht', 0.0)]}
    bylhs=[[], [], [], [Rule(3, 6, 5, array('I', [2L]), array('H', [3]), 0.0)],
        [], [], [Rule(6, 6, 4, array('I', [0L, 2L]), array('H', [1, 2]),
        -log(0.5)), Rule(6, 2, 7, array('I', [0L, 1L]), array('H', [1,
        1]), -log(0.5))], []] 
    toid = {'VP2': 6, 'Epsilon': 0, 'VVPP': 7, 'S': 3, 'VMFIN': 5, 'VAINF': 4,
        'ROOT': 1, 'PROAV': 2 }
    tolabel = {0: 'Epsilon', 1: 'ROOT', 2: 'PROAV', 3: 'S', 4: 'VAINF', 5:
        'VMFIN', 6: 'VP2', 7: 'VVPP'}
    grammar = Grammar(unary, lbinary, rbinary, lexical, bylhs, toid, tolabel)

    chart, start = parse("Daruber muss nachgedacht werden".split(),
            grammar, "PROAV VMFIN VVPP VAINF".split(), toid['S'], False)
    pprint_chart(chart, "Daruber muss nachgedacht werden".split(), tolabel)
    assert (mostprobablederivation(chart, start, tolabel) ==
        ('(S (VP2 (VP2 (PROAV 0) (VVPP 2)) (VAINF 3)) (VMFIN 1))', -log(0.25)))
    assert do("Daruber muss nachgedacht werden", grammar)
    assert do("Daruber muss nachgedacht werden werden", grammar)
    assert do("Daruber muss nachgedacht werden werden werden", grammar)
    print "ungrammatical sentence:"
    assert not do("muss Daruber nachgedacht werden", grammar)
    print "(as expected)\n"
    print 'it worked'

if __name__ == '__main__': 
    main()
