#include <stdint.h>
#include <functional>
#include <utility>
#include <vector>
#include <queue>
#include <algorithm>

// more memory efficient hash tables, slightly slower insertions
#define SPP_ALLOC_SZ 1

#include "../sparsepp/sparsepp/spp.h"
// need to insert items in btree while iterating over it,
// so we need the safe version which supports this
#include "../cpp-btree/safe_btree_map.h"

// C++11 version of gheap is preferrable
#if __cplusplus >= 201103L
#define GHEAP_CPP11
#endif
#include "../gheap/gheap.hpp"
#include "../gheap/gpriority_queue.hpp"

/* The number of uint64_t elements in a FatChartItem bit vector. */
/* FIXME: check for 64 bits; also defined in constants.pxi */
#define SLOTS 2

typedef uint32_t ItemNo;
typedef uint32_t Label;
typedef double Prob;

// A stable priority queue with a key-value interface.
// Values are priorities (should be comparable),
// keeps only one priority per key (should be hashable).
// Internal heap may contain duplicates or invalidated items (cleaned up on
// pop), map tracks the canonical set of items and their best priorities.
template <typename Key,
		typename Value,
		typename KeyHasher = spp::spp_hash<Key> >
class Agenda {
public:
	struct Entry {
		Key key;
		Value value;
		uint32_t count;
		Entry() { };
		Entry(Key _key, Value _value, uint32_t _count):
			key(_key), value(_value), count(_count) { };
	};
	typedef Entry entry_type;
	typedef typename std::pair<Key, Value> item_type;

	Agenda() { counter = 0; }
	// The following three functions should be thought of as constructors
	// but more convenient in Cython to only use default constructor
	// (stack allocation).
	void reserve(size_t n) {
		map.reserve(n);
		heap.reserve(n);
	}
	// replace agenda with all items from vector
	// (does not consider priorities for duplicate keys)
	void replace_entries(typename std::vector<item_type> entries) {
		map = map_type(entries.size());
		counter = 0;
		for (typename std::vector<item_type>::iterator it=entries.begin();
				it != entries.end(); it++) {
			entry_type entry(it->first, it->second, ++counter);
			map[it->first] = entry;
		}
		// collect all non-duplicate items
		std::vector<entry_type> tmp;
		tmp.reserve(map.size());
		for (typename map_type::iterator it=map.begin();
				it != map.end(); it++) {
			tmp.push_back(it->second);
		}
		// heapify tmp vector
		heap = heap_type(tmp.begin(), tmp.end());
	}
	// replace agenda with up to k best items from vector
	// (only best priority kept in case of duplicates)
	void kbest_entries(
			typename std::vector<item_type> entries,
			size_t k) {
		map = map_type(k);
		counter = 0;
		// collect items, filtering out duplicates, adding counters
		for (typename std::vector<item_type>::iterator it=entries.begin();
				it != entries.end(); it++) {
			typename map_type::iterator x = map.find(it->first);
			if (x == map.end()) {
				entry_type entry(it->first, it->second, ++counter);
				map[it->first] = entry;
			} else if (it->second < x->second.value) {
				entry_type entry(it->first, it->second, x->second.count);
				x->second = entry;
			} // else: ignore duplicate item with lower/equal priority
		}
		// collect all non-duplicate items
		std::vector<entry_type> tmp;
		tmp.reserve(map.size());
		for (typename map_type::iterator it=map.begin();
				it != map.end(); it++) {
			tmp.push_back(it->second);
		}
		// select k-best items
		if (k < map.size()) {
			nth_element(tmp.begin(), tmp.begin() + k, tmp.end(),
					Cmp<entry_type>());
			// Plan A: remove items beyond top k (erase is probably slower)
			// for(typename std::vector<entry_type>::iterator
			// 		it=tmp.begin() + k; it != tmp.end(); it++) {
			// 	map.erase(it->first->first);
			// }
			tmp.resize(k);
			// Plan B: re-construct map with only top-k elements
			map.clear();
			for (typename std::vector<entry_type>::iterator it=tmp.begin();
					it != tmp.end(); it++) {
				map[it->key] = *it;
			}
		}
		// heapify tmp vector
		heap = heap_type(tmp.begin(), tmp.end());
	}
	bool member(Key k) { return map.find(k) != map.end(); }
	bool empty() { return map.empty(); }
	size_t size() { return map.size(); }
	void setitem(Key k, Value v) {
		// NB: unconditional set, three situations:
		// 1. the priority is better; better item will pop from heap first
		// 2. priority is equal: heap will have a duplicate, but map only
		//             has one, so other one is ignored.
		// 3. priority is worse: the better one will pop from heap first,
		//             so this item will be ignored.
		entry_type entry(k, v, ++counter);
		map[k] = entry;
		heap.push(entry);
	}
	void setifbetter(Key k, Value v) {
		// insert only when key is new or priority is better; worse or equal
		// priority is ignored, avoids polluting heap with bogus values.
		typename map_type::iterator x = map.find(k);
		if (x == map.end()) {
			entry_type entry(k, v, ++counter);
			map[k] = entry;
			heap.push(entry);
		} else if (v < x->second.value) {
			entry_type entry(k, v, x->second.count);
			x->second = entry;
			heap.push(entry);
		} // else: ignore duplicate item with lower/equal priority.
	}
	item_type pop() {
		item_type noitem;
		if (map.empty()) {
			return noitem;
		}
		typename map_type::iterator it;
		entry_type entry = heap.top();
		// skip entries which are no longer in the map; (could verify that
		// priority matches w/map, but heap gives us the best priority so we
		// are not interested in other priorities for this key)
		while (it = map.find(entry.key), it == map.end()
				// || it->second.value != entry.value
				) {
			heap.pop();
			if (heap.empty()) {
				return noitem;
			}
			entry = heap.top();
		}
		heap.pop();
		map.erase(entry.key);
		return item_type(entry.key, entry.value);
	}
private:
	// a comparison function that considers the priority and count, not the key.
	template <typename entry_type> class Cmp {
		public:
		size_t operator()(const entry_type& k1, const entry_type& k2) const {
			return k1.value < k2.value || (
					k1.value == k2.value && k1.count < k2.count);
		}
	};
	// a reversed version, because priority_queue is a max-heap by default.
	template <typename entry_type> class CmpRev {
		public:
		size_t operator()(const entry_type& k1, const entry_type& k2) const {
			return k1.value > k2.value || (
					k1.value == k2.value && k1.count > k2.count);
		}
	};
	typedef spp::sparse_hash_map<Key, entry_type, KeyHasher> map_type;
	typedef gpriority_queue<gheap<4, 1>, entry_type,
			std::vector<entry_type>, CmpRev<entry_type> > heap_type;
	map_type map;
	heap_type heap;
	size_t counter;
};


struct ProbRule {  // total: 32 bytes.
    Prob prob;  // 8 bytes
    Label lhs;  // 4 bytes
    Label rhs1;  // 4 bytes
    Label rhs2;  // 4 bytes
    uint32_t args;  // 4 bytes => 32 max vars per rule
    uint32_t lengths;  // 4 bytes => same
    uint32_t no;  // 4 bytes
};

union Position {  // 8 bytes
	short mid;  // CFG, end index of left child
	uint64_t lvec;  // LCFRS, bit vector of left child
	size_t lidx;  // idx to FatLCFRSChartItem in chart.items[]
};

struct LexicalRule {
	// std::string word;
	Prob prob;
	Label lhs;
};

struct Edge {  // 16 bytes
    ProbRule *rule;  // ruleno takes less space than pointer, but not convenient
    Position pos;
};
class SmallChartItem {  // 96 bits
public:
	Label label;
	uint64_t vec;
	SmallChartItem() { };
	SmallChartItem(Label _label, uint64_t _vec):
		label(_label), vec(_vec) { };
	bool operator == (const SmallChartItem& k2) const {
		return (label == k2.label) && (vec == k2.vec);
	}
	bool operator < (const SmallChartItem& k2) const {
		return label < k2.label || (label == k2.label && vec < k2.vec);
	}
	bool operator > (const SmallChartItem& k2) const {
		return label > k2.label || (label == k2.label && vec > k2.vec);
	}
};
struct SmallChartItemHasher {
	/* Juxtapose bits of label and vec, rotating vec if > 33 words.

	64              32            0
	|               ..........label
	|vec[0] 1st half
	|               vec[0] 2nd half
	------------------------------- XOR */
	size_t operator()(const SmallChartItem& k) const {
		// return (uint64_t)k.label ^ (k.vec << 31) ^ (k.vec >> 31);
		size_t _hash = 0;
		spp::hash_combine(_hash, k.label);
		spp::hash_combine(_hash, k.vec);
		return _hash;
	}
};


class FatChartItem { // 140 bits when vec has 2 slots;
public:
	Label label;
	uint64_t vec[SLOTS];
	FatChartItem() { };
	FatChartItem(Label _label): label(_label) {
		memset(vec, 0, SLOTS * 8);
	};
	bool operator == (const FatChartItem& k2) const {
		return ((label != k2.label) ? 0 :
				memcmp(vec, k2.vec, SLOTS * 8) == 0);
	}
	bool operator < (const FatChartItem& k2) const {
		// return label < k2.label;
		return label < k2.label || (label == k2.label
				&& memcmp(vec, k2.vec, SLOTS * 8) < 0);
	}
	bool operator > (const FatChartItem& k2) const {
		// return label > k2.label;
		return label > k2.label || (label == k2.label
				&& memcmp(vec, k2.vec, SLOTS * 8) > 0);
	}
};
struct FatChartItemHasher {
	/* Juxtapose bits of label and vec.

	64              32            0
	|               ..........label
	|vec[0] 1st half
	|               vec[0] 2nd half
	|........ rest of vec .........
	------------------------------- XOR */
	size_t operator()(const FatChartItem& k) const {
		size_t _hash = 0, n;
		// _hash = k.label ^ k.vec[0] << 31 ^ k.vec[0] >> 31;
		// /* add remaining bits, byte for byte */
		// for (n=sizeof(k.vec[0]); n < SLOTS * 8; n++) {
		// 	_hash *= 33 ^ ((char *)k.vec)[n];
		// }
		spp::hash_combine(_hash, k.label);
		for (n=0; n < SLOTS; n++) {
			spp::hash_combine(_hash, k.vec[n]);
		}
		return _hash;
	}
};


// NB: a version of ProbRule without probability, rule number.
class Rule {  // total: 20 bytes.
public:
    Label lhs;  // 4 bytes
    Label rhs1;  // 4 bytes
    Label rhs2;  // 4 bytes
    uint32_t args;  // 4 bytes => 32 max vars per rule
    uint32_t lengths;  // 4 bytes => same
	bool operator == (const Rule& k2) const {
		return (lhs == k2.lhs
				&& rhs1 == k2.rhs1
				&& rhs2 == k2.rhs2
				&& args == k2.args
				&& lengths == k2.lengths);
	}
};
struct RuleHasher {
	size_t operator()(const Rule& k) const {
		size_t _hash = 0;
		spp::hash_combine(_hash, k.lhs);
		spp::hash_combine(_hash, k.rhs1);
		spp::hash_combine(_hash, k.rhs2);
		spp::hash_combine(_hash, k.args);
		spp::hash_combine(_hash, k.lengths);
		return _hash;
	}
};

class RankedEdge {
public:
	Edge edge;  // rule / spans of children
	ItemNo head;  // chart item of this node
	int left, right;  // rank of left / right child
	RankedEdge() { };
	RankedEdge(ItemNo _head, Edge _edge, int _left, int _right):
		edge(_edge), head(_head), left(_left), right(_right) { };
	bool operator == (const RankedEdge& k2) const {
		return (head == k2.head
				&& (edge.rule != NULL && k2.edge.rule != NULL
					? edge.rule->no == k2.edge.rule->no
					: edge.rule == k2.edge.rule)
				&& edge.pos.lvec == k2.edge.pos.lvec
				&& left == k2.left && right == k2.right);
	}
	bool operator < (__attribute__((unused)) const RankedEdge& other) const {
		return false;
	}
	bool operator > (__attribute__((unused)) const RankedEdge& other) const {
		return false;
	}
};
struct RankedEdgeHasher {
	size_t operator()(const RankedEdge& k) const {
		size_t _hash = 0;
		spp::hash_combine(_hash, k.head);
		spp::hash_combine(_hash, k.edge.rule == NULL ? -1 : k.edge.rule->no);
		spp::hash_combine(_hash, k.edge.pos.lvec);
		spp::hash_combine(_hash, k.left);
		spp::hash_combine(_hash, k.right);
		return _hash;
	}
};


/* instantiate templates with fixed Key types, because non-type parameters
 * cannot be passed from Cython. */
/* fix type of key, type of value V can be templated from Cython. */
class SmallChartItemSet : public spp::sparse_hash_set<SmallChartItem,
		SmallChartItemHasher> {};
class FatChartItemSet : public spp::sparse_hash_set<FatChartItem,
		FatChartItemHasher> {};
template<typename V>
class SmallChartItemBtreeMap : public btree::safe_btree_map<
		SmallChartItem, V> {};
template<typename V>
class FatChartItemBtreeMap : public btree::safe_btree_map<
		FatChartItem, V> {};
template<typename V>
class SmallChartItemAgenda : public Agenda<SmallChartItem, V,
		SmallChartItemHasher> {};
template<typename V>
class FatChartItemAgenda : public Agenda<FatChartItem, V,
		FatChartItemHasher> {};
template<typename V>
class RankedEdgeAgenda : public Agenda<RankedEdge, V,
		RankedEdgeHasher> {};
class RankedEdgeSet : public spp::sparse_hash_set<RankedEdge,
		RankedEdgeHasher> {};
template<typename V>
class RuleHashMap : public spp::sparse_hash_map<Rule, V, RuleHasher> {};
