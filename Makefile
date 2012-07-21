all:
	python setup.py build_ext --inplace

clean:
	rm -f *.c *.so

test: all sample2.export
	python -c "\
import bit; bit.main(); \
import demos; demos.main(); \
import kbest; kbest.main(); \
import parser; parser.main(); \
import grammar; grammar.main(); \
import treebank; treebank.main(); \
import estimates; estimates.main(); \
import _fragments; _fragments.main(); \
import coarsetofine; coarsetofine.main(); \
import treetransforms; treetransforms.test(); \
import disambiguation; disambiguation.main(); \
print 'ran all modules'"

sample2.export:
	wget http://www.ims.uni-stuttgart.de/projekte/TIGER/TIGERCorpus/annotation/sample2.export

debug:
	python-dbg setup.py build_ext --inplace --debug #--pyrex-gdb

testdebug: debug valgrind-python.supp
	valgrind --tool=memcheck --leak-check=full --num-callers=30 --suppressions=valgrind-python.supp python-dbg testall.py

valgrind-python.supp:
	wget http://codespeak.net/svn/lxml/trunk/valgrind-python.supp
