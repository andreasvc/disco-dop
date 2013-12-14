import os
import glob

try:
	os.mkdir('python')
	os.mkdir('cython')
except:
	pass

mods1 = [os.path.basename(a[:a.rindex('.')])
		for a in glob.glob('../discodop/*.py')]
mods1 = sorted(a for a in mods1 if a != '__init__')
for mod in mods1:
	with open('python/%s.rst' % mod, 'w') as out:
		out.write("""%s\n%s

.. automodule:: discodop.%s
   :members:
   :undoc-members:
   :show-inheritance:

""" % (mod, '=' * len(mod), mod))



mods2 = sorted(os.path.basename(a[:a.rindex('.')])
		for a in glob.glob('../discodop/*.pyx'))
for mod in mods2:
	with open('cython/%s.rst' % mod, 'w') as out:
		out.write("""%s\n%s

.. automodule:: discodop.%s
   :members:
   :undoc-members:
   :show-inheritance:

""" % (mod, '=' * len(mod), mod))


with open('python/index.rst', 'w') as out:
	out.write("""\
Python modules
==============

Contents:

.. toctree::
   :maxdepth: 4

%s""" % ''.join('   %s\n' % mod for mod in mods1))


with open('cython/index.rst', 'w') as out:
	out.write("""\
Cython modules
==============

Contents:

.. toctree::
   :maxdepth: 4

%s""" % ''.join('   %s\n' % mod for mod in mods2))



with open('api.rst', 'w') as out:
	out.write("""\
API documentation
=================
.. automodule:: discodop.__init__
    :members:
    :undoc-members:
    :show-inheritance:

Contents:

.. toctree::
   :maxdepth: 4

   python/index
   cython/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
""")
