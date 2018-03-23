Coding style
------------

Code should follow  `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_,
except for the differences described below.
Part of these conventions can be checked automatically with ``make lint``;
this requires ``pip3 install pep8 pylint``.

Names
~~~~~
- Variable and function names in lower case preferably without underscores: ``examplevariable``.
- Classes ``CapitalizedWithoutUnderscores``.
- Globals in ``ALLCAPS`` (preferably used for constants).

Formatting
~~~~~~~~~~
- Tabs rather than spaces for indentation (with tabs set to display as 4
  spaces). Lines end with a linefeed (UNIX format).
- Strict 80 characters per line limit; only exception is where this hurts
  usability such as with long URLs.
- Spread long statements over several lines by leaving an open
  parenthesis/bracket/brace at the end of the first line and indent the following
  line(s) with an extra level of indentation (tabs, not spaces) to make the
  continued line stand out from regular indented statements, e.g.:

  .. code:: python

      if (longcondition
              and anothercondition
              or yetanothercondition(
                  verylongargument)):
          dosomething()

  Do not align statements or expressions with spaces or tabs to avoid noisy
  diffs. Do not use backslashes to continue long lines, except in strings.

Documentation
~~~~~~~~~~~~~
- Every file, class, method, and function should start with a docstring with
  at least a one-line description that fits on one line and ends with a period.
  Further documentation may follow after an empty line. The opening and
  closing triple quotes ``"""`` do not have to be on their own line.
- Docstrings are formatted with
  `reStructuredText <http://www.sphinx-doc.org/en/stable/rest.html>`_.
  Parameters may be documented with `Sphinx docstrings
  <http://www.sphinx-doc.org/en/stable/domains.html#info-field-lists>`_.
  Simple usage examples and behavior can be documented and tested with
  `doctests <https://docs.python.org/3.6/library/doctest.html>`_;
  more elaborate tests should go in ``tests/unittests.py``.
  Example:

  .. code:: python

      def foo(bar):
          """Apply foo transform to bar.

          :param bar: a string representation of bar.
          :returns: a list of foo strings.

          >>> foo('zed')
          ['z', 'e', 'd']"""
          ...

- Minimal comments; comments should be a last resort to explain important
  non-obvious details. Most code should be self-explanatory through
  descriptive variable names, clear structure, etc.

String literals
~~~~~~~~~~~~~~~
- Prefer single quotes ``'``, except when double quotes ``"``
  are more convenient. Similarly, use raw strings when this results in less
  need for escaping.
- The preferred format for long strings is implicit concatenation, since such
  strings can be properly indented without affecting the string itself:

  .. code:: python

      msg = ('A long error message, '
            'spread over two lines:\n%s' % err)

  But note that this requires newlines to be added explicitly.

  .. code:: python
      msg = '''\
      Alternatively, use a triple-quoted string, with a backslash for
      continuing long lines.'''
