Usage
=====

.. _installation:

Installation
------------

To use paleontologist, first install it using pip:

.. code-block:: console

   (.venv) $ pip install paleontologist

Creating recipes
----------------

To retrieve a dictionary of the image features,
you can use the ``mastodon_functions.xml_features(path_xml)`` function:

.. autofunction:: mastodon_functions.xml_features

The ``path_xml`` parameter should be the directory to where you have the .xml file saved ``"path/to/xml"``. Otherwise, :py:func:`mastodon_functions.xml_features`
will raise an exception.

.. autoexception:: mastodon_functions.InvalidKindError

For example:

>>> import mastodon_functions
>>> fts = mastodon_functions.xml_features(path_xml)
>>> fts.x_pixel
0.347
>>> fts.width
2048

