Paleontologist Documentation
============================

What is paleontologist?
=======================
Paleontologist is a modular python interface to read and analyse Mastodon (FIJI Plugin) .csv and .xml data.

[Mastodon](https://github.com/fiji/TrackMate3) is a large-scale tracking and track-editing framework for large, multi-view images. It allows you to track cells' dynamics over time and has a very useful and easy to use GUI. In order to use Mastodon, since it works with [Big Data Viewer](https://github.com/bigdataviewer), you need your data to be in HDF5/xml format. 

As an output, Mastodon provides either a -mamut.xml or a .csv file which containes, among many features, the XYZ coordinates of each cell and the 3D average intensity of the cells. This package provides the  tools to facilitate the organization of the data and enable the easy creation of figures for spatial, temporal and mitotic dynamics of the cells. 

Preparing your data for tracking
--------------------------------

[1] Conversion to HDF5 and XML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Before using Mastodon, you need to convert your files in a format that BigData viewer can read. For this, using either [Big Data Viewer](https://github.com/bigdataviewer), [BigStitcher](https://imagej.net/BigStitcher) or [Multiview Reconstruction](https://imagej.net/Multiview-Reconstruction) from Fiji, you can convert your data into HDF5 and XML. HDF5 will save the raw data whereas the XML file will save the metadata and any transformation performed to the raw data. 

[2] Time registration or deconvolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once your data is in the file format supported by BigDataViewer, there is a range of different Fiji Plugins which perform time registration, deconvolution, stitching... using the XML-HDF5 file format by using BigDataViewer for rendering the data. 

Cell tracking with Mastodon
---------------------------

[3] Using Mastodon
~~~~~~~~~~~~~~~~~~~
[Mastodon](https://github.com/fiji/TrackMate3) is a very user-friendly Tracking plugin from Fiji. It allows interactive visualization and navigation of large images thanks to the BigDataViewer. Any file that can be opened in the BigDataViewer will work in Mastodon (BDV HDF5 file format, KLB, Keller-Lab Blocks file format, N5 file format, ...). 

With Mastodon you will be able to track large amount of cells in a manual, semi-automatic or automatic way. The outputs from the tracking are two .csv files: name-edges.csv and name-vertices.csv . The first one contains the information obtained from the spots: mean, median and standard deviation of intensity of all the channels; x, y, z coordinates of the centroid of the spots; spots radius; detection quality for each spot; tags and sub-tags for the spots; the individual ID for each spot; the track ID to which each spot corresponds. 

Data analysis using paleontologist
----------------------------------

[4] Upload the data and start exploring the notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Paleontologist is a collection of notebooks which allow the user to organize the cell tracks and perform analysis on the temporal and spatial domain. Under the hood, there are 3 functions called: ``mastodon_functions.py``, ``paleo_functions.py`` and ``paleo.py``. Each contains classes to help tidy the cell tracks and perform analysis on these tracks. 

``mastodon_functions.py``

``paleo_functions.py``

``paleo.py``

Here is some text explaining some complicated stuff.::

   print 'hello'
   >> hello


Guide
^^^^^


.. toctree::
   :maxdepth: 3
   :caption: Contents:
   
   help


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
