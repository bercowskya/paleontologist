# Paleontologist
Read and analyse Mastodon (FIJI Plugin) .csv and .xml data

[Mastodon](https://github.com/fiji/TrackMate3) is a large-scale tracking and track-editing framework for large, multi-view images. It allows you to track cells' dynamics over time and has a very useful and easy to use GUI. In order to use Mastodon, since it works with [Big Data Viewer](https://github.com/bigdataviewer), you need your data to be in HDF5/xml format. 

As an output, Mastodon provides either a -mamut.xml or a .csv file which containes, among many features, the XYZ coordinates of each cell and the 3D average intensity of the cells. This package provides the  tools to facilitate the organization of the data and enable the easy creation of figures for spatial, temporal and mitotic dynamics of the cells. 

## [1] Conversion to HDF5 and XML
Before using Mastodon, you need to convert your files in a format that BigData viewer can read. For this, using either [Big Data Viewer](https://github.com/bigdataviewer), [BigStitcher](https://imagej.net/BigStitcher) or [Multiview Reconstruction](https://imagej.net/Multiview-Reconstruction) from Fiji, you can convert your data into HDF5 and XML. HDF5 will save the raw data whereas the XML file will save the metadata and any transformation performed to the raw data. 

## [2] Using Mastodon
[Mastodon](https://github.com/fiji/TrackMate3) is a very user-friendly Tracking plugin from Fiji. It allows interactive visualization and navigation of large images thanks to the BigDataViewer. Any file that can be opened in the BigDataViewer will work in Mastodon (BDV HDF5 file format, KLB, Keller-Lab Blocks file format, N5 file format, ...). 

With Mastodon you will be able to track large amount of cells in a manual, semi-automatic or automatic way. The outputs from the tracking are two .csv files: name-edges.csv and name-vertices.csv . The first one contains the information obtained from the spots: mean, median and standard deviation of intensity of all the channels; x, y, z coordinates of the centroid of the spots; spots radius; detection quality for each spot; tags and sub-tags for the spots; the individual ID for each spot; the track ID to which each spot corresponds. 

## [3] Using MastodonAnalysis.py to analyze Mastodon data
MastodonAnalysis.py is a collection of classes that allows you to obtain tidy arrays with the tracks' and spots' features in an easy way. 

Below you can find a description for each of these classes:

### Class ```xml_features```:
Gets as input the .xml file from the initial conversion using either BigdataViewer, Bigstitcher or Multiview reconstruction to convert the files into HDF5/XML. 
This class can be called by using the following line of code:

```python
fts = xml_features(path_xml)
```
where ```path_xml``` has the directory where the path of the .xml and .hdf5 files are. This line of code saves the output of the class xml_features into the object ```fts```. Therefore, if you write ```fts.``` and then press **Tab** you will get all possible outcomes from this class. The list of these outcomes are:

* channels
* dimensions
* width
* height
* number of slices
* x,y,z pixel size
* coordinate units ($ \mu $ m, mm, etc.)

### Class ```csv_features```:
This class allows you to obtain all the information that comes in the .csv files (-vertices and -edges) that are generated with Mastodon once you have computed the features in Mastodon and saved the results in the .csv format. The file called name-vertices.csv contains all the information concerning each individual spot. The file called name-edges.csv contains all the information concerning the links of each spot. 

To call this class use the following line of code:
```python
spots = csv_features(path_csv, path_xml)
```
This line of code saves the output of the class xml_features into the object ```spots```. Therefore, if you write ```spots.``` and then press **Tab** you will get all possible outcomes from this class.

The list of the outcomes are:

* number of links per spot
* ID for each individual spot
* ID of the source spot
* ID of the target spot
* Frame
* Spot gaussian-filtered intensity for each channel
* Standard deviation for each channel
* Median for each channel
* X,Y,Z coordinate in the units from the .xml
* Track ID
* Total number of tracks
* Total number of spots
* Tags and subtags

### Class ```ordering_tracks```:
This class order tracks acoording to whether they divide or not 

### Class ```xml_reader```:

### Class ```peak_detection```:

### Class ```bulk_peak_analysis```:

## Dependencies
numpy 

matplotlib.pylab

pandas

scipy

xml.etree.ElementTree

untangle



