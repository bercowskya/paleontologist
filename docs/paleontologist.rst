Documentation
=============

Paleontologist is a modular python interface which uses Jupyter Notebooks as a user-friendly tool to help perform analysis on cell tracking data without the need of coding skills. However, it also allows for those more experienced in coding to use Paleontologist to customize the results and even create new analysis. 

Because there are many different ways to perform tracking depending on the question of interest, Paleontologist is divided into 6 Jupyter Notebooks, each serving one purpose but which could be easily combine into 1 big analysis pipeline. Moreover, behind each notebook we have: ``paleo.py``, ``paleo_functions.py``, ``mastodon_functions.py`` and ``paleo_bulk.py``. The different classes to perform the analysis for each notebook and to arrange the tracking data is done inside these files, therefore under the hood! You can decided whether to use the interactive features of the notebooks so that you do not have to code a single line or, if you are motivated enough, at the end of each notebook we show you how you could start coding your own lines of code using the Paleontologist library. 



The ``AllTracks`` class
******************************
.. autoclass:: paleo.AllTracks
    :members:
    :undoc-members:
    :show-inheritance:

The ``IndividualTracks`` class
******************************
.. autoclass: paleo.IndividualTracks: 
    :members:
    :undoc-members:
    :show-inheritance:

The ``AllTracksTags`` class
***************************
.. autoclass:: paleo.AllTracksTags
    :members:
    :undoc-member:
    :show-inheritance:

