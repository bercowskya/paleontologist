import numpy as np
import untangle
import pandas as pd
from io import BytesIO
import time
from xml.etree.ElementTree import ElementTree
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import itertools


class XMLParser:
    """
    Bulk analysis for paleo

    Attributes
    ----------
    obj1 : untangle object
           parsed xml file
    width : int
            width of the image in the x-axis
    height : int
             height of the image in the y-axis
    n_slices : int
               number of acquired planes in the h5
    x_pixel : float
              x-pixel size in the specified units
    y_pixel : float
              y-pixel size in the specified units
    z_pixel : float
              z-pixel size in the specified units


    Methods
    -------

    """

    def __init__(self, path_xml, path_mamut, print_val):

        """Constructor method
        """
        super().__init__()

        # Parse the two xml files: mamut file and the hdf5-xml file
        self.obj = untangle.parse(path_mamut)
        self.obj1 = untangle.parse(path_xml)

        # Timelapse features saved in the xml file when creating the HDF5-XML (the metadata)
        try:
            # If timelapse has more than 1 time-point, it will succeed this exception
            self.width = int(self.obj1.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].size.cdata.split()[0])
            self.height = int(self.obj1.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].size.cdata.split()[1])
            self.n_slices = int(self.obj1.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].size.cdata.split()[2])

            self.x_pixel = float(self.obj1.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].voxelSize.size.cdata.
                                 split()[0])
            self.y_pixel = float(self.obj1.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].voxelSize.size.cdata.
                                 split()[1])
            self.z_pixel = float(self.obj1.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].voxelSize.size.cdata.
                                 split()[2])
            self.units = self.obj1.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].voxelSize.unit.cdata

        except AttributeError:
            # If timelapse only has 1 time-point, this exception will catch it and solve the parsing issue
            self.width = int(self.obj1.SpimData.SequenceDescription.ViewSetups.ViewSetup.size.cdata.split()[0])
            self.height = int(self.obj1.SpimData.SequenceDescription.ViewSetups.ViewSetup.size.cdata.split()[1])
            self.n_slices = int(self.obj1.SpimData.SequenceDescription.ViewSetups.ViewSetup.size.cdata.split()[2])

            self.x_pixel = float(self.obj1.SpimData.SequenceDescription.ViewSetups.ViewSetup.voxelSize.size.cdata.
                                 split()[0])
            self.y_pixel = float(self.obj1.SpimData.SequenceDescription.ViewSetups.ViewSetup.voxelSize.size.cdata.
                                 split()[1])
            self.z_pixel = float(self.obj1.SpimData.SequenceDescription.ViewSetups.ViewSetup.voxelSize.size.cdata.
                                 split()[2])

            self.units = self.obj1.SpimData.SequenceDescription.ViewSetups.ViewSetup.voxelSize.unit.cdata

        # Number of channels
        self.channels = len(self.obj1.SpimData.SequenceDescription.ViewSetups.ViewSetup)

        # Obtain the number of time-points and catch error if there is only 1 or if the XML is corrupted
        while True:
            try:
                self.n_frames = len(self.obj1.SpimData.SequenceDescription.Timepoints.integerpattern.cdata.split())
                break
            except AttributeError:
                pass  # fallback to dict
            try:
                self.n_frames = int(self.obj1.SpimData.SequenceDescription.Timepoints.last.cdata.split()[0]) + 1
                break
            except KeyError:
                raise AttributeError("There is something wrong with the .xml file") from None

        self.n_tracks = len(self.obj.TrackMate.Model.AllTracks)

        # Print the results of the .xml (but only if requested 1, otherwise do not print 0)
        if print_val == 1:
            print(f'The image has the following dimensions (XYZC): {self.width}, {self.height}, {self.n_slices}, '
                  f'{self.channels}')
            print(f'There are {self.n_frames:d} frames in total.')
            print(f'Pixel Size: x = {self.x_pixel:.3g} {self.units:s}, y = {self.y_pixel:.3g} {self.units:s} z = '
                  f'{self.z_pixel:.2g} {self.units:s}')
            print(f'There are {self.n_slices:d} Z-slices in total')
            print('\n')

            # Print some information about the number of cells and tracks
            n_spots = int(self.obj.TrackMate.Model.AllSpots['nspots'])
            print(f'There are {n_spots:d} number of cells in total in Looping.')
            print(f'There are {self.n_tracks:d} number of cell tracks in total in Looping.')


class MamutTracks:
    """
    Order the Mamut tracks. Here we create a list called spots_track_ID which has the following structure:
    [ track # 1: {spot ID1, spot ID2, ...}]; [track # 2: {spot ID1, spot ID2, ...}]; ...
    Each track represents the entire track of a single cell and each spot corresponds to the cell in a given frame.

    Attributes
    ----------
    obj : untangle object
           parsed _mamut.xml file
    spots_xpos_per_frame : dict
                           x-coordinates of each spot for each frame
    spots_ypos_per_frame : dict
                           y-coordinates of each spot for each frame
    spots_zpos_per_frame : dict
                           z-coordinates of each spot for each frame
    spots_ID_per_frame : dict
                         IDs of each spot in each frame
    spots_trackID_per_frame : dict
                              In case that there are more than 1 time-point, we get the organized IDs for all cells
                              per frame to form a cell track

    Methods
    -------

    """

    def __init__(self, path_xml, path_mamut):
        """Constructor method
        """
        super().__init__()

        # Load the xml features from the images
        features = XMLParser(path_xml, path_mamut, 0)

        # Parse the two xml files: mamut file and the hdf5-xml file
        obj = untangle.parse(path_mamut)

        # Create array with the ID of each track
        self.spots_xpos_per_frame = {key: [] for key in np.arange(features.n_frames)}
        self.spots_ypos_per_frame = {key: [] for key in np.arange(features.n_frames)}
        self.spots_zpos_per_frame = {key: [] for key in np.arange(features.n_frames)}
        self.spots_ID_per_frame = {key: [] for key in np.arange(features.n_frames)}
        self.spots_trackID_per_frame = {key: [] for key in np.arange(features.n_frames)}

        for i in range(features.n_frames):

            # If there are more than 1 time-point
            try:
                self.n_spots_in_frame = len(obj.TrackMate.Model.AllSpots.SpotsInFrame[i])
            # Catch the exception in case there is only 1 time-point
            except:
                self.n_spots_in_frame = len(obj.TrackMate.Model.AllSpots.SpotsInFrame)

            for j in range(self.n_spots_in_frame):
                # If there are more than 1 time-point
                try:
                    self.spots_xpos_per_frame[i].append(
                        float(obj.TrackMate.Model.AllSpots.SpotsInFrame[i].Spot[j]['POSITION_X']))
                    self.spots_ypos_per_frame[i].append(
                        float(obj.TrackMate.Model.AllSpots.SpotsInFrame[i].Spot[j]['POSITION_Y']))
                    self.spots_zpos_per_frame[i].append(
                        float(obj.TrackMate.Model.AllSpots.SpotsInFrame[i].Spot[j]['POSITION_Z']))
                    self.spots_ID_per_frame[i].append(int(obj.TrackMate.Model.AllSpots.SpotsInFrame[i].Spot[j]['ID']))
                    self.spots_trackID_per_frame[i].append(
                        int(float(obj.TrackMate.Model.AllSpots.SpotsInFrame[i].Spot[j]['Spot_track_ID'])))
                # Catch the exception in case there is only 1 time-point
                except:
                    self.spots_xpos_per_frame[i].append(
                        float(obj.TrackMate.Model.AllSpots.SpotsInFrame.Spot[j]['POSITION_X']))
                    self.spots_ypos_per_frame[i].append(
                        float(obj.TrackMate.Model.AllSpots.SpotsInFrame.Spot[j]['POSITION_Y']))
                    self.spots_zpos_per_frame[i].append(
                        float(obj.TrackMate.Model.AllSpots.SpotsInFrame.Spot[j]['POSITION_Z']))
                    self.spots_ID_per_frame[i].append(int(obj.TrackMate.Model.AllSpots.SpotsInFrame.Spot[j]['ID']))


class LabkitData:
    """
    Read the labels created using labkit. These labels will be used later to remove all the tracks which are not
    contained in the label(s) and then re-saved in a -mamut.xml file containing the tracks for each label individually.

    Attributes
    ----------
    obj : untangle object
           parsed _mamut.xml file
    spots_xpos_per_frame : dict
                           x-coordinates of each spot for each frame
    spots_ypos_per_frame : dict
                           y-coordinates of each spot for each frame
    spots_zpos_per_frame : dict
                           z-coordinates of each spot for each frame
    spots_ID_per_frame : dict
                         IDs of each spot in each frame
    spots_trackID_per_frame : dict
                              In case that there are more than 1 time-point, we get the organized IDs for all cells
                              per frame to form a cell track
    plot : int
           If 1, the masks are plotted, otherwise they are not.

    Methods
    -------

    """

    def __init__(self, path_labels, path_xml, path_mamut, plot):
        """Constructor method
        """
        super().__init__()

        # Load the xml features from the images
        features = XMLParser(path_xml, path_mamut, 0)

        # Spots per frame: tracks, IDs, coordinates
        spots = MamutTracks(path_xml, path_mamut)

        # Open the labeling xml file
        labeling_file = open(path_labels, "r", encoding="utf8", errors='ignore')
        for line in labeling_file:
            pass

        # Create a dict from the xml file
        labeling_dict = eval(line)

        # n_labels = len(labeling_dict['labels'].keys())
        self.labels = list(labeling_dict['labels'].keys())

        # If there are tracks, we take care of them, otherwise we just take care of the spots
        self.n_tracks = features.n_tracks

        if self.n_tracks > 0 :
            # frame in which the label of labkit was done
            self.frame = labeling_dict['labels'][self.labels[0]][0][-1]
        else:
            self.frame = 0

        # Dictionary to save all the labeled coordinates and label names
        if self.n_tracks > 0:
            data_dict = {'X': [], 'Y': [], 'Z': [], 'T': [], 'Labels': []}
        else:
            data_dict = {'X': [], 'Y': [], 'Z': [], 'Labels': []}

        # Read all the data inside each label created with labkit
        for label in self.labels:
            labels_vals = labeling_dict['labels'][label]
            x_aux = []
            y_aux = []
            z_aux = []
            t_aux = []

            for i in range(len(labels_vals)):
                x_aux.append(labels_vals[i][0])
                y_aux.append(labels_vals[i][1])
                z_aux.append(labels_vals[i][2])
                if self.n_tracks > 0:
                    t_aux.append(labels_vals[i][3])

            data_dict['X'] += x_aux
            data_dict['Y'] += y_aux
            data_dict['Z'] += z_aux
            if self.n_tracks > 0:
                data_dict['T'] += t_aux
            data_dict['Labels'] += [label] * len(labels_vals)

        # Convert the previous dictionary into a pandas DataFrame
        data_df = pd.DataFrame(data_dict)

        # Create a mask from the labeled coordinates
        self.masks = [np.zeros((features.width, features.height)), np.zeros((features.width, features.n_slices)),
                      np.zeros((features.height, features.n_slices))]

        '''
        Spots to keep according to the labeling: Here we chose the spots we want to keep according to the labeling
        from labkit
        '''

        self.spots_to_keep = []
        spots_to_remove = []
        frames_to_remove = []
        self.spots_to_keep_id = []
        spots_to_remove_id = []
        spots_to_remove_label = []
        tracks_to_remove = []
        tracks_to_keep = []

        count = 0

        for label in self.labels:
            x = data_df.loc[data_df['Labels'] == label]['X'].to_numpy()
            y = data_df.loc[data_df['Labels'] == label]['Y'].to_numpy()
            z = data_df.loc[data_df['Labels'] == label]['Z'].to_numpy()
            try:
                t = data_df.loc[data_df['Labels'] == label]['T'].to_numpy()
            except KeyError:
                t = 0

            bitmap = np.zeros((features.width, features.height, features.n_slices))
            bitmap[x, y, z] = 1 + count

            # XY
            self.masks[0] += np.max(bitmap, axis=2)

            # XZ
            self.masks[1] += np.max(bitmap, axis=1)

            # ZY
            self.masks[2] += np.max(bitmap, axis=0)

            count += 1

            spots_to_remove_aux = []
            spots_to_keep_aux = []
            spots_to_keep_id_aux = []
            spots_to_remove_id_aux = []
            spots_to_remove_label_aux = []

            frames = np.unique(t)

            if frames > 0:
                t_ = []
                tracks_to_remove_aux = []
                tracks_to_keep_aux = []

            for j in frames:
                n_spots = len(spots.spots_xpos_per_frame[j])
                for i in range(n_spots):
                    # Convert the spots position into pixels (fromt micrometers)
                    x_ = int(np.round(spots.spots_xpos_per_frame[j][i] / features.x_pixel))
                    y_ = int(np.round(spots.spots_ypos_per_frame[j][i] / features.y_pixel))
                    z_ = int(np.round(spots.spots_zpos_per_frame[j][i] / features.z_pixel))
                    ind = int(spots.spots_ID_per_frame[j][i])
                    if frames > 0:
                        track_ind = int(spots.spots_trackID_per_frame[j][i])

                    # Check if that position is inside the labeled data from labkit
                    if x_ < features.width and y_ < features.height and z_ < features.n_slices:
                        if bitmap[x_, y_, z_] == 0:
                            spots_to_remove_aux.append(i)
                            spots_to_remove_id_aux.append(ind)
                            spots_to_remove_label_aux.append(label)
                            if frames > 0:
                                tracks_to_remove_aux.append(track_ind)
                                t_.append(j)
                        else:
                            spots_to_keep_aux.append(i)
                            spots_to_keep_id_aux.append(ind)
                            if frames > 0:
                                tracks_to_keep_aux.append(track_ind)
                    else:
                        spots_to_remove_aux.append(i)
                        spots_to_remove_id_aux.append(ind)
                        spots_to_remove_label_aux.append(label)
                        if frames > 0:
                            tracks_to_remove_aux.append(track_ind)
                            t_.append(j)

            self.spots_to_keep.append(spots_to_keep_aux)
            spots_to_remove.append(spots_to_remove_aux)
            spots_to_remove_id.append(spots_to_remove_id_aux)
            self.spots_to_keep_id.append(spots_to_keep_id_aux)
            spots_to_remove_label.append(spots_to_remove_label_aux)

            if frames > 0:
                tracks_to_keep.append(tracks_to_keep_aux)
                tracks_to_remove.append(tracks_to_remove_aux)
                frames_to_remove.append(t_)

            # Plot the masks

            if plot == 1:
                combs = list(itertools.combinations(range(3), 2))
                labels_axis = ['X [$\mu$m]', 'Y [$\mu$m]', 'Z [$\mu$m]']

                self.fig = plt.figure(figsize=plt.figaspect(0.25))

                for i in range(3):
                    ax = self.fig.add_subplot(1, 3, i + 1)
                    plt.imshow(self.masks[i], cmap='jet', alpha=1, aspect='auto')
                    plt.xticks(np.arange(1), fontsize=25)
                    plt.yticks(np.arange(1), fontsize=25)
                    plt.xlabel(labels_axis[combs[i][0]], fontsize=30)
                    plt.ylabel(labels_axis[combs[i][1]], fontsize=30)
                    [i.set_linewidth(4) for i in ax.spines.values()]
                    plt.gca().invert_yaxis()

                plt.tight_layout()


class SpotsRemoval:
    """
    Read the labels created using labkit. These labels will be used later to remove all the tracks which are not
    contained in the label(s) and then re-saved in a -mamut.xml file containing the tracks for each label individually.

    Attributes
    ----------
    obj : untangle object
           parsed _mamut.xml file
    spots_xpos_per_frame : dict
                           x-coordinates of each spot for each frame
    spots_ypos_per_frame : dict
                           y-coordinates of each spot for each frame
    spots_zpos_per_frame : dict
                           z-coordinates of each spot for each frame
    spots_ID_per_frame : dict
                         IDs of each spot in each frame
    spots_trackID_per_frame : dict
                              In case that there are more than 1 time-point, we get the organized IDs for all cells
                              per frame to form a cell track

    Methods
    -------

    """

    def __init__(self, path_save, path_mamut, path_labels, path_xml):
        """Constructor method
        """
        super().__init__()

        labkit = LabkitData(path_labels, path_xml, path_mamut, 0)
        # FOR EACH LABEL

        for label, val in enumerate(labkit.labels):

            t = time.time()
            # Open the  XML file of your original data to write and edit
            # Then we will save and edit a new xml file
            tree = ElementTree()
            tree.parse(path_mamut)

            # XML Declaration
            f = BytesIO()
            tree.write(f, encoding='utf-8', xml_declaration=True)

            # Correct the units
            units = 'micron'
            for TrackMate in tree.iter('TrackMate'):
                for Model in TrackMate.iter('Model'):
                    # Change value
                    Model.set('spatialunits', '%s' % units)

            # Delete the Spots that are no longer in the tracks of interest (if there are tracks)
            if labkit.n_tracks > 0:
                tracks_keep = []
                for SpotsInFrame in tree.iter('SpotsInFrame'):
                    for Spot in SpotsInFrame.findall('Spot'):

                        if int(float(Spot.attrib['ID']) in labkit.spots_to_keep_id[label]):
                            tracks_keep.append(int(float(Spot.attrib['Spot_track_ID'])))

                for SpotsInFrame in tree.iter('SpotsInFrame'):
                    for Spot in SpotsInFrame.findall('Spot'):

                        if (int(float(Spot.attrib['ID']) not in labkit.spots_to_keep_id[label])) and \
                                (int(float(Spot.attrib['Spot_track_ID'])) not in tracks_keep):
                            SpotsInFrame.remove(Spot)

            # Delete the Spots that are no longer in the tracks of interest
            count = 0
            spots_keep = []
            for SpotsInFrame in tree.iter('SpotsInFrame'):
                for Spot in SpotsInFrame.findall('Spot'):
                    spots_keep.append(Spot.attrib['ID'])
                    count += 1

            n_spots_left = count

            # First thing to change is "nspots":
            for Model in tree.iter('Model'):
                for AllSpots in Model.iter('AllSpots'):
                    # Change value
                    AllSpots.set('nspots', '%d' % n_spots_left)

            if labkit.n_tracks > 0:
                track_ids = []
                spots_keep_ = [int(i) for i in spots_keep]
                for AllTracks in tree.iter('AllTracks'):
                    for Track in AllTracks.findall('Track'):
                        for Edge in Track.findall('Edge'):
                            if (int(Edge.attrib['SPOT_SOURCE_ID']) not in spots_keep_) or \
                                    (int(Edge.attrib['SPOT_TARGET_ID']) not in spots_keep_):
                                track_ids.append(Track.attrib['TRACK_ID'])
                                AllTracks.remove(Track)
                                break

                for FilteredTracks in tree.iter('FilteredTracks'):
                    for TrackID in FilteredTracks.findall('TrackID'):
                        if TrackID.attrib['TRACK_ID'] in track_ids:
                            FilteredTracks.remove(TrackID)

            print('Label: %s' % val)
            print('Elapsed time in seconds: ', time.time() - t)
            print(f'Number of spots: {n_spots_left:d}')
            if labkit.n_tracks > 0:
                print(f'Number of tracks: {len(track_ids):d}')

            # Write a new xml with only the kept spots
            tree.write('%s/%s_dataset_mamut.xml' % (path_save, labkit.labels[label]))


class Plots:

    """
    Plot the Gaussian Kernel Density of all the initial spots. As a result you will get a plot of the 3 axes
    combinations (XY, XZ, YZ) color coded as the cell density. Colorbar goes from low to high cell density.

    Attributes
    ----------
    obj : untangle object
           parsed _mamut.xml file
    spots_xpos_per_frame : dict
                           x-coordinates of each spot for each frame
    spots_ypos_per_frame : dict
                           y-coordinates of each spot for each frame
    spots_zpos_per_frame : dict
                           z-coordinates of each spot for each frame
    spots_ID_per_frame : dict
                         IDs of each spot in each frame
    spots_trackID_per_frame : dict
                              In case that there are more than 1 time-point, we get the organized IDs for all cells
                              per frame to form a cell track

    Methods
    -------

    """

    def __init__(self, path_mamut, path_xml, labkit, plot_type):
        """Constructor method
        """
        super().__init__()
        spots = MamutTracks(path_xml, path_mamut)
        features = XMLParser(path_xml, path_mamut, 0)

        # Obtain the coordinates for each dimension per frame
        self.x = spots.spots_xpos_per_frame[labkit.frame]
        self.y = spots.spots_ypos_per_frame[labkit.frame]
        self.z = spots.spots_zpos_per_frame[labkit.frame]
        self.data = [self.x, self.y, self.z]

        # Figure parameters
        self.combs = list(itertools.combinations(range(3), 2))
        self.labels_axis = ['X [$\mu$m]', 'Y [$\mu$m]', 'Z [$\mu$m]']
        self.sizes = [features.width * features.x_pixel, features.height * features.y_pixel,
                 features.n_slices * features.z_pixel]
        self.pixels = [features.x_pixel, features.y_pixel, features.z_pixel]
        self.colors = ['hotpink', 'blue', 'darkorange', 'limegreen', 'teal'] * 10
        self.cmap = 'inferno'


        if plot_type == 'gkd':
            self.gaussian_kernel_density()
        elif plot_type == 'masks_spots':
            self.masks_spots(labkit)
        elif plot_type == '3d':
            self.three_d_plot(labkit)

    def gaussian_kernel_density(self):
        # Calculate the point density for each 2D dimension combination
        xy = np.vstack([self.x, self.y])
        xz = np.vstack([self.x, self.z])
        yz = np.vstack([self.y, self.z])

        # Calculate the Gaussian kerdel density
        gkd_xy = gaussian_kde(xy)(xy)
        gkd_xz = gaussian_kde(xz)(xz)
        gkd_yz = gaussian_kde(yz)(yz)
        gkd = [gkd_xy, gkd_xz, gkd_yz]

        # Plot the results
        self.fig = plt.figure(figsize=plt.figaspect(0.25))

        for i in range(3):
            self.ax = self.fig.add_subplot(1, 3, i + 1)
            plt.scatter(self.data[self.combs[i][0]], self.data[self.combs[i][1]], c=gkd[i], cmap=self.cmap)
            plt.xlabel(self.labels_axis[self.combs[i][0]], fontsize=30)
            plt.ylabel(self.labels_axis[self.combs[i][1]], fontsize=30)
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, self.sizes[self.combs[i][0]]])
            plt.ylim([0, self.sizes[self.combs[i][1]]])
            [i.set_linewidth(4) for i in self.ax.spines.values()]

        # Colorbar
        self.cb = plt.colorbar(ticks=[], aspect=10)
        self.cb.outline.set_linewidth(4)
        self.cb.ax.text(0.5, -0.01, 'Low', transform=self.cb.ax.transAxes,
                        va='top', ha='center', fontsize=25)
        self.cb.ax.text(0.5, 1.0, 'High', transform=self.cb.ax.transAxes,
                        va='bottom', ha='center', fontsize=25)

        plt.tight_layout()

    def masks_spots(self, labkit):
        self.fig = plt.figure(figsize=plt.figaspect(0.25))

        for i in range(3):
            ax = self.fig.add_subplot(1, 3, i + 1)

            # Plot each label
            for label in range(len(labkit.labels)):
                if np.size(np.array(labkit.spots_to_keep[label])) > 0:
                    plt.scatter(
                        np.array(self.data[self.combs[i][0]])[np.array(labkit.spots_to_keep[label])] /
                        self.pixels[self.combs[i][0]], \
                        np.array(self.data[self.combs[i][1]])[np.array(labkit.spots_to_keep[label])] /
                        self.pixels[self.combs[i][1]],
                        c=self.colors[label], edgecolor='k', alpha=0.8, zorder=10, label=labkit.labels[label], s=20)

            plt.legend(loc='best', fontsize=16)

            plt.imshow(labkit.masks[i].T, cmap='Greys', alpha=0.5, aspect='auto')
            plt.xticks(np.arange(1), fontsize=25);
            plt.yticks(np.arange(1), fontsize=25);
            plt.xlabel(self.labels_axis[self.combs[i][0]], fontsize=30)
            plt.ylabel(self.labels_axis[self.combs[i][1]], fontsize=30)

            [i.set_linewidth(4) for i in ax.spines.values()]

        plt.tight_layout()
        plt.show()

    def three_d_plot(self, labkit):

        self.fig = plt.figure(figsize=plt.figaspect(0.25))

        ax = self.fig.add_subplot(1, 3, 1, projection='3d')

        # Data for three-dimensional scattered points
        zdata = self.z
        xdata = self.x
        ydata = self.y

        ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greys', s=2, alpha=0.05, zorder=0);

        for label in range(len(labkit.labels)):

            if np.size(np.array(labkit.spots_to_keep[label])) > 0:
                x_keep = np.array(self.x)[np.array(labkit.spots_to_keep[label])]
                y_keep = np.array(self.y)[np.array(labkit.spots_to_keep[label])]
                z_keep = np.array(self.z)[np.array(labkit.spots_to_keep[label])]

                ax.scatter3D(x_keep, y_keep, z_keep, c=self.colors[label], s=2, alpha=0.2,
                             zorder=20)

        ax.grid(False)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_zlabel('Z', fontsize=20)

        ax = self.fig.add_subplot(1, 3, 2, projection='3d')

        # Data for three-dimensional scattered points
        zdata = self.z
        xdata = self.x
        ydata = self.y
        ax.scatter3D(xdata, zdata, ydata, c=ydata, cmap='Greys', s=2, alpha=0.05, zorder=0);

        for label in range(len(labkit.labels)):

            if np.size(np.array(labkit.spots_to_keep[label])) > 0:
                x_keep = np.array(self.x)[np.array(labkit.spots_to_keep[label])]
                y_keep = np.array(self.y)[np.array(labkit.spots_to_keep[label])]
                z_keep = np.array(self.z)[np.array(labkit.spots_to_keep[label])]

                ax.scatter3D(x_keep, z_keep, y_keep, c=self.colors[label], s=2, alpha=0.2,
                             zorder=20)

        ax.grid(False)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Z', fontsize=20)
        ax.set_zlabel('Y', fontsize=20)

        ax = self.fig.add_subplot(1, 3, 3, projection='3d')

        # Data for three-dimensional scattered points
        zdata = self.z
        xdata = self.x
        ydata = self.y
        ax.scatter3D(ydata, zdata, xdata, c=xdata, cmap='Greys', s=2, alpha=0.05, zorder=0);

        for label in range(len(labkit.labels)):

            if np.size(np.array(labkit.spots_to_keep[label])) > 0:
                x_keep = np.array(self.x)[np.array(labkit.spots_to_keep[label])]
                y_keep = np.array(self.y)[np.array(labkit.spots_to_keep[label])]
                z_keep = np.array(self.z)[np.array(labkit.spots_to_keep[label])]

                ax.scatter3D(y_keep, z_keep, x_keep, c=self.colors[label], s=2, alpha=0.2,
                             zorder=20)

        ax.grid(False)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlabel('Y', fontsize=20)
        ax.set_ylabel('Z', fontsize=20)
        ax.set_zlabel('X', fontsize=20)

        plt.show()






