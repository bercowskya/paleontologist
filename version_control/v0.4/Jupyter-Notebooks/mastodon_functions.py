import numpy as np
import pandas as pd
import untangle
import tqdm

# Obtain all the features that are in the .xml file which has been generated when the data
# is converted to .hdf5 using BigDataViewer/BigStitcher/MultiviewReconstruction in Fiji
class xml_features:
    def __init__(self, path_xml):
        # Parse .xml file
        obj = untangle.parse(path_xml)
        # Data Features
        try:
            self.channels = len(obj.SpimData.SequenceDescription.ViewSetups.Attributes[1])
        except:
            self.channels = len(obj.SpimData.SequenceDescription.ViewSetups.Attributes.Channel)
        ch = self.channels
        self.dim = 3
        
        if ch > 1:
            self.width = int(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].size.cdata.split()[0])
            self.height = int(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].size.cdata.split()[1])
            self.n_slices = int(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].size.cdata.split()[2])

            self.x_pixel = float(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].voxelSize.size.cdata.split()[0])
            self.y_pixel = float(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].voxelSize.size.cdata.split()[1])
            self.z_pixel = float(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].voxelSize.size.cdata.split()[2])
            
            self.units = obj.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].voxelSize.unit.cdata
        else:
            self.width = int(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup.size.cdata.split()[0])
            self.height = int(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup.size.cdata.split()[1])
            self.n_slices = int(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup.size.cdata.split()[2])

            self.x_pixel = float(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup.voxelSize.size.cdata.split()[0])
            self.y_pixel = float(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup.voxelSize.size.cdata.split()[1])
            self.z_pixel = float(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup.voxelSize.size.cdata.split()[2])
            
            self.units = obj.SpimData.SequenceDescription.ViewSetups.ViewSetup.voxelSize.unit.cdata
            

        #self.channels = len(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup)
      
        while True:
            try:
                self.n_frames = len(obj.SpimData.SequenceDescription.Timepoints.integerpattern.cdata.split())
                break
            except AttributeError:
                pass  # fallback to dict
            try:
                self.n_frames = int(obj.SpimData.SequenceDescription.Timepoints.last.cdata.split()[0])
                break
            except KeyError:
                raise AttributeError("There is something wrong with the .xml file - Did you compute the features?") from None
        

class csv_reader():
    def __init__(self, path_csv, path_xml):

        # Read data from file 'file_path/file_name.csv' 
        data = pd.read_csv('%s-vertices.csv'%(path_csv), header=[0,1], low_memory=False)
        edges = pd.read_csv('%s-edges.csv'%(path_csv), header=[0,1], low_memory=False)

        # Arrange -edges df
        # Create a df with the Link target IDs: spot source and target
        # We start at [1:] because it is a multi-index, to remove the label 'Link Targt IDs'
        self.edges_df_sorted = edges.loc[1:, 'Link target IDs'].reset_index(drop=True)

        # Rename the columns for easy access
        self.edges_df_sorted = self.edges_df_sorted.rename(columns={'Source spot id': 'source ID', 'Target spot id': 'target ID'})
        self.edges_df_sorted = self.edges_df_sorted.reset_index()

        # Check if indeed the target and source are in the correct order (e.g. source smaller ID than target)
        if self.edges_df_sorted['source ID'][0] > self.edges_df_sorted['target ID'][0]:
            self.edges_df_sorted[['source ID', 'target ID']] = self.edges_df_sorted[['target ID', 'source ID']]


        # Add the Track IDs columns
        self.edges_df_sorted = pd.concat([self.edges_df_sorted, edges.loc[1:, 'ID'].rename(
            columns={f'{edges.loc[1:, "ID"].keys()[0]}': 'Spot ID'}).reset_index(drop=True)], axis=1)

        # Get the keys for all the features computed in each csv file
        data_labels = np.array([list(data.keys())[i][0] for i in range(len(data.keys()))])
        edges_labels = np.array([list(edges.keys())[i][0] for i in range(len(edges.keys()))])

        # To know in the end if we have any tags
        ind_tag = []

        # To save the labels of the data we have from the csv file
        final_labels = []

        # To save the data which we will use to make a new DataFrame
        features = []

        # -vertices feature: ID
        if any(data_labels == 'ID'):            
            # To know in the end if we have any tags
            ind_tag.append(np.max(np.where(data_labels == 'ID')[0]))

            # Save the final label
            final_labels.append('Spot ID')
            
            # Save the data
            features.append(np.array([i[0] for i in data['ID'][1:].to_numpy(dtype=int)]))

        # -vertices feature: Spot Intensity - usually contains: Mean, Median, Min, Max, Sum and Std for each channel
        if any(data_labels == 'Spot intensity'):
            keys = data['Spot intensity'].keys()

            for i in keys:
                # Save the data
                features.append(data['Spot intensity'][i][1:].to_numpy(dtype='float'))
            
            # To know in the end if we have any tags
            ind_tag.append(np.max(np.where(data_labels == 'Spot intensity')[0]))

            # Save the final label
            final_labels += list(keys)

        # -vertices feature: Spot N Links
        if any(data_labels == 'Spot N links'):
            # To know in the end if we have any tags
            ind_tag.append(np.max(np.where(data_labels == 'Spot N links')[0]))

            # Save the final label
            final_labels.append('N Links')
            
            # Save the data
            features.append(np.array([i[0] for i in data['Spot N links'][1:].to_numpy(dtype=int)]))

        # - vertices feature: Spot center intensity - usually contains: Mean for each channel
        if any(data_labels == 'Spot center intensity'):
            keys = data['Spot center intensity'].keys()

            for i in keys:
                # Save the data
                features.append(data['Spot center intensity'][i][1:].to_numpy(dtype='float'))
            
            # To know in the end if we have any tags
            ind_tag.append(np.max(np.where(data_labels == 'Spot center intensity')[0]))

            # Save the final label
            final_labels += list(keys)


        # - vertices feature: Spot Frame
        if any(data_labels == 'Spot frame'):
            # To know in the end if we have any tags
            ind_tag.append(np.max(np.where(data_labels == 'Spot frame')[0]))

            # Save the final label
            final_labels.append('Frames')
            
            # Save the data
            features.append(np.array([i[0] for i in data['Spot frame'][1:].to_numpy(dtype=int)]))

        # - vertices feature: Spot Position - usually contains: X, Y and Z (according to the number of dimensions)
        if any(data_labels == 'Spot position'):
            keys = data['Spot position'].keys()

            for i in keys:
                # Save the data
                features.append(data['Spot position'][i][1:].to_numpy(dtype='float'))
            
            # To know in the end if we have any tags
            ind_tag.append(np.max(np.where(data_labels == 'Spot position')[0]))

            # Save the final label
            final_labels += list(keys)

        # - vertices feature: Spot Quick Mean - usually contains: Mean for each channel
        if any(data_labels == 'Spot quick mean'):
            keys = data['Spot quick mean'].keys()

            for i in keys:
                # Save the data
                features.append(data['Spot quick mean'][i][1:].to_numpy(dtype='float'))
            
            # To know in the end if we have any tags
            ind_tag.append(np.max(np.where(data_labels == 'Spot quick mean')[0]))

            # Save the final label
            final_labels += [ 'Q-'+list(data['Spot quick mean'].keys())[i] for i in range(len(keys))]

        # - vertices feature: Spot Radius
        if any(data_labels == 'Spot radius'):
            # To know in the end if we have any tags
            ind_tag.append(np.max(np.where(data_labels == 'Spot radius')[0]))

            # Save the final label
            final_labels.append('Radius')
            
            # Save the data
            features.append(np.array([i[0] for i in data['Spot radius'][1:].to_numpy(dtype=float)]))

        # - vertices feature: Spot track ID
        if any(data_labels == 'Spot track ID'):
            # To know in the end if we have any tags
            ind_tag.append(np.max(np.where(data_labels == 'Spot track ID')[0]))

            # Save the final label
            final_labels.append('Track ID')
                
            # Save the data
            features.append(np.array([i[0] for i in data['Spot track ID'][1:].to_numpy(dtype=int)]))

        # - vertices feature: Track N Spots
        if any(data_labels == 'Track N spots'):
            # To know in the end if we have any tags
            ind_tag.append(np.max(np.where(data_labels == 'Track N spots')[0]))

            # Save the final label
            final_labels.append('Track N Spots')
                
            # Save the data
            features.append(np.array([i[0] for i in data['Track N spots'][1:].to_numpy(dtype=int)]))

        # - vertices feature: Tags
        # If this happens, there are tags otherwise there are non
        if np.max(ind_tag)<len(data_labels):
            
            ind = np.max(ind_tag)+1
            
            # We create a new dataframe with the tags only
            tags_df = data.iloc[:, ind:].drop([0])
            
            # Get tags and subtags
            self.tags = [list(tags_df.keys())[i][0] for i in range(len(tags_df.keys()))]
            self.sub_tags = [list(tags_df.keys())[i][1] for i in range(len(tags_df.keys()))]
            
            space = ' '
            
            for i in range(len(self.tags)):
                final_labels.append(self.tags[i]+space+self.sub_tags[i])
                features.append(tags_df[self.tags[i]][self.sub_tags[i]])

        # - edges feature: Link Displacement
        if any(edges_labels == 'Link displacement'):
            self.link_disp = np.array([i[0] for i in edges['Link displacement'][1:].to_numpy(dtype=float)])

        # - edges feature: Link Velocity
        if any(edges_labels == 'Link velocity'):
            self.link_vel = np.array([i[0] for i in edges['Link velocity'][1:].to_numpy(dtype=float)])

        # edges feature: Link Target IDs - usually contains: Source spot ID and Target spot ID
        if any(edges_labels == 'Link target IDs'):
            keys = edges['Link target IDs'].keys()

            self.target_ids = dict.fromkeys(keys)

            for i in keys:
                # Save and convert into array
                self.target_ids[i] = edges['Link target IDs'][i][1:].to_numpy(dtype='float')
            
        else:
            print('Your -edges.csv file does not contain Link Target IDs, \
            without this we cannot reconstruct the tracks')

        # Outputs:
        # 1. DataFrame with only 1 Index Level (no more multi-index) for the -vertices data
        data_df = pd.DataFrame(dict(zip(final_labels, features)))

        #. 2. Group DataFrame by Track IDs and sort the Frames
        data_df_groupby_track_ids = data_df.groupby(['Track ID'])
        self.data_df_sorted = data_df_groupby_track_ids.apply(lambda x: x.sort_values('Frames', ascending = True))


# Obtain all the information that comes in the .csv files (-vertices and -edges) that is
# generated with Mastodon once you have computed the features and saved as .csv
class csv_features:
    def __init__(self, path_csv, path_xml):

        xml_params = xml_features(path_xml)  # To use the parameters from the other function

        # Read data from file 'file_path/file_name.csv'
        data = pd.read_csv('%s-vertices.csv' % (path_csv), header=[0, 1], low_memory=False)
        edges = pd.read_csv('%s-edges.csv' % (path_csv), header=[0, 1], low_memory=False)
        # sep=',', error_bad_lines=False, index_col=False, dtype='unicode'

        n_features = len(data.keys())  # All features that appear in data

        '''
        Rearanging and capturing all the features from the .csv files called -vertices and -edges
        '''

        # Number of links in each spot
        for i in range(n_features):
            if data.keys()[i][0] == 'Spot N links':
                self.n_links = np.array(data[data.keys()[i]][1:], dtype=int)
                break

        # ID --> For linking spot source to spot target (for cell divisions)
        for i in range(n_features):
            if data.keys()[i][0] == 'ID':
                self.IDs = np.array(data[data.keys()[1]][1:], dtype=int)
                break

        # Spot source ID
        for i in range(len(edges.keys())):
            if edges.keys()[i][1] == 'Target spot id' and len(edges.Label) > 1:  # len(edges.Label) --> In case there are no tracks
                self.spot_source = np.array(edges[edges.keys()[i]][1:], dtype=int)
                break

        # Spot target ID
        for i in range(len(edges.keys())):
            if edges.keys()[i][1] == 'Source spot id' and len(edges.Label) > 1:
                self.spot_target = np.array(edges[edges.keys()[i]][1:], dtype=int)
                break

        # Spot frame
        for i in range(n_features):
            if data.keys()[i][0] == 'Spot frame':
                self.frames = np.array(data[data.keys()[i]][1:], dtype=int)
                break

        # Track ID
        for i in range(n_features):
            if data.keys()[i][0] == 'Spot track ID':
                self.track_id = np.array(data[data.keys()[i]][1:], dtype=int)
                break

        self.n_tracks = len(np.unique(self.track_id))

        # Number of spots per track
        for i in range(n_features):
            if data.keys()[i][0] == 'Track N spots':
                self.n_spot = np.array(data[data.keys()[i]][1:], dtype=int)
                break

        # Mean and Median intensity per track
        self.mean = {key: [] for key in np.arange(xml_params.channels)}
        self.median = {key: [] for key in np.arange(xml_params.channels)}
        self.std = {key: [] for key in np.arange(xml_params.channels)}
        for i in range(n_features):
            if data.keys()[i][0] == 'Spot intensity':
                for c in range(xml_params.channels):
                    if data.keys()[i][1] == f'Mean ch{c+1}':
                        self.mean[c] = np.array(data[data.keys()[i]][1:], dtype=float)
                    elif data.keys()[i][1] == f'Median ch{c+1}':
                        self.median[c] = np.array(data[data.keys()[i]][1:], dtype=float)
                    elif data.keys()[i][1] == f'Std ch{c+1}':
                        self.std[c] = np.array(data[data.keys()[i]][1:], dtype=float)




        # Position
        self.x = data['Spot position']['X'][1:].to_numpy(dtype=float)
        self.y = data['Spot position']['Y'][1:].to_numpy(dtype=float)
        self.z = data['Spot position']['Z'][1:].to_numpy(dtype=float)


# Order tracks according to whether they divide or not:
# using the data obtained from csv_features
class ordering_tracks:
    def __init__(self, path_csv, path_xml):

        csv_params = csv_features(path_csv, path_xml)  # To use the parameters from the other function

        # ORDER TRACKS BY THEIR ID
        # Number of tracks
        len_id = len(np.unique(csv_params.track_id))

        frames_by_track = []
        links_by_track = []
        source_by_track = []
        target_by_track = []
        ids_by_track = []
        cells_track = []  # Keep tracks of the spots which have a track

        # Positions
        x_by_track = []
        y_by_track = []
        z_by_track = []


        for i in tqdm.tqdm(range(len_id), desc='Load Cell Tracks'):
            idx = np.where(csv_params.track_id == np.unique(csv_params.track_id)[i])[0]

            # In case there are spots with tracks
            if len(idx) > 1:
                sorted_frames = csv_params.frames[idx].argsort()

                # Save the sorted frames
                frames_by_track.append(np.sort(csv_params.frames[idx]))

                # Save the sorted links (1, 2 or 3 links in case of division)
                links_by_track.append(csv_params.n_links[idx][sorted_frames])

                # IDs by tracks
                ids_by_track.append(csv_params.IDs[idx][sorted_frames])

                # Sorted spot source ID and spot target ID
                ind_ids = csv_params.IDs[idx][sorted_frames]
                source_by_track.append(csv_params.spot_source[np.array([ind for ind, element in enumerate \
                    (csv_params.spot_source) if element in ind_ids[:-1]])])
                target_by_track.append(csv_params.spot_target[np.array([ind for ind, element in enumerate \
                    (csv_params.spot_target) if element in ind_ids[1:]])])

                # cells with tracks
                cells_track.append(i)

                # Positions
                x_by_track.append(csv_params.x[idx][sorted_frames])
                y_by_track.append(csv_params.y[idx][sorted_frames])
                z_by_track.append(csv_params.z[idx][sorted_frames])

            # If there are spots without any tracks
            else:
                continue

        # ORDER TRACKS ACCORDING WHETHER THEY DIVIDE OR NOT
        self.spots_features = {key: [] for key in ['Frames', 'ID', 'DivID', 'Track ID', 'X', 'Y', 'Z']}

        # DivID : If division, the an ID equal to its sibling. If not, nan
        DivID = 0
        self.n_tracks_divs = 0  # Number of tracks including divisions
        self.n_division_tracks = 0  # Number of tracks with divisions

        for i in tqdm.tqdm(range(len(cells_track)), desc='Arrange Cell Tracks'):
            # Are there any divisions in the track?
            # (e.g. the spot divides in two different opportunities during all the timeseries)
            n_divs = len(list(map(int, np.where(links_by_track[i] > 2)[0])))

            # How many times the spot divides per division?
            # (e.g. in one specific division, in how many daughters the spot divided?)
            # n_divs_cell = links_by_track[links_by_track[i]>2]

            if n_divs == 0:  # There is no cell division
                self.spots_features['Frames'].append(frames_by_track[i])
                self.spots_features['ID'].append(ids_by_track[i])
                self.spots_features['DivID'].append(0)
                self.spots_features['Track ID'].append(cells_track[i])
                self.spots_features['X'].append(x_by_track[i])
                self.spots_features['Y'].append(y_by_track[i])
                self.spots_features['Z'].append(z_by_track[i])
                self.n_tracks_divs += 1


            else:  # Otherwise, there is cell division
                DivID += 1
                div_vect = []  # one vector for [each division+1] we want to keep track of
                val0 = np.where(links_by_track[i] == 3)[0][0]  # index of first division

                # save the IDs up to first division for all tracks
                for j in range(n_divs + 1):
                    div_vect.append(ids_by_track[i][:val0 + 1].tolist())  # IDS[0:first division]

                # store the list of already saved IDs to not use them again
                list_idx_sources_used = []  # To save the indices of used sources spots
                list_used = ids_by_track[i][:val0 + 1].tolist()  # list of IDs used --> finish while loop when all IDs used
                # while we have not stored all IDs, loop across tracks and fill them with targets (if not in list_used)

                while not (all(elem in list_used for elem in ids_by_track[i])):

                    for j in range(n_divs + 1):
                        idx = np.where(source_by_track[i] == div_vect[j][-1])[0]
                        # In the exact moment of division
                        if len(idx) > 1:  # point of division
                            cond = True
                            k = 0
                            while cond:
                                if idx[k] not in list_idx_sources_used:
                                    list_idx_sources_used.append(idx[k])
                                    idx = idx[k]
                                    cond = False
                                else:
                                    k += 1

                        # In the cases where there is no division
                        if np.size(idx) == 1:
                            div_vect[j].append(int(target_by_track[i][idx]))
                            list_used.append(int(target_by_track[i][idx]))

                        # This means it finished at least one of the tracks
                        if np.size(idx) == 0:
                            continue

                # Save each division tracks with its corresponding division ID and its frames and mean
                for j in range(n_divs + 1):
                    # inds = np.where(np.array(div_vect[j]) == np.array(ids_by_track[i]))[0] # Indices for the IDS of the tracks in one of the divisions
                    inds = [np.where(ids_by_track[i] == div_vect[j][ind])[0][0] for ind in range(len(div_vect[j]))]
                    self.spots_features['Frames'].append(frames_by_track[i][inds])
                    self.spots_features['ID'].append(ids_by_track[i][inds])
                    self.spots_features['DivID'].append(DivID)
                    self.spots_features['Track ID'].append(cells_track[i])
                    self.spots_features['X'].append(x_by_track[i][inds])
                    self.spots_features['Y'].append(y_by_track[i][inds])
                    self.spots_features['Z'].append(z_by_track[i][inds])

                    self.n_tracks_divs += 1
                    self.n_division_tracks += 1


# Order tracks acoording to whether they divide or not:
# using the data obtained from csv_features
class ordering_tracks_all:
    def __init__(self, path_csv, path_xml):
        xml_params = xml_features(path_xml)  # To use the parameters from the other function
        csv_params = csv_features(path_csv, path_xml)  # To use the parameters from the other function

        # ORDER TRACKS BY THEIR ID
        # Number of tracks
        len_id = len(np.unique(csv_params.track_id))

        # Dictionaries to save mean, std and XYZ coordinates, links, source and target ids of spots
        mean_by_track = {key: [] for key in np.arange(xml_params.channels)}
        median_by_track = {key: [] for key in np.arange(xml_params.channels)}
        frames_by_track = []
        std_by_track = {key: [] for key in np.arange(xml_params.channels)}
        pos_by_track = {key: [] for key in np.arange(xml_params.dim)}  # for XYZ Data or XY Data
        links_by_track = []
        source_by_track = []
        target_by_track = []
        ids_by_track = []
        cells_track = []  # Keep tracks of the spots which have a track

        # Number of subtags
        #n_tags = np.unique(csv_params.tag)
        #tags_all_by_track = {key: [] for key in n_tags}

        for i in tqdm.tqdm(range(len_id), desc='Load Cell Tracks'):
            idx = np.where(csv_params.track_id == np.unique(csv_params.track_id)[i])[0]

            # In case there are spots with tracks
            if len(idx) > 1:
                sorted_frames = csv_params.frames[idx].argsort()

                # Save the sorted frames
                frames_by_track.append(np.sort(csv_params.frames[idx]))

                # Save the sorted links (1, 2 or 3 links in case of division)
                links_by_track.append(csv_params.n_links[idx][sorted_frames])

                # IDs by tracks
                ids_by_track.append(csv_params.IDs[idx][sorted_frames])

                # Sorted spot source ID and spot target ID
                ind_ids = csv_params.IDs[idx][sorted_frames]
                source_by_track.append(csv_params.spot_source[np.array([ind for ind, element in enumerate \
                    (csv_params.spot_source) if element in ind_ids[:-1]])])
                target_by_track.append(csv_params.spot_target[np.array([ind for ind, element in enumerate \
                    (csv_params.spot_target) if element in ind_ids[1:]])])

                positions = [csv_params.x, csv_params.y, csv_params.z]
                # Save the coordinates
                for j in range(xml_params.dim):
                    pos_by_track[j].append(positions[j][idx][sorted_frames])

                # Save the mean, median and std according to their channel
                for j in range(xml_params.channels):
                    mean_by_track[j].append(csv_params.mean[j][idx][sorted_frames])
                    median_by_track[j].append(csv_params.median[j][idx][sorted_frames])
                    std_by_track[j].append(csv_params.std[j][idx][sorted_frames])

                # Save the tags and subtags
                #for j, val in enumerate(n_tags):
                #    tags_all_by_track[val].append(csv_params.tags_all_save[val][idx][sorted_frames])

                # cells with tracks
                cells_track.append(i)

            # If there are spots without any tracks
            else:
                continue

        # ORDER TRACKS ACCORDING WHETHER THEY DIVIDE OR NOT
        self.spots_features = {key: [] for key in ['Frames', 'Mean1', 'Mean2', 'Median', 'ID', 'DivID', 'X', 'Y', 'Z']}
        #self.spots_tags = {key: [] for key in n_tags}  # to save all the tags and subtags

        # DivID : If division, the an ID equal to its sibling. If not, nan
        DivID = 0
        self.n_tracks_divs = 0  # Number of tracks including divisions
        self.n_division_tracks = 0  # Number of tracks with divisions

        for i in tqdm.tqdm(range(len(cells_track)), desc='Arrange Cell Tracks'):
            # Are there any divisions in the track?
            # (e.g. the spot divides in two different opportunities during all the timeseries)
            n_divs = len(list(map(int, np.where(links_by_track[i] > 2)[0])))

            # How many times the spot divides per division?
            # (e.g. in one specific division, in how many daughters the spot divided?)
            # n_divs_cell = links_by_track[links_by_track[i]>2]

            if n_divs == 0:  # There is no cell division
                self.spots_features['Frames'].append(frames_by_track[i])
                self.spots_features['Mean1'].append(mean_by_track[0][i])
                self.spots_features['Mean2'].append(mean_by_track[1][i])
                self.spots_features['Median'].append(median_by_track[0][i])
                self.spots_features['ID'].append(ids_by_track[i])
                self.spots_features['DivID'].append(0)
                self.spots_features['X'].append(pos_by_track[0][i])
                self.spots_features['Y'].append(pos_by_track[1][i])
                self.spots_features['Z'].append(pos_by_track[2][i])
                self.n_tracks_divs += 1

                #for k, val in enumerate(n_tags):
                #    self.spots_tags[val].append(tags_all_by_track[val][i])

            else:  # Otherwise, there is cell division
                DivID += 1
                div_vect = []  # one vector for [each division+1] we want to keep track of
                val0 = np.where(links_by_track[i] == 3)[0][0]  # index of first division

                # save the IDs up to first division for all tracks
                for j in range(n_divs + 1):
                    div_vect.append(ids_by_track[i][:val0 + 1].tolist())  # IDS[0:first division]

                # store the list of already saved IDs to not use them again
                list_idx_sources_used = []  # To save the indices of used sources spots

                # list of IDs used --> finish while loop when all IDs used
                list_used = ids_by_track[i][:val0 + 1].tolist()

                # while we have not stored all IDs, loop across tracks and fill them with targets (if not in list_used)
                while not (all(elem in list_used for elem in ids_by_track[i])):

                    for j in range(n_divs + 1):
                        idx = np.where(source_by_track[i] == div_vect[j][-1])[0]
                        # In the exact moment of division
                        if len(idx) > 1:  # point of division
                            cond = True
                            k = 0
                            while cond:
                                if idx[k] not in list_idx_sources_used:
                                    list_idx_sources_used.append(idx[k])
                                    idx = idx[k]
                                    cond = False
                                else:
                                    k += 1

                        # In the cases where there is no division
                        if np.size(idx) == 1:
                            div_vect[j].append(int(target_by_track[i][idx]))
                            list_used.append(int(target_by_track[i][idx]))

                        # This means it finished at least one of the tracks
                        if np.size(idx) == 0:
                            continue

                # Save each division tracks with its corresponding division ID and its frames and mean
                for j in range(n_divs + 1):
                    # inds = np.where(np.array(div_vect[j]) == np.array(ids_by_track[i]))[0] # Indices for the IDS of the tracks in one of the divisions
                    inds = [np.where(ids_by_track[i] == div_vect[j][ind])[0][0] for ind in range(len(div_vect[j]))]
                    self.spots_features['Frames'].append(frames_by_track[i][inds])
                    self.spots_features['Mean1'].append(mean_by_track[0][i][inds])
                    self.spots_features['Mean2'].append(mean_by_track[1][i][inds])
                    self.spots_features['Median'].append(median_by_track[0][i][inds])
                    self.spots_features['ID'].append(ids_by_track[i][inds])
                    self.spots_features['DivID'].append(DivID)
                    self.spots_features['X'].append(pos_by_track[0][i][inds])
                    self.spots_features['Y'].append(pos_by_track[1][i][inds])
                    self.spots_features['Z'].append(pos_by_track[2][i][inds])
                    self.n_tracks_divs += 1
                    self.n_division_tracks += 1

                    #for k, val in enumerate(n_tags):
                    #    self.spots_tags[val].append(tags_all_by_track[val][i][inds])