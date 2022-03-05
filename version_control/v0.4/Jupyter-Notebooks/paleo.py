import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.spatial.distance import squareform, pdist
from mastodon_functions import xml_features, csv_reader
import paleo_functions
from psutil import cpu_percent
from IPython.display import display
from scipy import signal
from sklearn.cluster import KMeans

__version__ = "0.1.4"

# Taking into account cell division -  Interactivity with all cells

class AllTracks(widgets.HBox):
    """This class is called to plot all the traces so the user can have an idea of how the data looks like.
    :class:
    :path_csv: Path where the two csv files (-edges.csv and -data.csv) are located. These are the files which are
        obtained from Mastodon
    :path_xml: Path where the .xml file created from BigDataViewer is located. This file should be in the same folder
        as the .hdf5 data
    """
    def __init__(self, path_csv, path_xml, tr_min):
        """Constructor method
        """
        super().__init__()

        # Load the data from the -csv which we organized in the MastodonFunctions.py
        self.spots = csv_reader(path_csv, path_xml)
        fts = xml_features(path_xml)

        # All possible intensity measures Mastodon can export
        intensity_labels = np.array(['Mean', 'Std', 'Min', 'Max', 'Median', 'Sum', 'Center'])  # For each channel

        # Description for all possible intensity labels
        description = np.array(['Mean', 'Standard deviation', 'Minimum intensity', 'Maximum intensity',
                                'Median intensity', 'Sum intensity', 'Center intensity'])

        # Which were the measures Mastodon exported?
        vals_to_check = np.array([i.split()[0] for i in list(self.spots.data_df_sorted.keys())])
        inds_vals = np.array([any(i == intensity_labels) for i in vals_to_check])
        intensity_labels_to_use = np.unique(vals_to_check[inds_vals])

        # Descriptions for measures to use
        descriptions_to_use = description[np.array([any(i == intensity_labels) for i in intensity_labels_to_use])]

        # The X is always the frames
        self.x = self.spots.data_df_sorted['Frames']

        # Number of channels
        self.n_channels = fts.channels

        # Mean, Median, Sum, Min, Max (with or without Std)
        self.y = []
        # Initialize the first plot
        self.y_value = intensity_labels_to_use[0]
        for i in range(self.n_channels):
            self.y.append(self.spots.data_df_sorted[f'{self.y_value} ch{i+1}'])

        self.tr_min = tr_min

        # Number of tracks in total
        self.N = len(np.unique(self.spots.data_df_sorted['Track ID'].to_numpy()))

        # default line color
        colors1 = ['green', 'limegreen', 'darkgreen', 'yellowgreen', 'seagreen', 'forestgreen', 'palegreen',
                   'darkolivegreen', 'lime']*self.N
        colors2 = ['hotpink', 'magenta', 'deeppink', 'pink', 'crimson', 'orchid', 'palevioletred', 'mediumvioletred',
                   'darkmagenta']*self.N
        colors3 = ['blue', 'navy', 'dodgerblue', 'royalblue', 'mediumblue', 'slateblue', 'cyan', 'darkturquoise',
                   'teal']*self.N

        self.channels = [f'Channel {i+1}' for i in range(self.n_channels)]

        output = widgets.Output()

        with output:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=(6, 4))

        self.colors = [colors1, colors2, colors3]

        # To setup min and max value for the intensities
        min_val = np.inf
        self.max_val = np.zeros(self.n_channels)
        time_val = 0

        # Initial number of cells to plot - default is 10
        self.N1 = 0
        self.N2 = 10

        # Initial value for CPU percentage
        self.cpu_per = cpu_percent()

        # By defualt, interactivity is disabled
        self.chose_interactivity = False

        # By default, alpha is 1
        self.chose_alpha = 1

        # Define min and max values for X-axis (time) and Y-axis (intensity)
        for c in range(self.n_channels):
            for idx in np.unique(self.spots.data_df_sorted['Track ID'])[self.N1:self.N2]:

                i = self.spots.data_df_sorted['Track ID'] == idx
                # Min value for intensities (y-axis)
                if np.min(self.y[c][i].to_numpy()) < min_val:
                    min_val = np.min(self.y[c][i].to_numpy())
                # Max value for intensities (y-axis)
                if np.max(self.y[c][i].to_numpy()) > self.max_val[c]:
                    self.max_val[c] = np.max(self.y[c][i].to_numpy())
                # Max value for time (x-axis)
                if np.max(self.x[i].to_numpy() * self.tr_min) > time_val:
                    time_val = np.max(self.x[i].to_numpy() * self.tr_min)

        # move the toolbar to the bottom
        self.fig.canvas.toolbar_position = 'bottom'
        self.ax.grid(True)

        # Control Elements
        # -----------------
        # Intensity labels to plot - Mean, Median, Sum, Max, Min, Std
        style = {'description_width': 'initial'}
        y_axis_values = widgets.ToggleButtons(options=intensity_labels_to_use.tolist(),
                                              description='Intensity Measures:', disabled=False, button_style='',
                                              tooltips=descriptions_to_use.tolist(), layout=dict(width='50%'),
                                              style=style)

        # Labels for x and y axis
        text_xlabel = widgets.Text(value='', description='X-Label', continuous_update=False, layout=dict(width='90%'))
        text_ylabel = widgets.Text(value='', description='Y-Label', continuous_update=False, layout=dict(width='90%'))

        # Title for plot
        text_title = widgets.Text(value='', description='Title', continuous_update=False, layout=dict(width='90%'))

        # Min-max slider for y axis (intensity)
        int_range_slider1 = widgets.IntRangeSlider(value=(min_val, np.max(self.max_val)), min=0,
                                                   max=np.max(self.max_val)*2, step=10, description='Y-Limit')

        # Min-max slider for x axis (time)
        int_range_slider2 = widgets.IntRangeSlider(value=(0, time_val), min=0, max=time_val*2, step=10,
                                                   description='X-Limit')

        # Normalize options: Yes, No
        style = {'description_width': 'initial'}
        dropbox2 = widgets.Dropdown(options=['Yes', 'No'], description='Normalize Data: ', value='No', rows=2,
                                    interactive=True, layout=dict(width='90%'), style=style)

        # Slider to select the number of cell tracks to plot
        int_range_slider_n = widgets.IntRangeSlider(value=(self.N1, self.N2), min=0, max=self.N, step=1,
                                                    description='# Cells to plot')

        # Channels to plot
        checkbox1_kwargs = {'value': True, 'disabled': False, 'indent': False}
        self.checkbox1_arg_dict = {channel: widgets.Checkbox(description=channel, **checkbox1_kwargs)
                                   for channel in self.channels}
        checkbox1 = widgets.VBox(children=[self.checkbox1_arg_dict[channel] for channel in self.channels])

        # Checkbox to select whether you want to include or not the dividing cell tracks
        exclude_divs_box = widgets.Checkbox(value=True, description='Exclude divisions', disabled=False, indent=False)

        # Bar to display the percentage of CPU used in each operation
        self.cpu_percentage = widgets.FloatProgress(value=self.cpu_per, min=0, max=100, description='CPU %:',
                                                    bar_style='', style={'bar_color': 'blue'}, orientation='horizontal')
        display(self.cpu_percentage)

        # Interactivity toggle button - if clicked, the plot button appears
        interactivity_button = widgets.ToggleButton(value=False, description='Interactivity', disabled=False,
                                                    button_style='', tool_tip='Disable interactivity mode',
                                                    icon='check')
        self.plot_button = widgets.Button(description='Plot', disabled=False, button_style='',
                                          tooltip='Plot the tracks', icon='')

        # Float slider to change the alpha of the lines
        style = {'description_width': 'initial'}  # This is so that the description fits in 1 line
        float_slider_alpha = widgets.FloatSlider(value=1.0, min=0, max=1.0, step=0.1, description='Transparency: ',
                                                 disabled=False, continuous_update=False, orientation='horizontal',
                                                 readout=True, readout_format='.1f', style=style)

        # Change line-width of lines
        float_slider_lw = widgets.FloatSlider(value=1.0, min=0, max=4.0, step=0.2, description='Linewidth: ',
                                              disabled=False, continuous_update=False, orientation='horizontal',
                                              readout=True, readout_format='.1f', style=style)

        # Connect callbacks and traits
        # -----------------------------
        int_range_slider1.observe(self.update1, 'value')
        int_range_slider2.observe(self.update2, 'value')
        int_range_slider_n.observe(self.update_n, 'value')
        text_xlabel.observe(self.update_xlabel, 'value')
        text_ylabel.observe(self.update_ylabel, 'value')
        text_title.observe(self.update_title, 'value')
        y_axis_values.observe(self.update_y, 'value')
        dropbox2.observe(self.update_norm, 'value')
        exclude_divs_box.observe(self.update_div_ex, 'value')
        interactivity_button.observe(self.update_interactivity, 'value')
        float_slider_alpha.observe(self.update_alpha, 'value')
        float_slider_lw.observe(self.update_lw, 'value')

        # Obtain the values from the widgets to use later in the functions
        # ----------------------------------------------------------------
        self.chose_channel = [True] * self.n_channels
        self.chose_norm = dropbox2.value
        self.x_label = text_xlabel.value
        self.y_label = text_ylabel.value
        self.title_label = text_title.value
        self.chose_y = y_axis_values.value
        self.chose_n = int_range_slider_n.value
        self.exclude_divs = exclude_divs_box.value
        self.chose_alpha = float_slider_alpha.value
        self.chose_lw = float_slider_lw.value
        self.chose_interactivity = interactivity_button.value

        # Interactive widgets which depend on other widgets
        # -------------------
        widgets.interactive_output(self.update_channel, self.checkbox1_arg_dict)
        out_manual = widgets.interact_manual(self.plot_lines)

        # Change the label from "Run Interact" to a specific one
        out_manual.widget.children[0].description = 'Plot'

        # Controls for the menu
        # ----------------------
        # Box for changing the plotting values, number of cell tracks, channels, exclude divisions...
        controls1 = widgets.VBox([interactivity_button, int_range_slider_n, exclude_divs_box,
                                  checkbox1, dropbox2, y_axis_values], layout={'width': '300px'})

        # Box for styling the plot - both the lines of the data and the skeleton of the plot
        controls2 = widgets.VBox([int_range_slider1, int_range_slider2, text_xlabel, text_ylabel, text_title,
                                  float_slider_alpha, float_slider_lw],
                                 layout={'width': '300px'})

        # Tabs for the menu
        # ------------------
        # Combine the controls from above into a single tab to show for the user
        tab = widgets.Tab([controls1, controls2], layout={'height': '80%'})
        tab.set_title(0, 'Plotting values')
        tab.set_title(1, 'Plotting axes')

        self.children = [tab, output]

        self.plot_lines()

    # Functions for widgets
    # ----------------------
    def update_lw(self, change):
        """Updates the value of the linewidth post-plotting
        Parameters
        ----------
        * self : figure
            The figure object created at the beginning. This way we can change values of the plotted figure instead of
            re-making it
        * change : line-width selected by the user in the float-slider
        Returns
        -------
        * chose_lw : float
            The value which the user selected from the float slider for the linewidth of the lines in the plot
        """
        self.chose_lw = change.new

        for line in self.fig.gca().lines:
            line.set_linewidth(change.new)

    def update_alpha(self, change):
        """Updates the value of the alpha value post-plotting
        Parameters
        ----------
        * self : figure
            The figure object created at the beginning. This way we can change values of the plotted figure instead of
            re-making it
        * change : alpha value selected by the user in the float-slider
        Returns
        -------
        * chose_alpha : float
            The value which the user selected from the float slider for the alpha of the lines in the plot
        """
        self.chose_alpha = change.new

        for line in self.fig.gca().lines:
            line.set_alpha(change.new)

    def update_interactivity(self, change):
        """Chose whether the interactivity should be disabled or not
        Parameters
        ----------
        * change : True or False
            The value from the Interactivity button clicked by the user to enable or disable interactivity
        Returns
        -------
        * chose_interactivity : bool
            The boolean value which the user selected from the interactivity button
        """

        self.chose_interactivity = change.new

    def update1(self, change):
        """Updates the values of the y-limits of the plot
        Parameters
        ----------
        * change : ax
            The ax object created at the beginning. This way we can change values of the y-axes limits

        Returns
        -------
        * array
            An array containing the min y limit and te max y limit
        """

        self.ax.set_ylim([change.new[0], change.new[1]])

        # redraw line (update plot)
        self.fig.canvas.draw_idle()

    def update2(self, change):
        """Updates the values of the x-limits of the plot
        Parameters
        ----------
        * change : ax
            The ax object created at the beginning. This way we can change values of the x-axes limits

        Returns
        -------
        * array
            An array containing the min x limit and te max x limit
        """

        self.ax.set_xlim([change.new[0], change.new[1]])

        # redraw line (update plot)
        self.fig.canvas.draw_idle()

    def update_n(self, change):
        """Obtain the number of cells to plot
        Parameters
        ----------
        * change : array
            Min value of cells to start the plot and max value of the cells to stop the plot. E.g. if you have 100 cells
             in total but you only want to plot cells from 20-60, these are the two values exported in this function

        Returns
        -------
        * choose_n : array
            Min value of cells to start the plot and max value of the cells to stop the plot. E.g. if you have 100
            but you only want to plot cells from 20-60, these are the two values exported in this function
        """
        self.chose_n = change.new

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                """Plot the lines when the user clicks the Plot button"""
                self.plot_lines()

    def update_div_ex(self, change):
        # If true, exclude cell division
        self.exclude_divs = change.new

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                self.plot_lines()

    def update_channel(self, **kwargs):
        self.chose_channel = []
        for key in kwargs:
            self.chose_channel.append(kwargs[key])

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                self.plot_lines()

    def update_norm(self, change):
        self.chose_norm = change.new
        '''
        if change.new == 'Yes':
            self.int_range_slider1.max = 1
        else:
            self.int_range_slider1.max = np.max(self.max_val)
        '''

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                self.plot_lines()

    def update_y(self, change):
        self.chose_y = change.new

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                self.plot_lines()

    def plot_lines(self):
        self.ax.clear()

        inds_channel = np.where(np.array(self.chose_channel))[0]

        # Use the measure the user chose
        y_value = self.chose_y

        self.N1 = self.chose_n[0]
        self.N2 = self.chose_n[1]

        ex_divs = self.exclude_divs

        # Alpha value
        alpha = self.chose_alpha

        # Line-width value
        lw = self.chose_lw

        self.y = []

        for i in range(self.n_channels):
            self.y.append(self.spots.data_df_sorted[f'{y_value} ch{i+1}'])

        if self.chose_norm == 'Yes':

            self.max_val = np.zeros(self.n_channels)
            # Calculate the new max intensity value (depending on the intensity measure chosen by user)
            for c in inds_channel:
                for idx in np.unique(self.spots.data_df_sorted['Track ID'])[self.N1:self.N2]:

                    i = self.spots.data_df_sorted['Track ID'] == idx
                    n_links = self.spots.data_df_sorted.loc[i, 'N Links'].to_numpy()

                    # If we clicked to exclude divisions and we find a division, we exclude it
                    if ex_divs is True and np.max(n_links) == 3:
                        continue
                    else:
                        if np.max(self.y[c][i].to_numpy()) > self.max_val[c]:
                            self.max_val[c] = np.max(self.y[c][i].to_numpy())

            # Plot the data
            for c in inds_channel:
                for i, val in enumerate(np.unique(self.spots.data_df_sorted['Track ID'])[self.N1:self.N2]):

                    idx = self.spots.data_df_sorted['Track ID'] == val
                    n_links = self.spots.data_df_sorted.loc[idx, 'N Links'].to_numpy()

                    # If we clicked to exclude divisions and we find a division, we exclude it
                    if ex_divs is True and np.max(n_links) == 3:
                        continue

                    # if we did not click to exclude divisions and we do not find a division
                    elif ex_divs is False and np.max(n_links) == 3:

                        spot_source = self.spots.edges_df_sorted['source ID'].to_numpy(dtype=int)
                        spot_target = self.spots.edges_df_sorted['target ID'].to_numpy(dtype=int)
                        spots_id = self.spots.data_df_sorted.loc[idx, 'Spot ID'].values

                        frames_out, intensity_out = self.cell_division_arrange(self.x[idx].to_numpy(), n_links,
                                                                               spots_id, self.y[c][idx].to_numpy() /
                                                                               self.max_val[c],
                                                                               spot_source, spot_target)
                        for div_i in range(len(frames_out)):
                            self.line, = self.ax.plot(frames_out[div_i]*self.tr_min,
                                                      intensity_out[div_i],
                                                      color=self.colors[c][i],
                                                      alpha=alpha, linewidth=lw)

                    # if we clicked to exclude divisions or not but there are no divisions, plot
                    else:
                        self.line, = self.ax.plot(self.x[idx].to_numpy() * self.tr_min,
                                                  self.y[c][idx].to_numpy() / self.max_val[c], color=self.colors[c][i],
                                                  alpha=alpha, linewidth=lw)

            self.fig.canvas.draw_idle()

        else:
            for c in inds_channel:
                for i, val in enumerate(np.unique(self.spots.data_df_sorted['Track ID'])[self.N1:self.N2]):

                    idx = self.spots.data_df_sorted['Track ID'] == val
                    n_links = self.spots.data_df_sorted.loc[idx, 'N Links'].to_numpy()

                    # If we clicked to exclude divisions and we find a division, we exclude it
                    if ex_divs is True and np.max(n_links) == 3:
                        continue

                    # if we did not click to exclude divisions and we do not find a division
                    elif ex_divs is False and np.max(n_links) == 3:

                        spot_source = self.spots.edges_df_sorted['source ID'].to_numpy(dtype=int)
                        spot_target = self.spots.edges_df_sorted['target ID'].to_numpy(dtype=int)
                        spots_id = self.spots.data_df_sorted.loc[idx, 'Spot ID'].values

                        frames_out, intensity_out = self.cell_division_arrange(self.x[idx].to_numpy(), n_links,
                                                                               spots_id,
                                                                               self.y[c][idx].to_numpy(),
                                                                               spot_source, spot_target)
                        for div_i in range(len(frames_out)):
                            self.line, = self.ax.plot(frames_out[div_i] * self.tr_min, intensity_out[div_i],
                                                      color=self.colors[c][i], alpha=alpha, linewidth=lw)

                    # if we clicked to exclude divisions or not but there are no divisions, plot
                    else:
                        self.line, = self.ax.plot(self.x[idx].to_numpy() * self.tr_min, self.y[c][idx].to_numpy(),
                                                  color=self.colors[c][i], alpha=alpha, linewidth=lw)

            self.fig.canvas.draw_idle()

        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_title(self.title_label)
        self.cpu_percentage.value = cpu_percent()

    def update_xlabel(self, change):
        self.x_label = change.new
        self.ax.set_xlabel(change.new, fontsize=20)

    def update_ylabel(self, change):
        self.y_label = change.new
        self.ax.set_ylabel(change.new, fontsize=20)

    def update_title(self, change):
        self.title_label = change.new
        self.ax.set_title(change.new, fontsize=25)

    @staticmethod
    def cell_division_arrange(frames, n_links, spots_id, intensity, spot_source, spot_target):
        """Orders the cell tracks that divide

            Arguments:
                * frames: Time points (also called frames) for a given cell track
                * n_links: Number of links for each spot in a cell track
                * spots_id: The ID for each individual spot for all the spots in the given cell track
                * intensity: Intensity values for a given cell track
                * spot_source: Spot ID sources
                * spot_target: Spot ID target

            Returns:
                * frames_out: Array of the size of the number of divisions. Each array contains all the frames
                * intensity_out: Array of the size of the number of divisions. Each array contains all the intensities
        """
        # ORDER TRACK BY ITS ID
        sorted_frames = frames.argsort()

        # Save the sorted frames
        frames_by_track = np.sort(frames)

        # Save the sorted links (1, 2 or 3 links in case of division)
        links_by_track = n_links[sorted_frames]

        # spots_id by tracks
        spots_id_by_track = spots_id[sorted_frames]

        # Intensity by tracks
        intensity_by_track = intensity[sorted_frames]

        # Sorted spot source ID and spot target ID
        source_by_track = spot_source[np.array([ind for ind, element in enumerate(spot_source) if element in
                                                spots_id_by_track[:-1]])]
        target_by_track = spot_target[np.array([ind for ind, element in enumerate(spot_target) if element in
                                                spots_id_by_track[1:]])]

        frames_out = []
        intensity_out = []

        # Are there any divisions in the track?
        # (e.g. the spot divides in two different opportunities during all the timeseries)
        n_divs = len(list(map(int, np.where(links_by_track > 2)[0])))

        # How many times the spot divides per division?
        # (e.g. in one specific division, in how many daughters the spot divided?)
        # n_divs_cell = links_by_track[links_by_track[i]>2]

        val0 = np.where(links_by_track == 3)[0][0]  # index of first division
        div_vect = []
        # save the spots_id up to first division for all tracks
        for j in range(n_divs + 1):
            div_vect.append(spots_id_by_track[:val0 + 1].tolist())  # spots_id[0:first division]

        # store the list of already saved spots_id to not use them again
        list_idx_sources_used = []  # To save the indices of used sources spots

        # list of spots_id used --> finish while loop when all spots_id used
        list_used = spots_id_by_track[:val0 + 1].tolist()

        # while we have not stored all spots_id, loop across tracks and fill them with targets (if not in list_used)
        while not (all(elem in list_used for elem in spots_id_by_track)):

            for j in range(n_divs + 1):
                idx = np.where(source_by_track == div_vect[j][-1])[0]
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
                    div_vect[j].append(int(target_by_track[idx]))
                    list_used.append(int(target_by_track[idx]))

                # This means it finished at least one of the tracks
                if np.size(idx) == 0:
                    continue

        # Save each division tracks with its corresponding division ID and its frames and mean
        for j in range(n_divs + 1):
            # Indices for the spots_id of the tracks in one of the divisions
            inds = [np.where(spots_id_by_track == div_vect[j][ind])[0][0] for ind in range(len(div_vect[j]))]
            frames_out.append(frames_by_track[inds])
            intensity_out.append(intensity_by_track[inds])

        return frames_out, intensity_out


# Interactive Traces with individual cells but choosing the channel or channels to plot
class IndividualTracks(widgets.HBox):
    """This class is called to plot individual traces so that the user can has an idea of how the data looks like.
    :class:
    :path_csv: Path where the two csv files (-edges.csv and -data.csv) are located. These are the files which are
        obtained from Mastodon
    :path_xml: Path where the .xml file created from BigDataViewer is located. This file should be in the same folder
        as the .hdf5 data
    :tr_min: Temporal resolution with which the data was acquired
    """
    def __init__(self, path_csv, path_xml, tr_min):
        """Constructor method
        """
        super().__init__()

        # Load the data from the -csv which we organized in the MastodonFunctions.py
        self.spots = csv_reader(path_csv, path_xml)
        fts = xml_features(path_xml)

        # The X is always the frames
        self.x = self.spots.data_df_sorted['Frames']

        # All possible intensity measures Mastodon can export
        intensity_labels = np.array(['Mean', 'Std', 'Min', 'Max', 'Median', 'Sum', 'Center'])  # For each channel

        # Description for all possible intensity labels
        description = np.array(['Mean', 'Standard deviation', 'Minimum intensity', 'Maximum intensity',
                                'Median intensity', 'Sum intensity', 'Center intensity'])

        # Which were the measures Mastodon exported?
        vals_to_check = np.array([i.split()[0] for i in list(self.spots.data_df_sorted.keys())])
        inds_vals = np.array([any(i == intensity_labels) for i in vals_to_check])
        intensity_labels_to_use = np.unique(vals_to_check[inds_vals])

        # Descriptions for measures to use
        descriptions_to_use = description[np.array([any(i == intensity_labels) for i in intensity_labels_to_use])]

        # Number of channels
        self.n_channels = fts.channels

        # Mean, Median, Sum, Min, Max (with or sithout Std)
        self.y = []
        # Initialize the first plot
        self.y_value = intensity_labels_to_use[0]
        for i in range(self.n_channels):
            self.y.append(self.spots.data_df_sorted[f'{self.y_value} ch{i+1}'])

        self.tr_min = tr_min

        # Number of tracks in total
        self.N = len(np.unique(self.spots.data_df_sorted['Track ID'].to_numpy()))

        # Number of channels
        self.n_channels = fts.channels

        self.channels = [f'Channel {i+1}' for i in range(self.n_channels)]

        # Initial value for CPU percentage
        self.cpu_per = cpu_percent()

        # By default, interactivity is disabled
        self.chose_interactivity = False

        # By default, alpha is 1
        self.chose_alpha = 1

        output = widgets.Output()

        # default line color - magenta, lime, blue, orange, dark purple
        initial_color = ['#FF00DD', '#00FF00', '#0000FF', '#FF5733', '#581845'] * self.N

        with output:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=(6, 4))

        # To setup min and max value for the intensities
        min_val = np.inf
        self.max_val = np.zeros(self.n_channels)
        time_val = 0

        # Initial value for CPU percentage
        self.cpu_per = cpu_percent()

        # By default, interactivity is disabled
        self.chose_interactivity = False

        # By default, alpha is 1
        self.chose_alpha = 1

        # Define min and max values for X-axis (time) and Y-axis (intensity)
        # Also, define which cells IDs contain division to ignore them in the "exclude division" option
        self.n_links = np.zeros(len(np.unique(self.spots.data_df_sorted['Track ID'])))

        for c in range(self.n_channels):
            for idx_ind, idx in enumerate(np.unique(self.spots.data_df_sorted['Track ID'])):

                i = self.spots.data_df_sorted['Track ID'] == idx
                if c == 0:
                    self.n_links[idx_ind] = np.max(self.spots.data_df_sorted.loc[i, 'N Links'].to_numpy())

                # Min value for intensities (y-axis)
                if np.min(self.y[c][i].to_numpy()) < min_val:
                    min_val = np.min(self.y[c][i].to_numpy())
                # Max value for intensities (y-axis)
                if np.max(self.y[c][i].to_numpy()) > self.max_val[c]:
                    self.max_val[c] = np.max(self.y[c][i].to_numpy())
                # Max value for time (x-axis)
                if np.max(self.x[i].to_numpy()*self.tr_min) > time_val:
                    time_val = np.max(self.x[i].to_numpy()*self.tr_min)

        self.no_div_inds = self.n_links <= 2

        # move the toolbar to the bottom
        self.fig.canvas.toolbar_position = 'bottom'
        self.ax.grid(True)

        # Control Elements
        # -----------------
        # Intensity labels to plot - Mean, Median, Sum, Max, Min, Std
        style = {'description_width': 'initial'}
        y_axis_values = widgets.ToggleButtons(options=intensity_labels_to_use.tolist(), description='Measures:',
                                              disabled=False, button_style='', tooltips=descriptions_to_use.tolist(),
                                              layout=dict(width='50%'), style=style)

        # Cell track to plot
        self.int_slider = widgets.IntSlider(value=0, min=0, max=self.N-1, step=1, description='Cell #')

        # Labels for x and y axis
        text_xlabel = widgets.Text(value='', description='X-Label', continuous_update=False, layout=dict(width='90%'))
        text_ylabel = widgets.Text(value='', description='Y-Label', continuous_update=False, layout=dict(width='90%'))

        # Title for plot
        text_title = widgets.Text(value='', description='Title', continuous_update=False, layout=dict(width='90%'))

        # Min-max slider for y axis (intensity)
        int_range_slider1 = widgets.IntRangeSlider(value=(min_val, np.max(self.max_val)), min=0,
                                                   max=np.max(self.max_val)*2, step=10, description='Y-Limit')

        # Min-max slider for x axis (time)
        int_range_slider2 = widgets.IntRangeSlider(value=(0, time_val), min=0, max=time_val*2, step=10,
                                                   description='X-Limit')

        # Normalize options: Yes, No
        dropbox2 = widgets.Dropdown(options=['Yes', 'No'], description='Normalize Data: ', value='No', rows=2,
                                    interactive=True, layout=dict(width='90%'), style=style)

        # Checkbox for the channels to plot
        checkbox1_kwargs = {'value': True, 'disabled': False, 'indent': False}
        self.checkbox1_arg_dict = {channel: widgets.Checkbox(description=channel, **checkbox1_kwargs) for channel in
                                   self.channels}
        checkbox1 = widgets.VBox(children=[self.checkbox1_arg_dict[channel] for channel in self.channels])

        # Checkbox to select whether you want to include or not the dividing cell tracks
        exclude_divs_box = widgets.Checkbox(value=True, description='Exclude divisions', disabled=False, indent=False)

        # Bar to display the percentage of CPU used in each operation
        self.cpu_percentage = widgets.FloatProgress(value=self.cpu_per, min=0, max=100, description='CPU %:',
                                                    bar_style='', style={'bar_color': 'blue'}, orientation='horizontal')
        display(self.cpu_percentage)

        # Interactivity toggle button - if clicked, the plot button appears
        interactivity_button = widgets.ToggleButton(value=False, description='Interactivity', disabled=False,
                                                    button_style='', tool_tip='Disable interactivity mode',
                                                    icon='check')
        self.plot_button = widgets.Button(description='Plot', disabled=False, button_style='',
                                          tooltip='Plot the tracks', icon='')

        # Several color pickers according to number of channels
        self.checkbox2_arg_dict = {channel: widgets.ColorPicker(value=initial_color[i],
                                                                description='Color for channel %d' % i, style=style,
                                                                layout=dict(width='90%'))
                                   for i, channel in enumerate(self.channels)}
        checkbox2 = widgets.VBox(children=[self.checkbox2_arg_dict[channel] for channel in self.channels])

        # Float slider to change the alpha of the lines
        style = {'description_width': 'initial'}  # This is so that the description fits in 1 line
        float_slider_alpha = widgets.FloatSlider(value=1.0, min=0, max=1.0, step=0.1, description='Transparency: ',
                                                 disabled=False, continuous_update=False, orientation='horizontal',
                                                 readout=True, readout_format='.1f', style=style)

        # Change line-width of lines
        float_slider_lw = widgets.FloatSlider(value=1.0, min=0, max=4.0, step=0.2, description='Linewidth: ',
                                              disabled=False, continuous_update=False, orientation='horizontal',
                                              readout=True, readout_format='.1f', style=style)

        # Fix the X axis and the Y axis
        checkbox_xaxis = widgets.Checkbox(description='Fix X-Lim', value=False, disabled=False, indent=False)
        checkbox_yaxis = widgets.Checkbox(description='Fix Y-Lim', value=False, disabled=False, indent=False)

        # Connect callbacks and traits
        # -----------------------------
        self.int_slider.observe(self.update, 'value')
        int_range_slider1.observe(self.update1, 'value')
        int_range_slider2.observe(self.update2, 'value')
        text_xlabel.observe(self.update_xlabel, 'value')
        text_ylabel.observe(self.update_ylabel, 'value')
        text_title.observe(self.update_title, 'value')
        y_axis_values.observe(self.update_y, 'value')
        exclude_divs_box.observe(self.update_div_ex, 'value')
        interactivity_button.observe(self.update_interactivity, 'value')
        float_slider_alpha.observe(self.update_alpha, 'value')
        float_slider_lw.observe(self.update_lw, 'value')

        self.x_label = text_xlabel.value
        self.y_label = text_ylabel.value
        self.title_label = text_title.value

        widgets.interactive_output(self.update_channel, self.checkbox1_arg_dict)
        widgets.interactive_output(self.update_color, self.checkbox2_arg_dict)

        checkbox_xaxis.observe(self.update_xaxis, 'value')
        checkbox_yaxis.observe(self.update_yaxis, 'value')

        dropbox2.observe(self.update_norm, 'value')

        # Obtain the values from the widgets to use later in the functions
        # ----------------------------------------------------------------
        self.chose_channel = [True] * self.n_channels

        # Initially, the axis are not fixed
        self.chose_fix_x = False
        self.chose_fix_y = False

        # First cell to plot is cell 0
        self.chose_cell = 0

        # The initial colors to show are some pre-selected by me ;)
        self.chose_color = [initial_color[i] for i in range(self.n_channels)]
        # Default value for normalization defined when creating Dropbox widget
        self.chose_norm = dropbox2.value

        self.x_label = text_xlabel.value
        self.y_label = text_ylabel.value
        self.title_label = text_title.value
        self.chose_y = y_axis_values.value
        # Initial y and x lim value
        self.x_lim = [0, time_val]
        self.y_lim = [min_val, np.max(self.max_val)]
        self.exclude_divs = exclude_divs_box.value
        self.chose_alpha = float_slider_alpha.value
        self.chose_lw = float_slider_lw.value
        self.chose_interactivity = interactivity_button.value

        # Interactive widgets which depend on other widgets
        # -------------------------------------------------
        widgets.interactive_output(self.update_channel, self.checkbox1_arg_dict)
        out_manual = widgets.interact_manual(self.plot_lines)

        # Change the label from "Run Interact" to a specific one
        out_manual.widget.children[0].description = 'Plot'

        controls1 = widgets.VBox([interactivity_button, self.int_slider, exclude_divs_box, checkbox1, dropbox2,
                                  y_axis_values], layout={'width': '300px'})
        controls2 = widgets.VBox([int_range_slider1, int_range_slider2, checkbox_xaxis, checkbox_yaxis, text_xlabel,
                                  text_ylabel, text_title, float_slider_alpha, float_slider_lw, checkbox2],
                                 layout={'width': '300px'})

        # Tabs for the menu
        # ------------------
        # Combine the controls from above into a single tab to show for the user
        tab = widgets.Tab([controls1, controls2])
        tab.set_title(0, 'Plotting values')
        tab.set_title(1, 'Plotting axes')

        self.children = [tab, output]

        self.plot_lines()

    # Functions for widgets
    # ----------------------

    def update_lw(self, change):
        """Updates the value of the linewidth post-plotting
        Parameters
        ----------
        * self : figure
            The figure object created at the beginning. This way we can change values of the plotted figure instead of
            re-making it
        * change : line-width selected by the user in the float-slider
        Returns
        -------
        * chose_lw : float
            The value which the user selected from the float slider for the linewidth of the lines in the plot
        """
        self.chose_lw = change.new

        for line in self.fig.gca().lines:
            line.set_linewidth(change.new)

    def update_alpha(self, change):
        """Updates the value of the alpha value post-plotting
        Parameters
        ----------
        * self : figure
            The figure object created at the beginning. This way we can change values of the plotted figure instead of
            re-making it
        * change : alpha value selected by the user in the float-slider
        Returns
        -------
        * chose_alpha : float
            The value which the user selected from the float slider for the alpha of the lines in the plot
        """
        self.chose_alpha = change.new

        for line in self.fig.gca().lines:
            line.set_alpha(change.new)

    def update_interactivity(self, change):
        """Chose whether the interactivity should be disabled or not
        Parameters
        ----------
        * change : True or False
            The value from the Interactivity button clicked by the user to enable or disable interactivity
        Returns
        -------
        * chose_interactivity : bool
            The boolean value which the user selected from the interactivity button
        """

        self.chose_interactivity = change.new

    def update(self, change):
        self.chose_cell = change.new

        if self.exclude_divs is True:
            self.int_slider.max = self.N - len(self.n_links[self.n_links > 2]) - 1
        else:
            self.int_slider.max = self.N - 1

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                """Plot the lines when the user clicks the Plot button"""
                self.plot_lines()

    def update_div_ex(self, change):
        # If true, exclude cell division
        self.exclude_divs = change.new

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                self.plot_lines()

    def update_y(self, change):
        self.chose_y = change.new

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                self.plot_lines()

    def plot_lines(self):

        self.ax.clear()

        # Selected channels by the user
        inds_channel = np.where(np.array(self.chose_channel))[0]

        # Use the measure the user chose
        y_value = self.chose_y

        self.y = []

        for i in range(self.n_channels):
            self.y.append(self.spots.data_df_sorted[f'{y_value} ch{i+1}'])

        # If exclude division was selected, this variable will be True
        ex_divs = self.exclude_divs

        # Cell to plot, selected by the user
        if ex_divs is True:
            cell = np.unique(self.spots.data_df_sorted['Track ID'])[self.no_div_inds][self.chose_cell]
        else:
            cell = np.unique(self.spots.data_df_sorted['Track ID'])[self.chose_cell]

        # Selected colors for each channel by the user
        col = self.chose_color

        # Line style for dividing cells
        ls = ['solid', 'dashed', 'dotted']

        # Alpha value
        alpha = self.chose_alpha

        # Line-width value
        lw = self.chose_lw

        # Check the number of links to know whether the cell divides
        idx_ = self.spots.data_df_sorted['Track ID'] == cell

        if self.chose_norm == 'Yes':
            # Find the maximum value to normalize
            self.max_val = np.zeros(self.n_channels)
            # Calculate the new max intensity value (depending on the intensity measure chosen by user)
            for c in inds_channel:
                for idx_ind, idx in enumerate(np.unique(self.spots.data_df_sorted['Track ID'])):

                    i = self.spots.data_df_sorted['Track ID'] == idx

                    # If we clicked to exclude divisions and we find a division, we exclude it
                    if ex_divs is True and self.n_links[idx_ind] == 3:
                        continue
                    else:
                        if np.max(self.y[c][i].to_numpy()) > self.max_val[c]:
                            self.max_val[c] = np.max(self.y[c][i].to_numpy())

            for c in inds_channel:
                if self.n_links[cell] > 2:
                    spot_source = self.spots.edges_df_sorted['source ID'].to_numpy(dtype=int)
                    spot_target = self.spots.edges_df_sorted['target ID'].to_numpy(dtype=int)
                    spots_id = self.spots.data_df_sorted.loc[idx_, 'Spot ID'].values
                    n_links_all = self.spots.data_df_sorted.loc[idx_, 'N Links'].values

                    frames_out, intensity_out = self.cell_division_arrange(self.x[cell].to_numpy(), n_links_all,
                                                                           spots_id, self.y[c][cell].to_numpy() /
                                                                           self.max_val[c], spot_source, spot_target)
                    for div_i in range(len(frames_out)):
                        self.line, = self.ax.plot(frames_out[div_i] * self.tr_min, intensity_out[div_i],
                                                  color=col[c], linewidth=lw, linestyle=ls[div_i], alpha=alpha)
                else:
                    self.line, = self.ax.plot(self.x[cell].to_numpy()*self.tr_min,
                                              self.y[c][cell].to_numpy()/self.max_val[c], color=col[c], linewidth=lw,
                                              alpha=alpha)

            self.fig.canvas.draw_idle()

        else:
            for c in inds_channel:
                if self.n_links[cell] > 2:
                    spot_source = self.spots.edges_df_sorted['source ID'].to_numpy(dtype=int)
                    spot_target = self.spots.edges_df_sorted['target ID'].to_numpy(dtype=int)
                    spots_id = self.spots.data_df_sorted.loc[idx_, 'Spot ID'].values
                    n_links_all = self.spots.data_df_sorted.loc[idx_, 'N Links'].values

                    frames_out, intensity_out = self.cell_division_arrange(self.x[cell].to_numpy(), n_links_all,
                                                                           spots_id, self.y[c][cell].to_numpy(),
                                                                           spot_source, spot_target)
                    for div_i in range(len(frames_out)):
                        self.line, = self.ax.plot(frames_out[div_i] * self.tr_min, intensity_out[div_i], color=col[c],
                                                  linewidth=lw, linestyle=ls[div_i], alpha=alpha)
                else:
                    self.line, = self.ax.plot(self.x[cell].to_numpy()*self.tr_min, self.y[c][cell].to_numpy(),
                                              color=col[c], linewidth=lw, alpha=alpha)

            self.fig.canvas.draw_idle()

        # Fix y and x axis according to whether the fix Checkbox was checked
        if self.chose_fix_x:
            self.ax.set_xlim([self.x_lim[0], self.x_lim[1]])
        if self.chose_fix_y:
            self.ax.set_ylim([self.y_lim[0], self.y_lim[1]])

        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_title(self.title_label)

        self.cpu_percentage.value = cpu_percent()

    def update1(self, change):
        """redraw line (update plot)"""
        self.ax.set_ylim([change.new[0], change.new[1]])
        self.y_lim = change.new

        self.fig.canvas.draw_idle()

    def update2(self, change):
        """redraw line (update plot)"""
        self.ax.set_xlim([change.new[0], change.new[1]])
        self.x_lim = change.new

        self.fig.canvas.draw_idle()

    def update_xaxis(self, change):
        self.chose_fix_x = change.new

    def update_yaxis(self, change):
        self.chose_fix_y = change.new

    def update_channel(self, **kwargs):
        self.chose_channel = []
        for key in kwargs:
            self.chose_channel.append(kwargs[key])

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                self.plot_lines()

    def update_norm(self, change):
        self.chose_norm = change.new

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                self.plot_lines()

    def update_color(self, **kwargs):
        """set line color"""
        self.chose_color = []
        for key in kwargs:
            self.chose_color.append(kwargs[key])

        for i, line in enumerate(self.fig.gca().lines):
            line.set_color(self.chose_color[i])

    def update_xlabel(self, change):
        self.x_label = change.new
        self.ax.set_xlabel(change.new, fontsize=20)

    def update_ylabel(self, change):
        self.y_label = change.new
        self.ax.set_ylabel(change.new, fontsize=20)

    def update_title(self, change):
        self.title_label = change.new
        self.ax.set_title(change.new, fontsize=25)

    @staticmethod
    def cell_division_arrange(frames, n_links, spots_id, intensity, spot_source, spot_target):
        # ORDER TRACK BY ITS ID

        sorted_frames = frames.argsort()

        # Save the sorted frames
        frames_by_track = np.sort(frames)

        # Save the sorted links (1, 2 or 3 links in case of division)
        links_by_track = n_links[sorted_frames]

        # spots_id by tracks
        spots_id_by_track = spots_id[sorted_frames]

        # Intensity by tracks
        intensity_by_track = intensity[sorted_frames]

        # Sorted spot source ID and spot target ID

        source_by_track = spot_source[np.array([ind for ind, element in enumerate(spot_source) if element in
                                                spots_id_by_track[:-1]])]
        target_by_track = spot_target[np.array([ind for ind, element in enumerate(spot_target) if element in
                                                spots_id_by_track[1:]])]

        frames_out = []
        intensity_out = []

        # Are there any divisions in the track?
        # (e.g. the spot divides in two different opportunities during all the timeseries)
        n_divs = len(list(map(int, np.where(links_by_track > 2)[0])))

        # How many times the spot divides per division?
        # (e.g. in one specific division, in how many daughters the spot divided?)
        # n_divs_cell = links_by_track[links_by_track[i]>2]

        div_vect = []  # one vector for [each division+1] we want to keep track of
        val0 = np.where(links_by_track == 3)[0][0]  # index of first division

        # save the spots_id up to first division for all tracks
        for j in range(n_divs + 1):
            div_vect.append(spots_id_by_track[:val0 + 1].tolist())  # spots_id[0:first division]

        # store the list of already saved spots_id to not use them again
        list_idx_sources_used = []  # To save the indices of used sources spots
        list_used = spots_id_by_track[
                    :val0 + 1].tolist()  # list of spots_id used --> finish while loop when all spots_id used
        # while we have not stored all spots_id, loop across tracks and fill them with targets (if not in list_used)

        while not (all(elem in list_used for elem in spots_id_by_track)):

            for j in range(n_divs + 1):
                idx = np.where(source_by_track == div_vect[j][-1])[0]
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
                    div_vect[j].append(int(target_by_track[idx]))
                    list_used.append(int(target_by_track[idx]))

                # This means it finished at least one of the tracks
                if np.size(idx) == 0:
                    continue

        # Save each division tracks with its corresponding division ID and its frames and mean
        for j in range(n_divs + 1):
            # Indices for the spots_id of the tracks in one of the divisions
            inds = [np.where(spots_id_by_track == div_vect[j][ind])[0][0] for ind in range(len(div_vect[j]))]
            frames_out.append(frames_by_track[inds])
            intensity_out.append(intensity_by_track[inds])

        return frames_out, intensity_out


class AllTracksTags(widgets.HBox):
    """# Interactivity with cell tags and subtags - all tracks are plotted together
    """
    def __init__(self, path_csv, path_xml, tr_min):

        super().__init__()

        # Load the data from the -csv which we organized in the MastodonFunctions.py
        self.spots = csv_reader(path_csv, path_xml)
        fts = xml_features(path_xml)

        # All possible intensity measures Mastodon can export
        intensity_labels = np.array(['Mean', 'Std', 'Min', 'Max', 'Median', 'Sum', 'Center'])  # For each channel

        # Description for all possible intensity labels
        description = np.array(['Mean', 'Standard deviation', 'Minimum intensity', 'Maximum intensity',
                                'Median intensity', 'Sum intensity', 'Center intensity'])
        # Unique tags to plot
        all_tags = np.unique(self.spots.tags)

        # Which were the measures Mastodon exported?
        vals_to_check = np.array([i.split()[0] for i in list(self.spots.data_df_sorted.keys())])
        inds_vals = np.array([any(i == intensity_labels) for i in vals_to_check])
        intensity_labels_to_use = np.unique(vals_to_check[inds_vals])

        # Descriptions for measures to use
        descriptions_to_use = description[np.array([any(i == intensity_labels) for i in intensity_labels_to_use])]

        # The X is always the frames
        self.x = self.spots.data_df_sorted['Frames']

        # Number of channels
        self.n_channels = fts.channels

        # Mean, Median, Sum, Min, Max (with or sithout Std)
        self.y = []
        # Initialize the first plot
        self.y_value = intensity_labels_to_use[0]
        for i in range(self.n_channels):
            self.y.append(self.spots.data_df_sorted[f'{self.y_value} ch{i+1}'])

        self.tr_min = tr_min

        # Number of tracks in total
        self.N = len(np.unique(self.spots.data_df_sorted['Track ID'].to_numpy()))

        # default line color
        colors1 = ['green', 'limegreen', 'darkgreen', 'yellowgreen', 'seagreen', 'forestgreen', 'palegreen',
                   'darkolivegreen', 'lime']*self.N
        colors2 = ['hotpink', 'magenta', 'deeppink', 'pink', 'crimson', 'orchid', 'palevioletred', 'mediumvioletred',
                   'darkmagenta']*self.N
        colors3 = ['blue', 'navy', 'dodgerblue', 'royalblue', 'mediumblue', 'slateblue', 'cyan', 'darkturquoise',
                   'teal']*self.N

        self.channels = [f'Channel {i+1}' for i in range(self.n_channels)]

        output = widgets.Output()

        with output:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=(6, 4))

        self.colors = [colors1, colors2, colors3]

        # To setup min and max value for the intensities
        min_val = np.inf
        self.max_val = np.zeros(self.n_channels)
        time_val = 0

        # Define min and max values for X-axis (time) and Y-axis (intensity)
        for c in range(self.n_channels):
            for idx in np.unique(self.spots.data_df_sorted['Track ID']):

                i = self.spots.data_df_sorted['Track ID'] == idx

                if np.min(self.y[c][i].to_numpy()) < min_val:
                    min_val = np.min(self.y[c][i].to_numpy())

                if np.max(self.y[c][i].to_numpy())>self.max_val[c]:
                    self.max_val[c] = np.max(self.y[c][i].to_numpy())

                if np.max(self.x[i].to_numpy()*self.tr_min) > time_val:
                    time_val=np.max(self.x[i].to_numpy()*self.tr_min)

        # move the toolbar to the bottom
        self.fig.canvas.toolbar_position = 'bottom'
        self.ax.grid(True)

        # create some control elements
        y_axis_values = widgets.ToggleButtons(options=intensity_labels_to_use.tolist(), description='Measures:',
                                             disabled=False, button_style='', tooltips=descriptions_to_use.tolist())
                                              #layout=widgets.Layout(width='10%', height='80px'))
        text_xlabel = widgets.Text(value='', description='X-Label', continuous_update=False)
        text_ylabel = widgets.Text(value='', description='Y-Label', continuous_update=False)
        text_title = widgets.Text(value='', description='Title', continuous_update=False)
        int_range_slider1 = widgets.IntRangeSlider(value=(min_val, np.max(self.max_val)), min=0, max=np.max(self.max_val)*2, step=10, description='Y-Limit')
        int_range_slider2 = widgets.IntRangeSlider(value=(0, time_val), min=0, max=time_val*2, step=10, description='X-Limit')
        #dropbox1 = widgets.Dropdown(options=options_channels, description='Choose Channel',
        #			value=options_channels[0],rows=len(options_channels), interactive=True)
        dropbox2 = widgets.Dropdown(options=['Yes', 'No'], description='Normalize Data',
                    value='No',rows=2, interactive=True)

        checkbox1_kwargs = {'value':True, 'disabled':False, 'indent':False}
        self.checkbox1_arg_dict = {channel: widgets.Checkbox(description=channel, **checkbox1_kwargs) for channel in self.channels}
        checkbox1 = widgets.VBox(children=[self.checkbox1_arg_dict[channel] for channel in self.channels])

        # connect callbacks and traits
        int_range_slider1.observe(self.update1, 'value')
        int_range_slider2.observe(self.update2, 'value')
        text_xlabel.observe(self.update_xlabel, 'value')
        text_ylabel.observe(self.update_ylabel, 'value')
        text_title.observe(self.update_title, 'value')
        y_axis_values.observe(self.update_y, 'value')

        #dropbox1.observe(self.update_channel, 'value')
        #checkbox1.observe(self.update_channel, 'value')
        widgets.interactive_output(self.update_channel, self.checkbox1_arg_dict)
        dropbox2.observe(self.update_norm, 'value')

        self.chose_channel = [True for i in range(self.n_channels)]
        self.chose_norm = dropbox2.value
        self.x_label = text_xlabel.value
        self.y_label = text_ylabel.value
        self.title_label = text_title.value
        self.chose_y = y_axis_values.value

        #controls = widgets.VBox([y_axis_values, int_range_slider1, int_range_slider2, checkbox1, dropbox2, text_xlabel, text_ylabel, text_title])
        controls1 = widgets.VBox([checkbox1, dropbox2, y_axis_values], layout={'width': '300px'})
        controls2 = widgets.VBox([int_range_slider1, int_range_slider2, text_xlabel, text_ylabel, text_title],
                                 layout={'width': '300px'})

        tab = widgets.Tab([controls1, controls2])
        tab.set_title(0, 'Plotting values')
        tab.set_title(1, 'Plotting axes')

        #controls.layout = make_box_layout()
        #out_box = widgets.Box([output])
        #output.layout = make_box_layout()

        self.children = [tab, output]

        self.plot_lines()

    def update1(self, change):
        """redraw line (update plot)"""
        self.ax.set_ylim([change.new[0], change.new[1]])
        #fig.canvas.draw()
        self.fig.canvas.draw_idle()

    def update2(self, change):
        """redraw line (update plot)"""
        self.ax.set_xlim([change.new[0], change.new[1]])
        #fig.canvas.draw()
        self.fig.canvas.draw_idle()

    def update_channel(self, **kwargs):
        #self.chose_channel = change.new
        self.chose_channel = []
        for key in kwargs:
            self.chose_channel.append(kwargs[key])

        self.plot_lines()


    def update_norm(self, change):
        self.chose_norm = change.new
        '''
        if change.new == 'Yes':
            self.int_range_slider1.max = 1
        else:
            self.int_range_slider1.max = np.max(self.max_val)
        '''

        self.plot_lines()

    def update_y(self, change):
        self.chose_y = change.new

        self.plot_lines()


    def plot_lines(self):
        self.ax.clear()

        inds_channel = np.where(np.array(self.chose_channel))[0]

        # Use the measure the user chose
        y_value = self.chose_y

        self.y = []
        #for i in inds_channel:
        for i in range(self.n_channels):
            self.y.append(self.spots.data_df_sorted[f'{y_value} ch{i+1}'])

        if self.chose_norm == 'Yes':

            self.max_val = np.zeros(self.n_channels)
            # Calculate the new max intensity value (depending on the intensity measure chosen by user)
            for c in inds_channel:
                for idx in np.unique(self.spots.data_df_sorted['Track ID']):

                    i = self.spots.data_df_sorted['Track ID'] == idx

                    if np.max(self.y[c][i].to_numpy()) > self.max_val[c]:
                        self.max_val[c] = np.max(self.y[c][i].to_numpy())

            # Plot the data
            for c in inds_channel:
                for i,val in enumerate(np.unique(self.spots.data_df_sorted['Track ID'])):

                    idx = self.spots.data_df_sorted['Track ID']==val
                    self.line, = self.ax.plot(self.x[idx].to_numpy()*self.tr_min, self.y[c][idx].to_numpy()/self.max_val[c], color=self.colors[c][i])

            self.fig.canvas.draw_idle()

        else:
            for c in inds_channel:
                for i,val in enumerate(np.unique(self.spots.data_df_sorted['Track ID'])):

                    idx = self.spots.data_df_sorted['Track ID']==val
                    self.line, = self.ax.plot(self.x[idx].to_numpy()*self.tr_min, self.y[c][idx].to_numpy(), color=self.colors[c][i])

            self.fig.canvas.draw_idle()

        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_title(self.title_label)

    def update_xlabel(self, change):
        self.x_label = change.new
        self.ax.set_xlabel(change.new, fontsize=20)

    def update_ylabel(self, change):
        self.y_label = change.new
        self.ax.set_ylabel(change.new, fontsize=20)

    def update_title(self, change):
        self.title_label = change.new
        self.ax.set_title(change.new, fontsize=25)


# Interactive Traces with individual cells but choosing the channel or channels to plot
class IndividualTracksNeighbours(widgets.HBox):
    #def __init__(self, x, y, division, ids, tr_min, N, n_channels, cell_neighbors):
    def __init__(self, path_xml, path_csv, tr_min, tracks):
        super().__init__()

        # Contains all the spots information
        self.spots_all= csv_reader(path_csv, path_xml)

        # From csv_features and ordering tracks - specific for cell division
        self.division = tracks.spots_features['DivID']
        self.ids = tracks.spots_features['ID']
        self.x_pos = tracks.spots_features['X']
        self.y_pos = tracks.spots_features['Y']
        self.z_pos = tracks.spots_features['Z']
        self.n = tracks.n_tracks_divs

        # Default cell distance is 10 um (or 10 whatever the units are)
        self.cell_dist = 10

        # Get image characteristics
        fts = xml_features(path_xml)

        # All possible intensity measures Mastodon can export
        intensity_labels = np.array(['Mean', 'Std', 'Min', 'Max', 'Median', 'Sum', 'Center'])  # For each channel

        # Description for all possible intensity labels
        description = np.array(['Mean', 'Standard deviation', 'Minimum intensity', 'Maximum intensity',
                                'Median intensity', 'Sum intensity', 'Center intensity'])

        # Which were the measures Mastodon exported?
        vals_to_check = np.array([i.split()[0] for i in list(self.spots_all.data_df_sorted.keys())])
        inds_vals = np.array([any(i == intensity_labels) for i in vals_to_check])
        intensity_labels_to_use = np.unique(vals_to_check[inds_vals])

        # Descriptions for measures to use
        descriptions_to_use = description[np.array([any(i == intensity_labels) for i in intensity_labels_to_use])]

        # Number of channels
        self.n_channels = fts.channels

        # Spot IDs from csv reader and csv features
        self.spot_ids = self.spots_all.data_df_sorted['Spot ID'].values
        self.ids = tracks.spots_features['ID']

        # Mean, Median, Sum, Min, Max (wit h or without Std)
        self.y = []
        # Initialize the first plot
        self.y_value = intensity_labels_to_use[0]
        for i in range(self.n_channels):
            self.y.append(self.spots_all.data_df_sorted[f'{self.y_value} ch{i + 1}'])

        # Number of tracks in total
        self.N = len(np.unique(self.spots_all.data_df_sorted['Track ID'].to_numpy()))

        # The x-axis will always be time
        self.t = self.spots_all.data_df_sorted['Frames']
        self.tr_min = tr_min

        # Initial neighborhood to plot
        self.cell_neighbors, self.dist_matrix = self.cell_neighbors_func(self.n, self.division, self.x_pos, self.y_pos, self.z_pos, self.cell_dist)

        self.channels = [f'Channel {i+1}' for i in range(self.n_channels)]

        output = widgets.Output()


        with output:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=(6, 4))

        # To setup min and max value for the intensities
        min_val = np.inf
        self.max_val = np.zeros(self.n_channels)
        time_val = 0

        i = 0
        for c in range(self.n_channels):
            for idx in np.unique(self.spots_all.data_df_sorted['Track ID']):

                i = self.spots_all.data_df_sorted['Track ID']==idx

                if np.min(self.y[c][i].to_numpy()) < min_val:
                    min_val = np.min(self.y[c][i].to_numpy())

                if np.max(self.y[c][i].to_numpy())>self.max_val[c]:
                    self.max_val[c] = np.max(self.y[c][i].to_numpy())

                if np.max(self.t[i].to_numpy()*self.tr_min) > time_val:
                    time_val=np.max(self.t[i].to_numpy()*self.tr_min)


        # move the toolbar to the bottom
        self.fig.canvas.toolbar_position = 'bottom'
        self.ax.grid(True)

        # create some control elements
        y_axis_values = widgets.ToggleButtons(options=intensity_labels_to_use.tolist(), description='Measures:',
                                             disabled=False, button_style='', tooltips=descriptions_to_use.tolist())
        self.int_slider = widgets.IntSlider(value=0, min=0, max=len(self.cell_neighbors)-1, step=1, description='Cell #')
        float_text = widgets.FloatText(value=10, description=f'Cell distance ({fts.units})', min=0, max=np.max(self.dist_matrix))
        text_xlabel = widgets.Text(value='', description='X-Label', continuous_update=False)
        text_ylabel = widgets.Text(value='', description='Y-Label', continuous_update=False)
        text_title = widgets.Text(value='', description='Title', continuous_update=False)
        int_range_slider1 = widgets.IntRangeSlider(value=(min_val, np.max(self.max_val)), min=0, max=np.max(self.max_val)*2, step=10, description='Y-Limit')
        int_range_slider2 = widgets.IntRangeSlider(value=(0, time_val), min=0, max=time_val*2, step=10, description='X-Limit')

        dropbox2 = widgets.Dropdown(options=['Yes', 'No'], description='Normalize Data',
                    value='No',rows=2, interactive=True)

        # Checkbox for the channels
        checkbox1_kwargs = {'value':True, 'disabled':False, 'indent':False}
        self.checkbox1_arg_dict = {channel: widgets.Checkbox(description=channel, **checkbox1_kwargs) for channel in self.channels}
        checkbox1 = widgets.VBox(children=[self.checkbox1_arg_dict[channel] for channel in self.channels])

        # Several color pickers according to number of channels
        #self.checkbox2_arg_dict = {channel: widgets.ColorPicker(value=initial_color[i], description='Color for channel %d'%i) for i,channel in enumerate(self.channels)}
        #checkbox2 = widgets.VBox(children=[self.checkbox2_arg_dict[channel] for channel in self.channels])

        # Fix the X axis and the Y axis
        checkbox_xaxis = widgets.Checkbox(description='Fix X-Lim', value=False, disabled=False, indent=False)
        checkbox_yaxis = widgets.Checkbox(description='Fix Y-Lim', value=False, disabled=False, indent=False)

        # Make the widgets look good! :P
        #int_slider.style.handle_color = 'black'

        # connect callbacks and traits
        self.int_slider.observe(self.update, 'value')
        #self.int_slider.observe(self.plot_lines, 'value')
        float_text.observe(self.update_cell_dist, 'value')
        int_range_slider1.observe(self.update1, 'value')
        int_range_slider2.observe(self.update2, 'value')
        #color_picker.observe(self.line_color, 'value')
        text_xlabel.observe(self.update_xlabel, 'value')
        text_ylabel.observe(self.update_ylabel, 'value')
        text_title.observe(self.update_title, 'value')
        y_axis_values.observe(self.update_y, 'value')

        self.x_label = text_xlabel.value
        self.y_label = text_ylabel.value
        self.title_label = text_title.value

        checkbox_xaxis.observe(self.update_xaxis, 'value')
        checkbox_yaxis.observe(self.update_yaxis, 'value')

        dropbox2.observe(self.update_norm, 'value')

        ### CHOSE INITIAL VALUES TO PLOT
        # Initially, all channels are shown
        self.chose_channel = [True for i in range(self.n_channels)]

        # Intially, the axis are not fixed
        self.chose_fix_x = False
        self.chose_fix_y = False

        # First cell to plot is cell 0
        self.chose_cell = 0
        # Initial y value
        self.chose_y = y_axis_values.value

        # The initial colors to show are some pre-selected by me ;)
        #self.chose_color = [initial_color[i] for i in range(self.n_channels)]
        # Default value for normalization defined when creating Dropbox widget
        self.chose_norm = dropbox2.value

        self.x_label = text_xlabel.value
        self.y_label = text_ylabel.value
        self.title_label = text_title.value

        out1 = widgets.interactive_output(self.update_channel, self.checkbox1_arg_dict)

        # Initial y and x lim value
        self.x_lim = [0, time_val]
        self.y_lim = [min_val, np.max(self.max_val)]

        controls1 = widgets.VBox([self.int_slider, float_text, checkbox1, dropbox2, y_axis_values], layout={'width': '300px'})
        controls2 = widgets.VBox([int_range_slider1, int_range_slider2, checkbox_xaxis, checkbox_yaxis, text_xlabel, text_ylabel, text_title], layout={'width': '300px'})
        #controls.layout = make_box_layout()
        #out_box = widgets.Box([output])
        #output.layout = make_box_layout()

        tab = widgets.Tab([controls1, controls2])
        tab.set_title(0, 'Plotting values')
        tab.set_title(1, 'Plotting axes')

        self.children = [tab, output]

        self.plot_lines()

        # callback functions
    def update(self, change):
        self.chose_cell = change.new
        self.plot_lines()

    def update_y(self, change):
        self.chose_y = change.new

        self.plot_lines()

    def plot_lines(self):

        self.ax.clear()

        # Selected channels by the user
        inds_channel = np.where(np.array(self.chose_channel))[0]

        # Selected cell by the user
        cell = self.chose_cell

        # Use the measure the user chose
        y_value = self.chose_y

        self.y = []
        #for i in inds_channel:
        for i in range(self.n_channels):
            self.y.append(self.spots_all.data_df_sorted[f'{y_value} ch{i+1}'])

        # Number of cells in a neighborhood
        neighbors = self.cell_neighbors[cell]

        colors = [['black', 'dimgray', 'gray', 'darkgray', 'silver', 'lightgray']*self.N,
                  ['black', 'dimgray', 'gray', 'darkgray', 'silver', 'lightgray'] * self.N]

        colors_div = [['red', 'orange', 'blue', 'green', 'magenta', 'cyan', 'yellow', 'limegreen']*self.N,
                      ['red', 'orange', 'blue', 'green', 'magenta', 'cyan', 'yellow', 'limegreen'] * self.N]


        if self.chose_norm == 'Yes':

            # Find the maximum value to normalize
            self.max_val = np.zeros(self.n_channels)
            # Calculate the new max intensity value (depending on the intensity measure chosen by user)
            for c in inds_channel:
                for idx in np.unique(self.spots_all.data_df_sorted['Track ID']):

                    i = self.spots_all.data_df_sorted['Track ID'] == idx

                    if np.max(self.y[c][i].to_numpy()) > self.max_val[c]:
                        self.max_val[c] = np.max(self.y[c][i].to_numpy())

            for c in inds_channel:
                count1 = 0
                count2 = 0
                div_aux = []

                for j in neighbors:
                    vals = np.array([int(np.where(self.spot_ids == i)[0]) for i in self.ids[j]])
                    # If this cell does not divide, plot in grey
                    if self.division[j]==0:
                        self.line, = self.ax.plot(self.t.values[vals]*self.tr_min, self.y[c].values[vals]/self.max_val[c], color=colors[c][count1], linewidth=3)
                        count1 += 1

                    if self.division[j]>0:
                        # Keep track of the ID of dividing cells to not repeat them
                        if np.size(div_aux)==0:
                            # First find the other sibling
                            ind1 = self.division[j]
                            ind2 = np.where(np.array(self.division)==ind1)[0]
                            div_aux += list(ind2)
                        else:
                            # First find the other sibling
                            ind1 = self.division[j]
                            ind2 = np.where(np.array(self.division)==ind1)[0]

                            aux = set(div_aux).intersection(ind2)
                            if np.size(list(aux))>0:
                                div_aux += list(ind2)
                                continue
                            else:
                                div_aux += list(ind2)

                        vals1 = np.array([int(np.where(self.spot_ids == i)[0]) for i in self.ids[ind2[0]]])
                        vals2 = np.array([int(np.where(self.spot_ids == i)[0]) for i in self.ids[ind2[1]]])

                        self.line, = self.ax.plot(self.t.values[vals1]*self.tr_min, self.y[c].values[vals1]/self.max_val[c], color=colors_div[c][count2], linewidth=3)
                        self.line, = self.ax.plot(self.t.values[vals2]*self.tr_min, self.y[c].values[vals2]/self.max_val[c], color=colors_div[c][count2+1], linewidth=3)
                        count2 += 2

                        # Find timepoint where cells divide
                        list1 = self.ids[ind2[0]]
                        list2 = self.ids[ind2[1]]

                        ind = self.Diff(list1, list2)

                        self.line = self.ax.axvline(self.t.values[vals1][ind], color='black', linewidth=2, linestyle='--')


            self.fig.canvas.draw_idle()

        else:
            for c in inds_channel:
                count1 = 0
                count2 = 0
                div_aux = []
                for j in neighbors:
                    # If this cell does not divide, plot in grey
                    if self.division[j]==0:
                        vals = np.array([int(np.where(self.spot_ids == i)[0]) for i in self.ids[j]])
                        self.line, = self.ax.plot(self.t.values[vals]*self.tr_min, self.y[c].values[vals], color=colors[c][count1], linewidth=3)
                        count1 += 1

                    if self.division[j]>0:
                        # Keep track of the ID of dividing cells to not repeat them
                        if np.size(div_aux)==0:
                            # First find the other sibling
                            ind1 = self.division[j]
                            ind2 = np.where(np.array(self.division)==ind1)[0]
                            div_aux += list(ind2)
                        else:
                            # First find the other sibling
                            ind1 = self.division[j]
                            ind2 = np.where(np.array(self.division)==ind1)[0]

                            aux = set(div_aux).intersection(ind2)
                            if np.size(list(aux))>0:
                                div_aux += list(ind2)
                                continue
                            else:
                                div_aux += list(ind2)

                        vals1 = np.array([int(np.where(self.spot_ids == i)[0]) for i in self.ids[ind2[0]]])
                        vals2 = np.array([int(np.where(self.spot_ids == i)[0]) for i in self.ids[ind2[1]]])

                        self.line, = self.ax.plot(self.t.values[vals1]*self.tr_min, self.y[c].values[vals1], color=colors_div[c][count2], linewidth=3)
                        self.line, = self.ax.plot(self.t.values[vals2]*self.tr_min, self.y[c].values[vals2], color=colors_div[c][count2+1], linewidth=3)
                        count2 += 2

                        # Find timepoint where cells divide
                        list1 = self.ids[ind2[0]]
                        list2 = self.ids[ind2[1]]

                        ind = self.Diff(list1, list2)

                        self.line = self.ax.axvline(self.t.values[vals1][ind], color='black', linewidth=2, linestyle='--')


            self.fig.canvas.draw_idle()

        # Fix y and x axis according to whether the fix Checkbox was checked
        if self.chose_fix_x:
            self.ax.set_xlim([self.x_lim[0], self.x_lim[1]])
        if self.chose_fix_y:
            self.ax.set_ylim([self.y_lim[0], self.y_lim[1]])

        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_title(self.title_label)



    def update1(self, change):
        """redraw line (update plot)"""
        self.ax.set_ylim([change.new[0], change.new[1]])
        self.y_lim = change.new
        #fig.canvas.draw()
        self.fig.canvas.draw_idle()

    def update2(self, change):
        """redraw line (update plot)"""
        self.ax.set_xlim([change.new[0], change.new[1]])
        self.x_lim = change.new
        #fig.canvas.draw()
        self.fig.canvas.draw_idle()

    def update_xaxis(self, change):
        self.chose_fix_x = change.new

    def update_yaxis(self, change):
        self.chose_fix_y = change.new

    def update_channel(self, **kwargs):
        self.chose_channel = []
        for key in kwargs:
            self.chose_channel.append(kwargs[key])
        self.plot_lines()

    def update_norm(self, change):
        self.chose_norm = change.new
        self.plot_lines()

    #def update_color(self, **kwargs):
    #	"""set line color"""
    #	self.chose_color = []
    #	for key in kwargs:
    #		self.chose_color.append(kwargs[key])
#
    #	self.plot_lines()

    def update_cell_dist(self, change):
        self.cell_dist = change.new
        # Initial neighborhood to plot
        self.cell_neighbors, self.max_dist = self.cell_neighbors_func(self.n, self.division, self.x_pos, self.y_pos, self.z_pos, self.cell_dist)
        self.int_slider.max = len(self.cell_neighbors)-1

        self.plot_lines()

    def update_xlabel(self, change):
        self.x_label = change.new
        self.ax.set_xlabel(change.new, fontsize=20)

    def update_ylabel(self, change):
        self.y_label = change.new
        self.ax.set_ylabel(change.new, fontsize=20)

    def update_title(self, change):
        self.title_label = change.new
        self.ax.set_title(change.new, fontsize=25)

    # Find the timepoint where the cells divide
    def Diff(self, list1, list2):
        return np.where(np.array(list1)==[item for item in list1 if item not in list2][0])[0]

    def cell_neighbors_func(self, N, division, x, y, z, cell_dist):
        # Save the neighbors' IDs
        cell_neighbors = []
        coords_matrix = np.zeros((3, N))

        for j in range(N):

            # Position of cell 1
            x1 = x[j][0]
            y1 = y[j][0]
            z1 = z[j][0]

            # Fill in the coordinates matrix for later creating a distance matrix
            coords_matrix[0, j] = x1
            coords_matrix[1, j] = y1
            coords_matrix[2, j] = z1

            aux = []

            for i in range(j + 1, N):
                # Position of cell 2
                x2 = x[i][0]
                y2 = y[i][0]
                z2 = z[i][0]

                aux_dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

                if aux_dist <= cell_dist:

                    # If both cells divide
                    if division[i] > 0 and division[j]:
                        # Check if these cells are siblings? Only save if they are not
                        if division[i] != division[j]:
                            aux.append([i, j])

                    # If only one divides or non of them divide, always safe
                    else:
                        aux.append([i, j])

            # Save only if there is something to save
            if np.size(aux) > 0:
                cell_neighbors.append(np.unique(aux))

        dist_matrix = squareform(pdist(coords_matrix.T))

        return cell_neighbors, dist_matrix

# Interactive Traces with individual cells but choosing the channel or channels to plot
class interactive_traces_channels_division(widgets.HBox):
    #def __init__(self, x, y, division, ids, tr_min, N, n_channels, cell_neighbors):
    def __init__(self, path_xml, path_csv, tr_min, tracks):
        super().__init__()

        # Contains all the spots information
        self.spots_all= csv_reader(path_csv, path_xml)

        # From csv_features and ordering tracks - specific for cell division
        self.division = tracks.spots_features['DivID']
        self.ids = tracks.spots_features['ID']
        self.n = tracks.n_tracks_divs
        self.n_divs_total = tracks.n_division_tracks//2
        # Get image characteristics
        fts = xml_features(path_xml)

        # All possible intensity measures Mastodon can export
        intensity_labels = np.array(['Mean', 'Std', 'Min', 'Max', 'Median', 'Sum', 'Center'])  # For each channel

        # Description for all possible intensity labels
        description = np.array(['Mean', 'Standard deviation', 'Minimum intensity', 'Maximum intensity',
                                'Median intensity', 'Sum intensity', 'Center intensity'])

        # Which were the measures Mastodon exported?
        vals_to_check = np.array([i.split()[0] for i in list(self.spots_all.data_df_sorted.keys())])
        inds_vals = np.array([any(i == intensity_labels) for i in vals_to_check])
        intensity_labels_to_use = np.unique(vals_to_check[inds_vals])

        # Descriptions for measures to use
        descriptions_to_use = description[np.array([any(i == intensity_labels) for i in intensity_labels_to_use])]

        # Number of channels
        self.n_channels = fts.channels

        # Spot IDs from csv reader and csv features
        self.spot_ids = self.spots_all.data_df_sorted['Spot ID'].values
        self.ids = tracks.spots_features['ID']

        # Mean, Median, Sum, Min, Max (wit h or without Std)
        self.y = []
        # Initialize the first plot
        self.y_value = intensity_labels_to_use[0]
        for i in range(self.n_channels):
            self.y.append(self.spots_all.data_df_sorted[f'{self.y_value} ch{i + 1}'])

        # Number of tracks in total
        self.N = len(np.unique(self.spots_all.data_df_sorted['Track ID'].to_numpy()))

        # The x-axis will always be time
        self.t = self.spots_all.data_df_sorted['Frames']
        self.tr_min = tr_min

        self.channels = [f'Channel {i+1}' for i in range(self.n_channels)]

        output = widgets.Output()

        with output:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=(6, 4))

        # To setup min and max value for the intensities
        min_val = np.inf
        self.max_val = np.zeros(self.n_channels)
        time_val = 0

        for c in range(self.n_channels):
            for idx in np.unique(self.spots_all.data_df_sorted['Track ID']):

                i = self.spots_all.data_df_sorted['Track ID']==idx

                if np.min(self.y[c][i].to_numpy()) < min_val:
                    min_val = np.min(self.y[c][i].to_numpy())

                if np.max(self.y[c][i].to_numpy())>self.max_val[c]:
                    self.max_val[c] = np.max(self.y[c][i].to_numpy())

                if np.max(self.t[i].to_numpy()*self.tr_min) > time_val:
                    time_val=np.max(self.t[i].to_numpy()*self.tr_min)


        # move the toolbar to the bottom
        self.fig.canvas.toolbar_position = 'bottom'
        self.ax.grid(True)

        # create some control elements
        y_axis_values = widgets.ToggleButtons(options=intensity_labels_to_use.tolist(), description='Measures:',
                                             disabled=False, button_style='', tooltips=descriptions_to_use.tolist())
        int_slider = widgets.IntSlider(value=0, min=0, max=self.n_divs_total-1, step=1, description='Cell #')
        text_xlabel = widgets.Text(value='', description='X-Label', continuous_update=False)
        text_ylabel = widgets.Text(value='', description='Y-Label', continuous_update=False)
        text_title = widgets.Text(value='', description='Title', continuous_update=False)
        int_range_slider1 = widgets.IntRangeSlider(value=(min_val, np.max(self.max_val)), min=0, max=np.max(self.max_val)*2, step=10, description='Y-Limit')
        int_range_slider2 = widgets.IntRangeSlider(value=(0, time_val), min=0, max=time_val*2, step=10, description='X-Limit')

        dropbox2 = widgets.Dropdown(options=['Yes', 'No'], description='Normalize Data',
                    value='No',rows=2, interactive=True)

        # Checkbox for the channels
        checkbox1_kwargs = {'value':True, 'disabled':False, 'indent':False}
        self.checkbox1_arg_dict = {channel: widgets.Checkbox(description=channel, **checkbox1_kwargs) for channel in self.channels}
        checkbox1 = widgets.VBox(children=[self.checkbox1_arg_dict[channel] for channel in self.channels])

        # Several color pickers according to number of channels
        #self.checkbox2_arg_dict = {channel: widgets.ColorPicker(value=initial_color[i], description='Color for channel %d'%i) for i,channel in enumerate(self.channels)}
        #checkbox2 = widgets.VBox(children=[self.checkbox2_arg_dict[channel] for channel in self.channels])

        # Fix the X axis and the Y axis
        checkbox_xaxis = widgets.Checkbox(description='Fix X-Lim', value=False, disabled=False, indent=False)
        checkbox_yaxis = widgets.Checkbox(description='Fix Y-Lim', value=False, disabled=False, indent=False)

        # Make the widgets look good! :P
        #int_slider.style.handle_color = 'black'

        # connect callbacks and traits
        int_slider.observe(self.update, 'value')
        int_range_slider1.observe(self.update1, 'value')
        int_range_slider2.observe(self.update2, 'value')
        #color_picker.observe(self.line_color, 'value')
        text_xlabel.observe(self.update_xlabel, 'value')
        text_ylabel.observe(self.update_ylabel, 'value')
        text_title.observe(self.update_title, 'value')
        y_axis_values.observe(self.update_y, 'value')

        self.x_label = text_xlabel.value
        self.y_label = text_ylabel.value
        self.title_label = text_title.value

        checkbox_xaxis.observe(self.update_xaxis, 'value')
        checkbox_yaxis.observe(self.update_yaxis, 'value')

        dropbox2.observe(self.update_norm, 'value')

        ### CHOSE INITIAL VALUES TO PLOT
        # Initially, all channels are shown
        self.chose_channel = [True for i in range(self.n_channels)]

        # Intially, the axis are not fixed
        self.chose_fix_x = False
        self.chose_fix_y = False

        # First cell to plot is cell 0
        self.chose_cell = 0
        # Initial y value
        self.chose_y = y_axis_values.value

        # The initial colors to show are some pre-selected by me ;)
        #self.chose_color = [initial_color[i] for i in range(self.n_channels)]
        # Default value for normalization defined when creating Dropbox widget
        self.chose_norm = dropbox2.value

        self.x_label = text_xlabel.value
        self.y_label = text_ylabel.value
        self.title_label = text_title.value

        out1 = widgets.interactive_output(self.update_channel, self.checkbox1_arg_dict)

        # Initial y and x lim value
        self.x_lim = [0, time_val]
        self.y_lim = [min_val, np.max(self.max_val)]

        controls1 = widgets.VBox([int_slider, checkbox1, dropbox2, y_axis_values], layout={'width': '300px'})
        controls2 = widgets.VBox([int_range_slider1, int_range_slider2, checkbox_xaxis, checkbox_yaxis, text_xlabel, text_ylabel, text_title], layout={'width': '300px'})
        #controls.layout = make_box_layout()
        #out_box = widgets.Box([output])
        #output.layout = make_box_layout()

        tab = widgets.Tab([controls1, controls2])
        tab.set_title(0, 'Plotting values')
        tab.set_title(1, 'Plotting axes')

        self.children = [tab, output]

        self.plot_lines()

        # callback functions
    def update(self, change):
        self.chose_cell = change.new
        self.plot_lines()

    def update_y(self, change):
        self.chose_y = change.new
        self.plot_lines()

    def plot_lines(self):

        self.ax.clear()

        # Selected channels by the user
        inds_channel = np.where(np.array(self.chose_channel))[0]

        # Selected cell by the user
        cell = self.chose_cell

        # Find siblings IDs
        div_cells = np.where(np.array(self.division)==(cell+1))[0]

        # Use the measure the user chose
        y_value = self.chose_y

        self.y = []
        #for i in inds_channel:
        for i in range(self.n_channels):
            self.y.append(self.spots_all.data_df_sorted[f'{y_value} ch{i+1}'])

        colors_div = [['limegreen', 'blue'], ['magenta', 'orange']]

        if self.chose_norm == 'Yes':

            # Find the maximum value to normalize
            self.max_val = np.zeros(self.n_channels)
            # Calculate the new max intensity value (depending on the intensity measure chosen by user)
            for c in inds_channel:
                for idx in np.unique(self.spots_all.data_df_sorted['Track ID']):

                    i = self.spots_all.data_df_sorted['Track ID'] == idx

                    if np.max(self.y[c][i].to_numpy()) > self.max_val[c]:
                        self.max_val[c] = np.max(self.y[c][i].to_numpy())

            for c in inds_channel:

                # Keep track of the ID of dividing cells to not repeat them
                vals1 = np.array([int(np.where(self.spot_ids == i)[0]) for i in self.ids[div_cells[0]]])
                vals2 = np.array([int(np.where(self.spot_ids == i)[0]) for i in self.ids[div_cells[1]]])

                self.line, = self.ax.plot(self.t.values[vals1]*self.tr_min, self.y[c].values[vals1]/self.max_val[c], color=colors_div[c][0], linewidth=3)
                self.line, = self.ax.plot(self.t.values[vals2]*self.tr_min, self.y[c].values[vals2]/self.max_val[c], color=colors_div[c][1], linewidth=3)

                # Find timepoint where cells divide
                list1 = self.ids[div_cells[0]]
                list2 = self.ids[div_cells[1]]

                ind = self.Diff(list1, list2)

                self.line = self.ax.axvline(self.t.values[vals1][ind], color='black', linewidth=2, linestyle='--')


            self.fig.canvas.draw_idle()

        else:
            for c in inds_channel:

                vals1 = np.array([int(np.where(self.spot_ids == i)[0]) for i in self.ids[div_cells[0]]])
                vals2 = np.array([int(np.where(self.spot_ids == i)[0]) for i in self.ids[div_cells[1]]])

                self.line, = self.ax.plot(self.t.values[vals1]*self.tr_min, self.y[c].values[vals1], color=colors_div[c][0], linewidth=3)
                self.line, = self.ax.plot(self.t.values[vals2]*self.tr_min, self.y[c].values[vals2], color=colors_div[c][1], linewidth=3)

                # Find timepoint where cells divide
                list1 = self.ids[div_cells[0]]
                list2 = self.ids[div_cells[1]]

                ind = self.Diff(list1, list2)

                self.line = self.ax.axvline(self.t.values[vals1][ind], color='black', linewidth=2, linestyle='--')


            self.fig.canvas.draw_idle()

        # Fix y and x axis according to whether the fix Checkbox was checked
        if self.chose_fix_x:
            self.ax.set_xlim([self.x_lim[0], self.x_lim[1]])
        if self.chose_fix_y:
            self.ax.set_ylim([self.y_lim[0], self.y_lim[1]])

        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_title(self.title_label)

    def update1(self, change):
        """redraw line (update plot)"""
        self.ax.set_ylim([change.new[0], change.new[1]])
        self.y_lim = change.new
        #fig.canvas.draw()
        self.fig.canvas.draw_idle()

    def update2(self, change):
        """redraw line (update plot)"""
        self.ax.set_xlim([change.new[0], change.new[1]])
        self.x_lim = change.new
        #fig.canvas.draw()
        self.fig.canvas.draw_idle()

    def update_xaxis(self, change):
        self.chose_fix_x = change.new

    def update_yaxis(self, change):
        self.chose_fix_y = change.new

    def update_channel(self, **kwargs):
        self.chose_channel = []
        for key in kwargs:
            self.chose_channel.append(kwargs[key])
        self.plot_lines()

    def update_norm(self, change):
        self.chose_norm = change.new
        self.plot_lines()

    #def update_color(self, **kwargs):
    #	"""set line color"""
    #	self.chose_color = []
    #	for key in kwargs:
    #		self.chose_color.append(kwargs[key])
#
    #	self.plot_lines()


    def update_xlabel(self, change):
        self.x_label = change.new
        self.ax.set_xlabel(change.new, fontsize=20)

    def update_ylabel(self, change):
        self.y_label = change.new
        self.ax.set_ylabel(change.new, fontsize=20)

    def update_title(self, change):
        self.title_label = change.new
        self.ax.set_title(change.new, fontsize=25)

    # Find the timepoint where the cells divide
    def Diff(self, list1, list2):
        return np.where(np.array(list1)==[item for item in list1 if item not in list2][0])[0]


# Check how far apart division cells move in 3D
class SpatialDivisionAnalaysis(widgets.HBox):
    #def __init__(self, x, y, division, ids, tr_min, N, n_channels, cell_neighbors):
    def __init__(self, path_xml, path_csv, tr_min, tracks):
        super().__init__()

        # Contains all the spots information
        self.spots_all= csv_reader(path_csv, path_xml)

        # From csv_features and ordering tracks - specific for cell division
        self.division = tracks.spots_features['DivID']
        self.ids = tracks.spots_features['ID']
        self.n = tracks.n_tracks_divs
        self.n_divs_total = tracks.n_division_tracks//2
        # Get image characteristics
        fts = xml_features(path_xml)

        # All possible intensity measures Mastodon can export
        intensity_labels = np.array(['Mean', 'Std', 'Min', 'Max', 'Median', 'Sum', 'Center'])  # For each channel

        # Description for all possible intensity labels
        description = np.array(['Mean', 'Standard deviation', 'Minimum intensity', 'Maximum intensity',
                                'Median intensity', 'Sum intensity', 'Center intensity'])

        # Which were the measures Mastodon exported?
        vals_to_check = np.array([i.split()[0] for i in list(self.spots_all.data_df_sorted.keys())])
        inds_vals = np.array([any(i == intensity_labels) for i in vals_to_check])
        intensity_labels_to_use = np.unique(vals_to_check[inds_vals])

        # Descriptions for measures to use
        descriptions_to_use = description[np.array([any(i == intensity_labels) for i in intensity_labels_to_use])]

        # Number of channels
        self.n_channels = fts.channels

        # Spot IDs from csv reader and csv features
        self.spot_ids = self.spots_all.data_df_sorted['Spot ID'].values
        self.ids = tracks.spots_features['ID']

        # Mean, Median, Sum, Min, Max (wit h or without Std)
        self.y = []
        # Initialize the first plot
        self.y_value = intensity_labels_to_use[0]
        for i in range(self.n_channels):
            self.y.append(self.spots_all.data_df_sorted[f'{self.y_value} ch{i + 1}'])

        # Number of tracks in total
        self.N = len(np.unique(self.spots_all.data_df_sorted['Track ID'].to_numpy()))

        # The x-axis will always be time
        self.t = self.spots_all.data_df_sorted['Frames']
        self.tr_min = tr_min

        self.channels = [f'Channel {i+1}' for i in range(self.n_channels)]

        output = widgets.Output()

        with output:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=(6, 4))
            self.ax2 = self.ax.twinx()
            self.ax2.spines['right'].set_visible(True)
            self.ax2.spines['right'].set_linewidth(4)
            self.ax.spines['left'].set_color('blue')
            self.ax2.spines['right'].set_color('grey')
            self.ax2.spines['left'].set_color('blue')
            self.ax2.tick_params(colors='grey', axis='y')
            self.ax.tick_params(colors='blue', axis='y')


        # To setup min and max value for the intensities
        min_val = np.inf
        self.max_val = np.zeros(self.n_channels)
        time_val = 0

        for c in range(self.n_channels):
            for idx in np.unique(self.spots_all.data_df_sorted['Track ID']):

                i = self.spots_all.data_df_sorted['Track ID']==idx

                if np.min(self.y[c][i].to_numpy()) < min_val:
                    min_val = np.min(self.y[c][i].to_numpy())

                if np.max(self.y[c][i].to_numpy())>self.max_val[c]:
                    self.max_val[c] = np.max(self.y[c][i].to_numpy())

                if np.max(self.t[i].to_numpy()*self.tr_min) > time_val:
                    time_val=np.max(self.t[i].to_numpy()*self.tr_min)

        # move the toolbar to the bottom
        self.fig.canvas.toolbar_position = 'bottom'
        self.ax.grid(True)

        # create some control elements
        y_axis_values = widgets.ToggleButtons(options=intensity_labels_to_use.tolist(), description='Measures:',
                                             disabled=False, button_style='', tooltips=descriptions_to_use.tolist())
        int_slider = widgets.IntSlider(value=0, min=0, max=self.n_divs_total-1, step=1, description='Cell #')
        text_xlabel = widgets.Text(value='', description='X-Label', continuous_update=False)
        text_ylabel = widgets.Text(value='', description='Y-Label-1', continuous_update=False)
        text_ylabel2 = widgets.Text(value='', description='Y-Label-2', continuous_update=False)
        text_title = widgets.Text(value='', description='Title', continuous_update=False)
        int_range_slider1 = widgets.IntRangeSlider(value=(min_val, np.max(self.max_val)), min=0,
                                                   max=np.max(self.max_val)*2, step=10, description='Y-Limit-1')
        int_range_slider3 = widgets.IntRangeSlider(value=(0, 30), min=0,
                                                   max=30, step=1, description='Y-Limit-2')
        int_range_slider2 = widgets.IntRangeSlider(value=(0, time_val), min=0, max=time_val*2, step=10, description='X-Limit')

        dropbox2 = widgets.Dropdown(options=['Yes', 'No'], description='Normalize Data',
                    value='No',rows=2, interactive=True)

        # Checkbox for the channels
        checkbox1_kwargs = {'value':True, 'disabled':False, 'indent':False}
        self.checkbox1_arg_dict = {channel: widgets.Checkbox(description=channel, **checkbox1_kwargs) for channel in self.channels}
        checkbox1 = widgets.VBox(children=[self.checkbox1_arg_dict[channel] for channel in self.channels])

        # Several color pickers according to number of channels
        #self.checkbox2_arg_dict = {channel: widgets.ColorPicker(value=initial_color[i], description='Color for channel %d'%i) for i,channel in enumerate(self.channels)}
        #checkbox2 = widgets.VBox(children=[self.checkbox2_arg_dict[channel] for channel in self.channels])

        # Fix the X axis and the Y axis
        checkbox_xaxis = widgets.Checkbox(description='Fix X-Lim', value=False, disabled=False, indent=False)
        checkbox_yaxis = widgets.Checkbox(description='Fix Y-Lim-1', value=False, disabled=False, indent=False)
        checkbox_yaxis2 = widgets.Checkbox(description='Fix Y-Lim-2', value=False, disabled=False, indent=False)

        # Make the widgets look good! :P
        #int_slider.style.handle_color = 'black'

        # connect callbacks and traits
        int_slider.observe(self.update, 'value')
        int_range_slider1.observe(self.update1, 'value')
        int_range_slider2.observe(self.update2, 'value')
        int_range_slider3.observe(self.update3, 'value')
        #color_picker.observe(self.line_color, 'value')
        text_xlabel.observe(self.update_xlabel, 'value')
        text_ylabel.observe(self.update_ylabel, 'value')
        text_ylabel2.observe(self.update_ylabel2, 'value')
        text_title.observe(self.update_title, 'value')
        y_axis_values.observe(self.update_y, 'value')

        self.x_label = text_xlabel.value
        self.y_label = text_ylabel.value
        self.y_label2 = text_ylabel2.value
        self.title_label = text_title.value

        checkbox_xaxis.observe(self.update_xaxis, 'value')
        checkbox_yaxis.observe(self.update_yaxis, 'value')
        checkbox_yaxis2.observe(self.update_yaxis2, 'value')

        dropbox2.observe(self.update_norm, 'value')

        ### CHOSE INITIAL VALUES TO PLOT
        # Initially, all channels are shown
        self.chose_channel = [True for i in range(self.n_channels)]

        # Intially, the axis are not fixed
        self.chose_fix_x = False
        self.chose_fix_y = False
        self.chose_fix_y2 = False

        # First cell to plot is cell 0
        self.chose_cell = 0
        # Initial y value
        self.chose_y = y_axis_values.value

        # The initial colors to show are some pre-selected by me ;)
        #self.chose_color = [initial_color[i] for i in range(self.n_channels)]
        # Default value for normalization defined when creating Dropbox widget
        self.chose_norm = dropbox2.value

        out1 = widgets.interactive_output(self.update_channel, self.checkbox1_arg_dict)

        # Initial y and x lim value
        self.x_lim = [0, time_val]
        self.y_lim = [min_val, np.max(self.max_val)]

        controls1 = widgets.VBox([int_slider, checkbox1, dropbox2, y_axis_values], layout={'width': '300px'})
        controls2 = widgets.VBox([int_range_slider1, int_range_slider3, int_range_slider2, checkbox_xaxis,
                                  checkbox_yaxis, checkbox_yaxis2, text_xlabel, text_ylabel, text_ylabel2, text_title],
                                 layout={'width': '300px'})
        #controls.layout = make_box_layout()
        #out_box = widgets.Box([output])
        #output.layout = make_box_layout()

        tab = widgets.Tab([controls1, controls2])
        tab.set_title(0, 'Plotting values')
        tab.set_title(1, 'Plotting axes')

        self.children = [tab, output]

        self.plot_lines()

        # callback functions
    def update(self, change):
        self.chose_cell = change.new
        self.plot_lines()

    def update_y(self, change):
        self.chose_y = change.new
        self.plot_lines()

    def plot_lines(self):

        self.ax.clear()
        self.ax2.clear()

        # Selected channels by the user
        inds_channel = np.where(np.array(self.chose_channel))[0]

        # Selected cell by the user
        cell = self.chose_cell

        # Find siblings IDs
        div_cells = np.where(np.array(self.division)==(cell+1))[0]

        # Use the measure the user chose
        y_value = self.chose_y

        self.y = []
        #for i in inds_channel:
        for i in range(self.n_channels):
            self.y.append(self.spots_all.data_df_sorted[f'{y_value} ch{i+1}'])

        colors_div = [['limegreen', 'blue'], ['magenta', 'orange']]

        if self.chose_norm == 'Yes':

            # Find the maximum value to normalize
            self.max_val = np.zeros(self.n_channels)
            # Calculate the new max intensity value (depending on the intensity measure chosen by user)
            for c in inds_channel:
                for idx in np.unique(self.spots_all.data_df_sorted['Track ID']):

                    i = self.spots_all.data_df_sorted['Track ID'] == idx

                    if np.max(self.y[c][i].to_numpy()) > self.max_val[c]:
                        self.max_val[c] = np.max(self.y[c][i].to_numpy())

            for c in inds_channel:

                # Keep track of the ID of dividing cells to not repeat them
                vals1 = np.array([int(np.where(self.spot_ids == i)[0]) for i in self.ids[div_cells[0]]])
                vals2 = np.array([int(np.where(self.spot_ids == i)[0]) for i in self.ids[div_cells[1]]])

                distance = self.distance_3d(vals1, vals2)

                self.line, = self.ax.plot(self.t.values[vals1]*self.tr_min, self.y[c].values[vals1]/self.max_val[c], color=colors_div[c][0], linewidth=3)
                self.line, = self.ax.plot(self.t.values[vals2]*self.tr_min, self.y[c].values[vals2]/self.max_val[c], color=colors_div[c][1], linewidth=3)

                # Find timepoint where cells divide
                list1 = self.ids[div_cells[0]]
                list2 = self.ids[div_cells[1]]

                ind = self.Diff(list1, list2)

                self.line = self.ax.axvline(self.t.values[vals1][ind], color='black', linewidth=2, linestyle='--')

                # Plot 3D distance
                self.line, = self.ax2.plot(np.arange(len(distance)) * self.tr_min, distance,
                                          color='grey', linewidth=3)

            self.fig.canvas.draw_idle()

        else:
            for c in inds_channel:

                vals1 = np.array([int(np.where(self.spot_ids == i)[0]) for i in self.ids[div_cells[0]]])
                vals2 = np.array([int(np.where(self.spot_ids == i)[0]) for i in self.ids[div_cells[1]]])

                self.line, = self.ax.plot(self.t.values[vals1]*self.tr_min, self.y[c].values[vals1], color=colors_div[c][0], linewidth=3)
                self.line, = self.ax.plot(self.t.values[vals2]*self.tr_min, self.y[c].values[vals2], color=colors_div[c][1], linewidth=3)

                # Find time-point where cells divide
                list1 = self.ids[div_cells[0]]
                list2 = self.ids[div_cells[1]]

                ind = self.Diff(list1, list2)

                self.line = self.ax.axvline(self.t.values[vals1][ind], color='black', linewidth=2, linestyle='--')

                # Plot 3D distance
                distance = self.distance_3d(vals1, vals2)
                self.line, = self.ax2.plot(np.arange(len(distance)) * self.tr_min, distance,
                                          color='grey', linewidth=3)


            self.fig.canvas.draw_idle()

        # Fix y and x axis according to whether the fix Checkbox was checked
        if self.chose_fix_x:
            self.ax.set_xlim([self.x_lim[0], self.x_lim[1]])
        if self.chose_fix_y:
            self.ax.set_ylim([self.y_lim[0], self.y_lim[1]])
        if self.chose_fix_y2:
            self.ax2.set_ylim([self.y_lim2[0], self.y_lim2[1]])

        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.yaxis.label.set_color('blue')
        self.ax2.set_ylabel(self.y_label2)
        self.ax2.yaxis.label.set_color('grey')
        self.ax.set_title(self.title_label)
        self.ax.spines['left'].set_color('blue')
        self.ax2.spines['right'].set_color('grey')

    def update1(self, change):
        """redraw line (update plot)"""
        self.ax.set_ylim([change.new[0], change.new[1]])
        self.y_lim = change.new
        #fig.canvas.draw()
        self.fig.canvas.draw_idle()

    def update2(self, change):
        """redraw line (update plot)"""
        self.ax.set_xlim([change.new[0], change.new[1]])
        self.x_lim = change.new
        #fig.canvas.draw()
        self.fig.canvas.draw_idle()

    def update3(self, change):
        """redraw line (update plot)"""
        self.ax2.set_ylim([change.new[0], change.new[1]])
        self.y_lim2 = change.new
        #fig.canvas.draw()
        self.fig.canvas.draw_idle()

    def update_xaxis(self, change):
        self.chose_fix_x = change.new

    def update_yaxis(self, change):
        self.chose_fix_y = change.new

    def update_yaxis2(self, change):
        self.chose_fix_y2 = change.new

    def update_channel(self, **kwargs):
        self.chose_channel = []
        for key in kwargs:
            self.chose_channel.append(kwargs[key])
        self.plot_lines()

    def update_norm(self, change):
        self.chose_norm = change.new
        self.plot_lines()

    #def update_color(self, **kwargs):
    #	"""set line color"""
    #	self.chose_color = []
    #	for key in kwargs:
    #		self.chose_color.append(kwargs[key])
#
    #	self.plot_lines()


    def update_xlabel(self, change):
        self.x_label = change.new
        self.ax.set_xlabel(change.new, fontsize=20)

    def update_ylabel(self, change):
        self.y_label = change.new
        self.ax.set_ylabel(change.new, fontsize=20)

    def update_ylabel2(self, change):
        self.y_label2 = change.new
        self.ax2.set_ylabel(change.new, fontsize=20)

    def update_title(self, change):
        self.title_label = change.new
        self.ax.set_title(change.new, fontsize=25)

    # Find the timepoint where the cells divide
    def Diff(self, list1, list2):
        return np.where(np.array(list1)==[item for item in list1 if item not in list2][0])[0]

    def distance_3d(self, vals1, vals2):
        # Find the timepoints in which the two cells appear - otherwise there is no distance to calculate
        t_intersect = set(list(self.t.values[vals1])).intersection(list(self.t.values[vals2]))

        # Now find the indices of this intersection
        inds1 = np.array(vals1)[np.array([list(self.t.values[vals1]).index(x) for x in t_intersect])]
        inds2 = np.array(vals2)[np.array([list(self.t.values[vals2]).index(x) for x in t_intersect])]

        # X, Y, Z coordinates
        x_coords = (self.spots_all.data_df_sorted['X'].values[inds1] -
                    self.spots_all.data_df_sorted['X'].values[inds2]) ** 2
        y_coords = (self.spots_all.data_df_sorted['Y'].values[inds1] -
                    self.spots_all.data_df_sorted['Y'].values[inds2]) ** 2
        z_coords = (self.spots_all.data_df_sorted['Z'].values[inds1] -
                    self.spots_all.data_df_sorted['Z'].values[inds2]) ** 2

        # Eucledian distance (3D)
        distance = np.sqrt(x_coords + y_coords + z_coords)
        distance[distance == 0] = np.nan  # To avoid seeing the 0 distance

        return distance


# Peak detection analysis
class PeakDetectionAnalysis(widgets.HBox):
    """This class is called to calculate the peaks of all the cells. This method combines an automatic detection with a
    manual curation system.
    :class:
    :path_csv: Path where the two csv files (-edges.csv and -data.csv) are located. These are the files which are
        obtained from Mastodon
    :path_xml: Path where the .xml file created from BigDataViewer is located. This file should be in the same folder
        as the .hdf5 data
    :tr_min: Temporal resolution with which the data was acquired
    """
    def __init__(self, path_csv, path_xml, tr_min):
        """Constructor method
        """
        super().__init__()

        # Load the data from the -csv which we organized in the MastodonFunctions.py
        self.spots = csv_reader(path_csv, path_xml)
        fts = xml_features(path_xml)

        # The X is always the frames
        self.x = self.spots.data_df_sorted['Frames']

        # All possible intensity measures Mastodon can export
        intensity_labels = np.array(['Mean', 'Std', 'Min', 'Max', 'Median', 'Sum', 'Center'])  # For each channel

        # Description for all possible intensity labels
        description = np.array(['Mean', 'Standard deviation', 'Minimum intensity', 'Maximum intensity',
                                'Median intensity', 'Sum intensity', 'Center intensity'])

        # Which were the measures Mastodon exported?
        vals_to_check = np.array([i.split()[0] for i in list(self.spots.data_df_sorted.keys())])
        inds_vals = np.array([any(i == intensity_labels) for i in vals_to_check])
        intensity_labels_to_use = np.unique(vals_to_check[inds_vals])

        # Descriptions for measures to use
        descriptions_to_use = description[np.array([any(i == intensity_labels) for i in intensity_labels_to_use])]

        # Number of channels
        self.n_channels = fts.channels

        # Mean, Median, Sum, Min, Max (with or sithout Std)
        self.y = []
        # Initialize the first plot
        self.y_value = intensity_labels_to_use[0]
        for i in range(self.n_channels):
            self.y.append(self.spots.data_df_sorted[f'{self.y_value} ch{i+1}'])

        self.tr_min = tr_min

        # Number of tracks in total
        self.N = len(np.unique(self.spots.data_df_sorted['Track ID'].to_numpy()))

        # Number of channels
        self.n_channels = fts.channels

        self.channels = [f'Channel {i+1}' for i in range(self.n_channels)]

        # Initial value for CPU percentage
        self.cpu_per = cpu_percent()

        # By default, interactivity is disabled
        self.chose_interactivity = False

        # By default, alpha is 1
        self.chose_alpha = 1

        # Peak detection parameters
        self.threshold_val = 0 # Threshold value for maxima
        self.prominence_val = 0# Prominence value for maxima
        self.width_val = 0 # Width value for maxima
        self.distance_val = 1 # Distance value for maxima
        self.window = 0 # Averaging window size

        output = widgets.Output()

        # default line color - magenta, lime, blue, orange, dark purple
        initial_color = ['#FF00DD', '#00FF00', '#0000FF', '#FF5733', '#581845'] * self.N

        with output:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=(6, 4))

        # To setup min and max value for the intensities
        min_val = np.inf
        self.max_val = np.zeros(self.n_channels)
        self.time_val = 0

        # Initial value for CPU percentage
        self.cpu_per = cpu_percent()

        # By default, interactivity is disabled
        self.chose_interactivity = False

        # By default, alpha is 1
        self.chose_alpha = 1

        # Define min and max values for X-axis (time) and Y-axis (intensity)
        # Also, define which cells IDs contain division to ignore them in the "exclude division" option
        self.n_links = np.zeros(len(np.unique(self.spots.data_df_sorted['Track ID'])))

        for c in range(self.n_channels):
            for idx_ind, idx in enumerate(np.unique(self.spots.data_df_sorted['Track ID'])):

                i = self.spots.data_df_sorted['Track ID'] == idx
                if c == 0:
                    self.n_links[idx_ind] = np.max(self.spots.data_df_sorted.loc[i, 'N Links'].to_numpy())

                # Min value for intensities (y-axis)
                if np.min(self.y[c][i].to_numpy()) < min_val:
                    min_val = np.min(self.y[c][i].to_numpy())
                # Max value for intensities (y-axis)
                if np.max(self.y[c][i].to_numpy()) > self.max_val[c]:
                    self.max_val[c] = np.max(self.y[c][i].to_numpy())
                # Max value for time (x-axis)
                if np.max(self.x[i].to_numpy()*self.tr_min) > self.time_val:
                    self.time_val = np.max(self.x[i].to_numpy()*self.tr_min)

        self.no_div_inds = self.n_links <= 2

        # move the toolbar to the bottom
        self.fig.canvas.toolbar_position = 'bottom'
        self.ax.grid(True)

        # Control Elements
        # -----------------
        # Intensity labels to plot - Mean, Median, Sum, Max, Min, Std
        style = {'description_width': 'initial'}
        y_axis_values = widgets.ToggleButtons(options=intensity_labels_to_use.tolist(), description='Measures:',
                                              disabled=False, button_style='', tooltips=descriptions_to_use.tolist(),
                                              layout=dict(width='50%'), style=style)

        # Cell track to plot
        self.int_slider = widgets.IntSlider(value=0, min=0, max=self.N-1, step=1, description='Cell #')

        # Labels for x and y axis
        text_xlabel = widgets.Text(value='', description='X-Label', continuous_update=False, layout=dict(width='90%'))
        text_ylabel = widgets.Text(value='', description='Y-Label', continuous_update=False, layout=dict(width='90%'))

        # Title for plot
        text_title = widgets.Text(value='', description='Title', continuous_update=False, layout=dict(width='90%'))

        # Min-max slider for y axis (intensity)
        self.int_range_slider1 = widgets.FloatRangeSlider(value=(min_val, np.max(self.max_val)), min=0,
                                                   max=np.max(self.max_val)*2, step=10, description='Y-Limit')

        # Min-max slider for x axis (time)
        int_range_slider2 = widgets.IntRangeSlider(value=(0, self.time_val), min=0, max=self.time_val*2, step=10,
                                                   description='X-Limit')

        # Normalize options: Yes, No
        dropbox2 = widgets.Dropdown(options=['Yes', 'No'], description='Normalize Data: ', value='No', rows=2,
                                    interactive=True, layout=dict(width='90%'), style=style)

        # Checkbox for the channels to plot
        checkbox1_kwargs = {'value': True, 'disabled': False, 'indent': False}
        self.checkbox1_arg_dict = {channel: widgets.Checkbox(description=channel, **checkbox1_kwargs) for channel in
                                   self.channels}
        checkbox1 = widgets.VBox(children=[self.checkbox1_arg_dict[channel] for channel in self.channels])

        # Checkbox to select whether you want to include or not the dividing cell tracks
        exclude_divs_box = widgets.Checkbox(value=True, description='Exclude divisions', disabled=False, indent=False)

        # Bar to display the percentage of CPU used in each operation
        self.cpu_percentage = widgets.FloatProgress(value=self.cpu_per, min=0, max=100, description='CPU %:',
                                                    bar_style='', style={'bar_color': 'blue'}, orientation='horizontal')
        display(self.cpu_percentage)

        # Interactivity toggle button - if clicked, the plot button appears
        interactivity_button = widgets.ToggleButton(value=False, description='Interactivity', disabled=False,
                                                    button_style='', tool_tip='Disable interactivity mode',
                                                    icon='check')
        self.plot_button = widgets.Button(description='Plot', disabled=False, button_style='',
                                          tooltip='Plot the tracks', icon='')

        # Several color pickers according to number of channels
        self.checkbox2_arg_dict = {channel: widgets.ColorPicker(value=initial_color[i],
                                                                description='Color for channel %d' % i, style=style,
                                                                layout=dict(width='90%'))
                                   for i, channel in enumerate(self.channels)}
        checkbox2 = widgets.VBox(children=[self.checkbox2_arg_dict[channel] for channel in self.channels])

        # Float slider to change the alpha of the lines
        style = {'description_width': 'initial'}  # This is so that the description fits in 1 line
        float_slider_alpha = widgets.FloatSlider(value=1.0, min=0, max=1.0, step=0.1, description='Transparency: ',
                                                 disabled=False, continuous_update=False, orientation='horizontal',
                                                 readout=True, readout_format='.1f', style=style)

        # Change line-width of lines
        float_slider_lw = widgets.FloatSlider(value=1.0, min=0, max=4.0, step=0.2, description='Linewidth: ',
                                              disabled=False, continuous_update=False, orientation='horizontal',
                                              readout=True, readout_format='.1f', style=style)

        # Fix the X axis and the Y axis
        checkbox_xaxis = widgets.Checkbox(description='Fix X-Lim', value=False, disabled=False, indent=False)
        checkbox_yaxis = widgets.Checkbox(description='Fix Y-Lim', value=False, disabled=False, indent=False)

        # Peak detection parameters chosen by user
        # ----------------------------------------
        self.window_slider = widgets.IntSlider(value=self.window, min=0, max=self.time_val/2, step=1,
                                               description='Average window size') # Default maxima is time-points/2
        self.threshold_slider = widgets.FloatSlider(value=self.threshold_val, min=0, max=10, step=0.1, description='Threshold')
        self.prominence_slider = widgets.FloatSlider(value=self.prominence_val, min=0, max=50, step=0.1, description='Prominence')
        self.width_slider = widgets.FloatSlider(value=self.width_val, min=0, max=20, step=0.1, description='Width')
        self.distance_slicer = widgets.FloatSlider(value=self.distance_val, min=1, max=20, step=0.1, description='Distance')

        # Connect callbacks and traits
        # -----------------------------
        self.int_slider.observe(self.update, 'value')
        self.int_range_slider1.observe(self.update1, 'value')
        int_range_slider2.observe(self.update2, 'value')
        text_xlabel.observe(self.update_xlabel, 'value')
        text_ylabel.observe(self.update_ylabel, 'value')
        text_title.observe(self.update_title, 'value')
        y_axis_values.observe(self.update_y, 'value')
        exclude_divs_box.observe(self.update_div_ex, 'value')
        interactivity_button.observe(self.update_interactivity, 'value')
        float_slider_alpha.observe(self.update_alpha, 'value')
        float_slider_lw.observe(self.update_lw, 'value')

        self.window_slider.observe(self.update_window, 'value')
        self.threshold_slider.observe(self.update_threshold, 'value')
        self.prominence_slider.observe(self.update_prominence, 'value')
        self.width_slider.observe(self.update_width, 'value')
        self.distance_slicer.observe(self.update_distance, 'value')

        self.x_label = text_xlabel.value
        self.y_label = text_ylabel.value
        self.title_label = text_title.value

        widgets.interactive_output(self.update_channel, self.checkbox1_arg_dict)
        widgets.interactive_output(self.update_color, self.checkbox2_arg_dict)

        checkbox_xaxis.observe(self.update_xaxis, 'value')
        checkbox_yaxis.observe(self.update_yaxis, 'value')

        dropbox2.observe(self.update_norm, 'value')

        # Obtain the values from the widgets to use later in the functions
        # ----------------------------------------------------------------
        self.chose_channel = [True] * self.n_channels

        # Initially, the axis are not fixed
        self.chose_fix_x = False
        self.chose_fix_y = False

        # First cell to plot is cell 0
        self.chose_cell = 0

        # The initial colors to show are some pre-selected by me ;)
        self.chose_color = [initial_color[i] for i in range(self.n_channels)]
        # Default value for normalization defined when creating Dropbox widget
        self.chose_norm = dropbox2.value

        self.x_label = text_xlabel.value
        self.y_label = text_ylabel.value
        self.title_label = text_title.value
        self.chose_y = y_axis_values.value
        # Initial y and x lim value
        self.x_lim = [0, self.time_val]
        self.y_lim = [min_val, np.max(self.max_val)]
        self.exclude_divs = exclude_divs_box.value
        self.chose_alpha = float_slider_alpha.value
        self.chose_lw = float_slider_lw.value
        self.chose_interactivity = interactivity_button.value

        # Interactive widgets which depend on other widgets
        # -------------------------------------------------
        widgets.interactive_output(self.update_channel, self.checkbox1_arg_dict)
        out_manual = widgets.interact_manual(self.plot_lines)

        # Change the label from "Run Interact" to a specific one
        out_manual.widget.children[0].description = 'Plot'

        controls1 = widgets.VBox([interactivity_button, self.int_slider, exclude_divs_box, checkbox1, dropbox2,
                                  y_axis_values], layout={'width': '300px'})
        controls2 = widgets.VBox([self.int_range_slider1, int_range_slider2, checkbox_xaxis, checkbox_yaxis, text_xlabel,
                                  text_ylabel, text_title, float_slider_alpha, float_slider_lw, checkbox2],
                                 layout={'width': '300px'})
        controls3 = widgets.VBox([self.window_slider, self.prominence_slider, self.width_slider, self.threshold_slider,
                                  self.distance_slicer], layout={'width': '300px'})

        # Tabs for the menu
        # ------------------
        # Combine the controls from above into a single tab to show for the user
        tab = widgets.Tab([controls1, controls2, controls3])
        tab.set_title(0, 'Plotting values')
        tab.set_title(1, 'Plotting axes')
        tab.set_title(2, 'Peak Detection Parameters')

        self.children = [tab, output]

        self.plot_lines()

    # Functions for widgets
    # ----------------------

    def update_window(self, change):
        """ Updates the value of the averaging window size."""
        # If true, exclude cell division
        self.window = change.new

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                self.plot_lines()

    def update_prominence(self, change):
        """ Updates the value of the prominence for the peak detection."""
        # If true, exclude cell division
        self.prominence_val = change.new

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                self.plot_lines()

    def update_width(self, change):
        """ Updates the value of the width for the peak detection."""
        # If true, exclude cell division
        self.width_val = change.new

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                self.plot_lines()

    def update_threshold(self, change):
        """ Updates the value of the threshold for the peak detection."""
        # If true, exclude cell division
        self.threshold_val = change.new

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                self.plot_lines()

    def update_distance(self, change):
        """ Updates the value of the distance for the peak detection."""
        # If true, exclude cell division
        self.distance_val = change.new

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                self.plot_lines()

    def update_lw(self, change):
        """Updates the value of the linewidth post-plotting
        Parameters
        ----------
        * self : figure
            The figure object created at the beginning. This way we can change values of the plotted figure instead of
            re-making it
        * change : line-width selected by the user in the float-slider
        Returns
        -------
        * chose_lw : float
            The value which the user selected from the float slider for the linewidth of the lines in the plot
        """
        self.chose_lw = change.new

        for line in self.fig.gca().lines:
            line.set_linewidth(change.new)

    def update_alpha(self, change):
        """Updates the value of the alpha value post-plotting
        Parameters
        ----------
        * self : figure
            The figure object created at the beginning. This way we can change values of the plotted figure instead of
            re-making it
        * change : alpha value selected by the user in the float-slider
        Returns
        -------
        * chose_alpha : float
            The value which the user selected from the float slider for the alpha of the lines in the plot
        """
        self.chose_alpha = change.new

        for line in self.fig.gca().lines:
            line.set_alpha(change.new)

    def update_interactivity(self, change):
        """Chose whether the interactivity should be disabled or not
        Parameters
        ----------
        * change : True or False
            The value from the Interactivity button clicked by the user to enable or disable interactivity
        Returns
        -------
        * chose_interactivity : bool
            The boolean value which the user selected from the interactivity button
        """

        self.chose_interactivity = change.new

    def update(self, change):
        self.chose_cell = change.new

        if self.exclude_divs is True:
            self.int_slider.max = self.N - len(self.n_links[self.n_links > 2]) - 1
        else:
            self.int_slider.max = self.N - 1

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                """Plot the lines when the user clicks the Plot button"""
                self.plot_lines()

    def update_div_ex(self, change):
        # If true, exclude cell division
        self.exclude_divs = change.new

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                self.plot_lines()

    def update_y(self, change):
        self.chose_y = change.new

        if self.chose_norm == 'Yes':
            self.int_range_slider1.max = 1
            self.int_range_slider1.step = 0.1
        else:
            self.int_range_slider1.max = np.max(self.max_val)
            self.int_range_slider1.step = 10

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                self.plot_lines()

    def plot_lines(self):

        self.ax.clear()

        # Selected channels by the user
        inds_channel = np.where(np.array(self.chose_channel))[0]

        # Use the measure the user chose
        y_value = self.chose_y

        self.y = []

        for i in range(self.n_channels):
            self.y.append(self.spots.data_df_sorted[f'{y_value} ch{i+1}'])

        # If exclude division was selected, this variable will be True
        ex_divs = self.exclude_divs

        # Cell to plot, selected by the user
        if ex_divs is True:
            cell = np.unique(self.spots.data_df_sorted['Track ID'])[self.no_div_inds][self.chose_cell]
        else:
            cell = np.unique(self.spots.data_df_sorted['Track ID'])[self.chose_cell]

        # Selected colors for each channel by the user
        col = self.chose_color

        # Line style for dividing cells
        ls = ['solid', 'dashed', 'dotted']

        # Alpha value
        alpha = self.chose_alpha

        # Line-width value
        lw = self.chose_lw

        # Check the number of links to know whether the cell divides
        idx_ = self.spots.data_df_sorted['Track ID'] == cell

        if self.chose_norm == 'Yes':
            # Find the maximum value to normalize
            self.max_val = np.zeros(self.n_channels)
            # Calculate the new max intensity value (depending on the intensity measure chosen by user)
            for c in inds_channel:
                for idx_ind, idx in enumerate(np.unique(self.spots.data_df_sorted['Track ID'])):

                    i = self.spots.data_df_sorted['Track ID'] == idx

                    # If we clicked to exclude divisions and we find a division, we exclude it
                    if ex_divs is True and self.n_links[idx_ind] == 3:
                        continue
                    else:
                        if np.max(self.y[c][i].to_numpy()) > self.max_val[c]:
                            self.max_val[c] = np.max(self.y[c][i].to_numpy())

            for c in inds_channel:
                if self.n_links[cell] > 2:
                    spot_source = self.spots.edges_df_sorted['source ID'].to_numpy(dtype=int)
                    spot_target = self.spots.edges_df_sorted['target ID'].to_numpy(dtype=int)
                    spots_id = self.spots.data_df_sorted.loc[idx_, 'Spot ID'].values
                    n_links_all = self.spots.data_df_sorted.loc[idx_, 'N Links'].values

                    frames_out, intensity_out = self.cell_division_arrange(self.x[cell].to_numpy(), n_links_all,
                                                                           spots_id, self.y[c][cell].to_numpy() /
                                                                           self.max_val[c], spot_source, spot_target)
                    for div_i in range(len(frames_out)):
                        # Peak detection
                        peaks, cA = self.peak_detection(intensity_out[div_i])

                        # Initial time-point where the trace starts
                        init_t = frames_out[div_i][0]

                        # Plot the smoothed line and the peaks
                        self.line1, = self.ax.plot(np.arange(init_t, len(cA)+init_t) * self.tr_min, cA,
                                                      color=col[c], linewidth=lw, linestyle=ls[div_i], alpha=alpha)
                        self.line2, = self.ax.plot((peaks + init_t) * self.tr_min, cA[peaks], 'xk',
                                                  markersize=10)
                else:
                    # Peak detection
                    peaks, cA = self.peak_detection(self.y[c][cell].to_numpy())

                    # Initial time-point where the trace starts
                    init_t = self.x[cell].to_numpy()[0]

                    # Plot the smoothed line and the peaks
                    self.line1, = self.ax.plot(np.arange(init_t, len(cA)+init_t) * self.tr_min, cA/self.max_val[c], color=col[c],
                                              linewidth=lw, alpha=alpha)
                    self.line2, = self.ax.plot((peaks + init_t) * self.tr_min, cA[peaks]/self.max_val[c], 'xk', markersize=10)

            self.fig.canvas.draw_idle()

        else:
            for c in inds_channel:
                if self.n_links[cell] > 2:
                    spot_source = self.spots.edges_df_sorted['source ID'].to_numpy(dtype=int)
                    spot_target = self.spots.edges_df_sorted['target ID'].to_numpy(dtype=int)
                    spots_id = self.spots.data_df_sorted.loc[idx_, 'Spot ID'].values
                    n_links_all = self.spots.data_df_sorted.loc[idx_, 'N Links'].values

                    frames_out, intensity_out = self.cell_division_arrange(self.x[cell].to_numpy(), n_links_all,
                                                                           spots_id, self.y[c][cell].to_numpy(),
                                                                           spot_source, spot_target)
                    for div_i in range(len(frames_out)):

                        # Peak detection
                        peaks, cA = self.peak_detection(intensity_out[div_i])

                        # Initial time-point where the trace starts
                        init_t = frames_out[div_i][0]

                        # Plot the smoothed line and the peaks
                        self.line1, = self.ax.plot(np.arange(init_t, len(cA)+init_t) * self.tr_min, cA,
                                                      color=col[c], linewidth=lw, linestyle=ls[div_i], alpha=alpha)
                        self.line2, = self.ax.plot((peaks + init_t) * self.tr_min, cA[peaks], 'xk',
                                                  markersize=10)

                else:
                    # Peak detection
                    peaks, cA = self.peak_detection(self.y[c][cell].to_numpy())

                    # Initial time-point where the trace starts
                    init_t = self.x[cell].to_numpy()[0]

                    # Plot the smoothed line and the peaks
                    self.line1, = self.ax.plot(np.arange(init_t, len(cA)+init_t) * self.tr_min, cA, color=col[c],
                                              linewidth=lw, alpha=alpha)
                    self.line2, = self.ax.plot((peaks + init_t) * self.tr_min, cA[peaks], 'xk', markersize=10)

            self.fig.canvas.draw_idle()

        # Fix y and x axis according to whether the fix Checkbox was checked
        if self.chose_fix_x:
            self.ax.set_xlim([self.x_lim[0], self.x_lim[1]])
        if self.chose_fix_y:
            self.ax.set_ylim([self.y_lim[0], self.y_lim[1]])

        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_title(self.title_label)

        self.cpu_percentage.value = cpu_percent()

    def update1(self, change):
        """redraw line (update plot)"""
        self.ax.set_ylim([change.new[0], change.new[1]])
        self.y_lim = change.new

        self.fig.canvas.draw_idle()

    def update2(self, change):
        """redraw line (update plot)"""
        self.ax.set_xlim([change.new[0], change.new[1]])
        self.x_lim = change.new

        self.fig.canvas.draw_idle()

    def update_xaxis(self, change):
        self.chose_fix_x = change.new

    def update_yaxis(self, change):
        self.chose_fix_y = change.new

    def update_channel(self, **kwargs):
        self.chose_channel = []
        for key in kwargs:
            self.chose_channel.append(kwargs[key])

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                self.plot_lines()

    def update_norm(self, change):
        self.chose_norm = change.new

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                self.plot_lines()

    def update_color(self, **kwargs):
        """set line color"""
        self.chose_color = []
        for key in kwargs:
            self.chose_color.append(kwargs[key])

        for i, line in enumerate(self.fig.gca().lines):
            line.set_color(self.chose_color[i])

    def update_xlabel(self, change):
        self.x_label = change.new
        self.ax.set_xlabel(change.new, fontsize=20)

    def update_ylabel(self, change):
        self.y_label = change.new
        self.ax.set_ylabel(change.new, fontsize=20)

    def update_title(self, change):
        self.title_label = change.new
        self.ax.set_title(change.new, fontsize=25)

    def peak_detection(self, y_signal):

        # Smooth the curve
        cA = paleo_functions.smoothing_filter(y_signal, self.window)  # To find Maxima
        # Peak detection
        peaks, _ = signal.find_peaks(cA, prominence=self.prominence_val, width=self.width_val,
                                     distance=self.distance_val, threshold=self.threshold_val)
        peaks = np.unique(peaks)

        return peaks, cA

    @staticmethod
    def cell_division_arrange(frames, n_links, spots_id, intensity, spot_source, spot_target):
        # ORDER TRACK BY ITS ID

        sorted_frames = frames.argsort()

        # Save the sorted frames
        frames_by_track = np.sort(frames)

        # Save the sorted links (1, 2 or 3 links in case of division)
        links_by_track = n_links[sorted_frames]

        # spots_id by tracks
        spots_id_by_track = spots_id[sorted_frames]

        # Intensity by tracks
        intensity_by_track = intensity[sorted_frames]

        # Sorted spot source ID and spot target ID

        source_by_track = spot_source[np.array([ind for ind, element in enumerate(spot_source) if element in
                                                spots_id_by_track[:-1]])]
        target_by_track = spot_target[np.array([ind for ind, element in enumerate(spot_target) if element in
                                                spots_id_by_track[1:]])]

        frames_out = []
        intensity_out = []

        # Are there any divisions in the track?
        # (e.g. the spot divides in two different opportunities during all the timeseries)
        n_divs = len(list(map(int, np.where(links_by_track > 2)[0])))

        # How many times the spot divides per division?
        # (e.g. in one specific division, in how many daughters the spot divided?)
        # n_divs_cell = links_by_track[links_by_track[i]>2]

        div_vect = []  # one vector for [each division+1] we want to keep track of
        val0 = np.where(links_by_track == 3)[0][0]  # index of first division

        # save the spots_id up to first division for all tracks
        for j in range(n_divs + 1):
            div_vect.append(spots_id_by_track[:val0 + 1].tolist())  # spots_id[0:first division]

        # store the list of already saved spots_id to not use them again
        list_idx_sources_used = []  # To save the indices of used sources spots
        list_used = spots_id_by_track[
                    :val0 + 1].tolist()  # list of spots_id used --> finish while loop when all spots_id used
        # while we have not stored all spots_id, loop across tracks and fill them with targets (if not in list_used)

        while not (all(elem in list_used for elem in spots_id_by_track)):

            for j in range(n_divs + 1):
                idx = np.where(source_by_track == div_vect[j][-1])[0]
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
                    div_vect[j].append(int(target_by_track[idx]))
                    list_used.append(int(target_by_track[idx]))

                # This means it finished at least one of the tracks
                if np.size(idx) == 0:
                    continue

        # Save each division tracks with its corresponding division ID and its frames and mean
        for j in range(n_divs + 1):
            # Indices for the spots_id of the tracks in one of the divisions
            inds = [np.where(spots_id_by_track == div_vect[j][ind])[0][0] for ind in range(len(div_vect[j]))]
            frames_out.append(frames_by_track[inds])
            intensity_out.append(intensity_by_track[inds])

        return frames_out, intensity_out


# Phase analysis for individual cells
class PhaseAnalysis(widgets.HBox):
    """This class is called to calculate and later plot the phase of each single cell using the Hilbert Transform.
    :class:
    :path_csv: Path where the two csv files (-edges.csv and -data.csv) are located. These are the files which are
        obtained from Mastodon
    :path_xml: Path where the .xml file created from BigDataViewer is located. This file should be in the same folder
        as the .hdf5 data
    :tr_min: Temporal resolution with which the data was acquired
    """
    def __init__(self, tracks, tr_min, window, peaks, channel):
        """Constructor method
        """
        super().__init__()

        # The X is always the frames
        self.x = tracks.spots_features['Frames']

        # Mean, Median, Sum, Min, Max (with or sithout Std)
        self.y = []
        # Initialize the first plot
        self.y = tracks.spots_features[f'Mean{channel+1}']

        self.tr_min = tr_min # in minutes
        self.T = int(tr_min * 60) # in seconds

        # Chosen channel to perform Phase analysis
        self.channel = channel

        # Smoothing window chosen for the peak detection
        self.window = window

        # Only chose the cells with peaks, avoid the excluded cells by user
        self.peak_inds = [np.size(peaks[i]) for i in range(len(peaks))]

        # Number of tracks in total
        self.N = len([i for i, val in enumerate(self.peak_inds) if val > 0]) #len(np.unique(self.spots.data_df_sorted['Track ID'].to_numpy()))

        # Initial value for CPU percentage
        self.cpu_per = cpu_percent()

        # By default, interactivity is disabled
        self.chose_interactivity = False

        # By default, alpha is 1
        self.chose_alpha = 1

        # Obtained peaks from the previous function
        self.peaks = peaks

        output = widgets.Output()

        with output:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=(6, 4))

        # To setup min and max value for the time frames

        time_val = 0

        # Initial value for CPU percentage
        self.cpu_per = cpu_percent()

        # By default, interactivity is disabled
        self.chose_interactivity = False

        # By default, alpha is 1
        self.chose_alpha = 1

        # Define min and max values for X-axis (time)
        for i in self.peak_inds:
            if np.max(self.x[i]*self.tr_min) > time_val:
                time_val = np.max(self.x[i]*self.tr_min)

        # move the toolbar to the bottom
        self.fig.canvas.toolbar_position = 'bottom'
        self.ax.grid(True)

        # Control Elements
        # -----------------
        # Intensity labels to plot - Mean, Median, Sum, Max, Min, Std
        style = {'description_width': 'initial'}

        # Cell track to plot
        self.int_slider = widgets.IntSlider(value=0, min=0, max=self.N-1, step=1, description='Cell #')

        # Order filter for Phase Analysis
        self.int_slider_order_filter = widgets.IntSlider(value=6, min=1, max=10, step=1, description='Order Filter',
                                                         layout=dict(width='90%'))

        # Critical frequencies for bandpass filter - phase analysis
        float_slider_freq = widgets.FloatRangeSlider(value=(0.0002, 0.0006), min=0.0001, max=0.001, step=0.0001,
                                                     description='Critical Frequencies: ', disabled=False,
                                                     continuous_update=False, orientation='horizontal', readout=True,
                                                     readout_format='.0e', style=style)

        # Labels for x and y axis
        text_xlabel = widgets.Text(value='', description='X-Label', continuous_update=False, layout=dict(width='90%'))
        text_ylabel = widgets.Text(value='', description='Y-Label', continuous_update=False, layout=dict(width='90%'))

        # Title for plot
        text_title = widgets.Text(value='', description='Title', continuous_update=False, layout=dict(width='90%'))

        # Min-max slider for y axis (intensity)
        int_range_slider1 = widgets.IntRangeSlider(value=(-10, 10), min=-10,  max=10, step=10, description='Y-Limit')

        # Min-max slider for x axis (time)
        int_range_slider2 = widgets.IntRangeSlider(value=(0, time_val), min=0, max=time_val*2, step=10,
                                                   description='X-Limit')

        # Bar to display the percentage of CPU used in each operation
        self.cpu_percentage = widgets.FloatProgress(value=self.cpu_per, min=0, max=100, description='CPU %:',
                                                    bar_style='', style={'bar_color': 'blue'}, orientation='horizontal')
        display(self.cpu_percentage)

        # Interactivity toggle button - if clicked, the plot button appears
        interactivity_button = widgets.ToggleButton(value=False, description='Interactivity', disabled=False,
                                                    button_style='', tool_tip='Disable interactivity mode',
                                                    icon='check')
        self.plot_button = widgets.Button(description='Plot', disabled=False, button_style='',
                                          tooltip='Plot the tracks', icon='')

        # Several color pickers according to signals to plot
        signal_desc = ['Original Signal', 'Hilbert Transform', 'Phase']
        initial_colors = ['#000000', '#0000FF', '#FF0000']
        self.color_signals = ['signal', 'hilbert', 'phase']
        self.checkbox2_arg_dict = {signal: widgets.ColorPicker(value=initial_colors[i],
                                                          description=f'Color for {signal_desc[i]}', style=style,
                                                          layout=dict(width='90%'))
                                   for i, signal in enumerate(self.color_signals)}

        checkbox2 = widgets.VBox(children=[self.checkbox2_arg_dict[signal] for signal in self.color_signals])

        # Float slider to change the alpha of the lines
        style = {'description_width': 'initial'}  # This is so that the description fits in 1 line
        float_slider_alpha = widgets.FloatSlider(value=1.0, min=0, max=1.0, step=0.1, description='Transparency: ',
                                                 disabled=False, continuous_update=False, orientation='horizontal',
                                                 readout=True, readout_format='.1f', style=style)

        # Change line-width of lines
        float_slider_lw = widgets.FloatSlider(value=1.0, min=0, max=4.0, step=0.2, description='Linewidth: ',
                                              disabled=False, continuous_update=False, orientation='horizontal',
                                              readout=True, readout_format='.1f', style=style)

        # Fix the X axis and the Y axis
        checkbox_xaxis = widgets.Checkbox(description='Fix X-Lim', value=False, disabled=False, indent=False)
        checkbox_yaxis = widgets.Checkbox(description='Fix Y-Lim', value=False, disabled=False, indent=False)

        # Connect callbacks and traits
        # -----------------------------
        self.int_slider.observe(self.update, 'value')
        int_range_slider1.observe(self.update1, 'value')
        int_range_slider2.observe(self.update2, 'value')
        text_xlabel.observe(self.update_xlabel, 'value')
        text_ylabel.observe(self.update_ylabel, 'value')
        text_title.observe(self.update_title, 'value')
        interactivity_button.observe(self.update_interactivity, 'value')
        float_slider_alpha.observe(self.update_alpha, 'value')
        float_slider_lw.observe(self.update_lw, 'value')
        self.int_slider_order_filter.observe(self.update_order_filter, 'value')
        float_slider_freq.observe(self.update_freq, 'value')

        self.x_label = text_xlabel.value
        self.y_label = text_ylabel.value
        self.title_label = text_title.value

        widgets.interactive_output(self.update_color, self.checkbox2_arg_dict)

        checkbox_xaxis.observe(self.update_xaxis, 'value')
        checkbox_yaxis.observe(self.update_yaxis, 'value')

        # Obtain the values from the widgets to use later in the functions
        # ----------------------------------------------------------------

        # Initially, the axis are not fixed
        self.chose_fix_x = False
        self.chose_fix_y = False

        # First cell to plot is cell 0
        self.chose_cell = 0

        # Order filter for Phase Analysis
        self.chose_order_filter = 6

        # The initial colors to show are some pre-selected by me ;)
        self.chose_color = [initial_colors[i] for i in range(3)]

        self.x_label = text_xlabel.value
        self.y_label = text_ylabel.value
        self.title_label = text_title.value

        # Initial y and x lim value
        self.x_lim = [0, time_val]
        self.y_lim = [-10, 10]
        self.chose_alpha = float_slider_alpha.value
        self.chose_lw = float_slider_lw.value
        self.chose_interactivity = interactivity_button.value
        self.chose_freq = float_slider_freq.value

        # Interactive widgets which depend on other widgets
        # -------------------------------------------------
        out_manual = widgets.interact_manual(self.plot_lines)

        # Change the label from "Run Interact" to a specific one
        out_manual.widget.children[0].description = 'Plot'

        # Vertical box for phase parameters
        phase_params_vbox = widgets.VBox([self.int_slider_order_filter, float_slider_freq])

        # All phase parameters inside an accordion widget - for easy accessibility
        phase_params_accordion = widgets.Accordion(children=[phase_params_vbox], layout={'width': '300px'})
        phase_params_accordion.set_title(0, 'Phase Parameters')

        controls1 = widgets.VBox([interactivity_button, self.int_slider, phase_params_accordion],
                                 layout={'width': '300px'})
        controls2 = widgets.VBox([int_range_slider1, int_range_slider2, checkbox_xaxis, checkbox_yaxis, text_xlabel,
                                  text_ylabel, text_title, float_slider_alpha, float_slider_lw, checkbox2],
                                 layout={'width': '300px'})

        # Tabs for the menu
        # ------------------
        # Combine the controls from above into a single tab to show for the user
        tab = widgets.Tab([controls1, controls2])
        tab.set_title(0, 'Plotting values')
        tab.set_title(1, 'Plotting axes')

        self.children = [tab, output]

        self.plot_lines()

    # Functions for widgets
    # ----------------------

    def update_lw(self, change):
        """Updates the value of the linewidth post-plotting
        Parameters
        ----------
        * self : figure
            The figure object created at the beginning. This way we can change values of the plotted figure instead of
            re-making it
        * change : line-width selected by the user in the float-slider
        Returns
        -------
        * chose_lw : float
            The value which the user selected from the float slider for the linewidth of the lines in the plot
        """
        self.chose_lw = change.new

        for line in self.fig.gca().lines:
            line.set_linewidth(change.new)

    def update_alpha(self, change):
        """Updates the value of the alpha value post-plotting
        Parameters
        ----------
        * self : figure
            The figure object created at the beginning. This way we can change values of the plotted figure instead of
            re-making it
        * change : alpha value selected by the user in the float-slider
        Returns
        -------
        * chose_alpha : float
            The value which the user selected from the float slider for the alpha of the lines in the plot
        """
        self.chose_alpha = change.new

        for line in self.fig.gca().lines:
            line.set_alpha(change.new)

    def update_interactivity(self, change):
        """Chose whether the interactivity should be disabled or not
        Parameters
        ----------
        * change : True or False
            The value from the Interactivity button clicked by the user to enable or disable interactivity
        Returns
        -------
        * chose_interactivity : bool
            The boolean value which the user selected from the interactivity button
        """

        self.chose_interactivity = change.new

    def update(self, change):
        self.chose_cell = change.new

        self.int_slider.max = self.N - 1

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                """Plot the lines when the user clicks the Plot button"""
                self.plot_lines()

    def update_order_filter(self, change):
        self.chose_order_filter = change.new

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                """Plot the lines when the user clicks the Plot button"""
                self.plot_lines()

    def update_freq(self, change):
        self.chose_freq = change.new

        # Re-plot the figure
        if self.chose_interactivity is True:
            self.plot_lines()
        else:
            @self.plot_button.on_click
            def plot_on_click():
                """Plot the lines when the user clicks the Plot button"""
                self.plot_lines()


    def plot_lines(self):

        self.ax.clear()

        # Cell to plot, selected by the user
        cell = self.peak_inds[self.chose_cell] #np.unique(self.spots.data_df_sorted['Track ID'])[self.chose_cell]

        # Selected colors for each channel by the user
        col = self.chose_color

        # Alpha value
        alpha = self.chose_alpha

        # Line-width value
        lw = self.chose_lw

        # Crop if there are peaks detected, otherwise do not crop
        if np.size(self.peaks[cell]) > 0:
            y_cropped = self.crop_trace(self.y[cell], self.window, self.peaks[cell])
        else:
            y_cropped = paleo_functions.smoothing_filter(self.y[cell], self.window)

        # Phase analysis
        instantaneous_phase_filt, y_filt = self.phase_analysis_calculation(y_cropped, self.T)

        x = np.linspace(0.0, len(y_cropped) * self.tr_min, len(y_cropped))

        self.line, = self.ax.plot(x*self.tr_min, scipy.stats.zscore(y_cropped), color=col[0], linewidth=lw, alpha=alpha,
                                  label='original')
        self.line, = self.ax.plot(x*self.tr_min, scipy.stats.zscore(y_filt), color=col[1], linewidth=lw, alpha=alpha,
                                  label='filtered')
        self.line, = self.ax.plot(x*self.tr_min, instantaneous_phase_filt, color=col[2], linewidth=lw, alpha=alpha,
                                  label='filtered-phase')

        self.fig.canvas.draw_idle()

        # Fix y and x axis according to whether the fix Checkbox was checked
        if self.chose_fix_x:
            self.ax.set_xlim([self.x_lim[0], self.x_lim[1]])
        if self.chose_fix_y:
            self.ax.set_ylim([self.y_lim[0], self.y_lim[1]])

        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_title(self.title_label)

        self.cpu_percentage.value = cpu_percent()

    def update1(self, change):
        """redraw line (update plot)"""
        self.ax.set_ylim([change.new[0], change.new[1]])
        self.y_lim = change.new

        self.fig.canvas.draw_idle()

    def update2(self, change):
        """redraw line (update plot)"""
        self.ax.set_xlim([change.new[0], change.new[1]])
        self.x_lim = change.new

        self.fig.canvas.draw_idle()

    def update_xaxis(self, change):
        self.chose_fix_x = change.new

    def update_yaxis(self, change):
        self.chose_fix_y = change.new

    def update_color(self, **kwargs):
        """set line color"""
        self.chose_color = []
        for key in kwargs:
            self.chose_color.append(kwargs[key])

        for i, line in enumerate(self.fig.gca().lines):
            line.set_color(self.chose_color[i])

    def update_xlabel(self, change):
        self.x_label = change.new
        self.ax.set_xlabel(change.new, fontsize=20)

    def update_ylabel(self, change):
        self.y_label = change.new
        self.ax.set_ylabel(change.new, fontsize=20)

    def update_title(self, change):
        self.title_label = change.new
        self.ax.set_title(change.new, fontsize=25)

    def crop_trace(self, y, window, peaks):
        """Crop the last part of the trace
        To perform the Phase analysis, we use only the oscillatory part of the cell, so after the last detected peak,
        we ignore that area since this could add noise into the Hilbert Transform analysis which will be done later. """

        y_smoothed = paleo_functions.smoothing_filter(y, window)
        # Only keep the part from the last peak towards the end
        offset_traces = y_smoothed[peaks[-1]:]
        # Cluster data into 2 groups
        window_cluster = 5

        trace = offset_traces[~np.isnan(offset_traces)]
        X = np.zeros((len(trace[window_cluster:]), 2))
        X[:, 0] = trace[window_cluster:]
        X[:, 1] = trace[:-window_cluster]
        y_pred = KMeans(n_clusters=2, n_init=200).fit_predict(X)
        ind_cut = np.where(y_pred[window:] == y_pred[-1])[0] + window

        # Save the new traces:
        end_trace = peaks[-1] + ind_cut[0]
        trace_save = y_smoothed[:end_trace]

        return trace_save

    def phase_analysis_calculation(self, y, T):

        # Sample spacing
        fs = 1/T

        # Order filter
        order_filter = self.chose_order_filter

        # Critical frequencies for butter-worth filter
        c_freq = self.chose_freq

        # Butter-worth filter
        b, a = scipy.signal.butter(N=order_filter, Wn=[c_freq[0], c_freq[1]], fs=fs, btype='band')
        y_filt = scipy.signal.filtfilt(b, a, y)

        # Hilbert Transform
        analytic_signal_filt = scipy.signal.hilbert(y_filt)

        # Phase calculation
        instantaneous_phase_filt = np.angle(analytic_signal_filt)

        return instantaneous_phase_filt, y_filt


# Check how far apart division cells move in 3D
class Spatial3DAnalaysis(widgets.HBox):
    #def __init__(self, x, y, division, ids, tr_min, N, n_channels, cell_neighbors):
    def __init__(self, path_xml, path_csv, tr_min, tracks):
        super().__init__()

        # Contains all the spots information
        self.spots_all= csv_reader(path_csv, path_xml)

        # From csv_features and ordering tracks - specific for cell division
        self.division = tracks.spots_features['DivID']
        self.ids = tracks.spots_features['ID']
        self.n = tracks.n_tracks_divs
        self.n_divs_total = tracks.n_division_tracks//2
        # Get image characteristics
        fts = xml_features(path_xml)

        # XYZ dimensions
        x_max = fts.width * fts.x_pixel
        y_max = fts.height * fts.y_pixel
        z_max = fts.n_slices * fts.z_pixel

        # All possible intensity measures Mastodon can export
        intensity_labels = np.array(['Mean', 'Std', 'Min', 'Max', 'Median', 'Sum', 'Center'])  # For each channel

        # Description for all possible intensity labels
        description = np.array(['Mean', 'Standard deviation', 'Minimum intensity', 'Maximum intensity',
                                'Median intensity', 'Sum intensity', 'Center intensity'])

        # Which were the measures Mastodon exported?
        vals_to_check = np.array([i.split()[0] for i in list(self.spots_all.data_df_sorted.keys())])
        inds_vals = np.array([any(i == intensity_labels) for i in vals_to_check])
        intensity_labels_to_use = np.unique(vals_to_check[inds_vals])

        # Descriptions for measures to use
        descriptions_to_use = description[np.array([any(i == intensity_labels) for i in intensity_labels_to_use])]

        # Number of channels
        self.n_channels = fts.channels

        # Spot IDs from csv reader and csv features
        self.spot_ids = self.spots_all.data_df_sorted['Spot ID'].values
        self.ids = tracks.spots_features['ID']

        # Mean, Median, Sum, Min, Max (wit h or without Std)
        self.y = []
        # Initialize the first plot
        self.y_value = intensity_labels_to_use[0]
        for i in range(self.n_channels):
            self.y.append(self.spots_all.data_df_sorted[f'{self.y_value} ch{i + 1}'])

        # Number of tracks in total
        self.N = len(np.unique(self.spots_all.data_df_sorted['Track ID'].to_numpy()))

        # The x-axis will always be time
        self.t = self.spots_all.data_df_sorted['Frames']
        self.tr_min = tr_min

        self.channels = [f'Channel {i+1}' for i in range(self.n_channels)]

        output = widgets.Output()

        with output:
            self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 3),
                                                                    constrained_layout=True)
        self.cb1 = None
        self.cb2 = None
        self.cb3 = None

        # To setup min and max value for the intensities
        min_val = np.inf
        self.max_val = np.zeros(self.n_channels)
        time_val = 0

        for c in range(self.n_channels):
            for idx in np.unique(self.spots_all.data_df_sorted['Track ID']):

                i = self.spots_all.data_df_sorted['Track ID']==idx

                if np.max(self.y[c][i].to_numpy())>self.max_val[c]:
                    self.max_val[c] = np.max(self.y[c][i].to_numpy())


        # move the toolbar to the bottom
        self.fig.canvas.toolbar_position = 'bottom'

        # create some control elements
        y_axis_values = widgets.ToggleButtons(options=intensity_labels_to_use.tolist(), description='Measures:',
                                             disabled=False, button_style='', tooltips=descriptions_to_use.tolist(),
                                              layout=widgets.Layout(width='500px'))
        int_slider = widgets.IntSlider(value=0, min=0, max=fts.n_frames , step=1, description='Frames')
        text_xlabel = widgets.Text(value='', description='X-Label', continuous_update=False)
        text_ylabel = widgets.Text(value='', description='Y-Label', continuous_update=False)
        text_zlabel = widgets.Text(value='', description='Z-Label', continuous_update=False)
        text_title = widgets.Text(value='', description='Title', continuous_update=False)
        int_range_slider1 = widgets.IntRangeSlider(value=(0, x_max), min=0, max=x_max, step=10, description='X-Limit')
        int_range_slider2 = widgets.IntRangeSlider(value=(0, y_max), min=0, max=y_max, step=10, description='Y-Limit')
        int_range_slider3 = widgets.IntRangeSlider(value=(0, z_max), min=0, max=z_max, step=10, description='Z-Limit')
        int_range_slider_int = widgets.IntRangeSlider(value=(0, np.max(self.max_val)), min=0, max=np.max(self.max_val),
                                                      step=10, description='Intensity Values')
        checkbox_int = widgets.Checkbox(description='Fix Intensity', value=False, disabled=False, indent=False)

        dropbox2 = widgets.Dropdown(options=['Yes', 'No'], description='Normalize Data',
                    value='No',rows=2, interactive=True)

        # Checkbox for the channels
        #checkbox1_kwargs = {'value':True, 'disabled':False, 'indent':False}
        #self.checkbox1_arg_dict = {channel: widgets.Checkbox(description=channel, **checkbox1_kwargs) for channel in self.channels}
        #checkbox1 = widgets.VBox(children=[self.checkbox1_arg_dict[channel] for channel in self.channels])

        radio_buttons = widgets.RadioButtons(options=[channel for channel in self.channels], value=self.channels[0])

        # Widgets for changing the colors, size, etc of the scatter plot
        alpha_slider = widgets.FloatSlider(value=1, min=0, max=1, step=0.1, description='Transparency')
        size_slider = widgets.IntSlider(value=80, min=10, max=500, step=20, description='Spot size')
        cmap_picker = widgets.Dropdown(options=['jet', 'inferno', 'viridis'], description='Colormap', value='jet',
                                       rows=3, interactive=True)

        # Several color pickers according to number of channels
        #self.checkbox2_arg_dict = {channel: widgets.ColorPicker(value=initial_color[i], description='Color for channel %d'%i) for i,channel in enumerate(self.channels)}
        #checkbox2 = widgets.VBox(children=[self.checkbox2_arg_dict[channel] for channel in self.channels])

        # Fix the X axis and the Y axis
        checkbox_xaxis = widgets.Checkbox(description='Fix X-Lim', value=False, disabled=False, indent=False)
        checkbox_yaxis = widgets.Checkbox(description='Fix Y-Lim', value=False, disabled=False, indent=False)
        checkbox_zaxis = widgets.Checkbox(description='Fix Z-Lim', value=False, disabled=False, indent=False)

        # Make the widgets look good! :P
        #int_slider.style.handle_color = 'black'

        # connect callbacks and traits
        int_slider.observe(self.update, 'value')
        int_range_slider1.observe(self.update1, 'value')
        int_range_slider2.observe(self.update2, 'value')
        int_range_slider3.observe(self.update3, 'value')
        int_range_slider_int.observe(self.update_int, 'value')
        checkbox_int.observe(self.update_fix_int, 'value')
        #color_picker.observe(self.line_color, 'value')
        text_xlabel.observe(self.update_xlabel, 'value')
        text_ylabel.observe(self.update_ylabel, 'value')
        text_zlabel.observe(self.update_zlabel, 'value')
        text_title.observe(self.update_title, 'value')
        y_axis_values.observe(self.update_y, 'value')

        alpha_slider.observe(self.update_alpha, 'value')
        size_slider.observe(self.update_size, 'value')
        cmap_picker.observe(self.update_cmap, 'value')

        self.x_label = text_xlabel.value
        self.y_label = text_ylabel.value
        self.z_label = text_zlabel.value
        self.title_label = text_title.value
        self.chose_fix_int = False

        checkbox_xaxis.observe(self.update_xaxis, 'value')
        checkbox_yaxis.observe(self.update_yaxis, 'value')
        checkbox_zaxis.observe(self.update_zaxis, 'value')

        radio_buttons.observe(self.update_channel, 'value')
        dropbox2.observe(self.update_norm, 'value')

        ### CHOSE INITIAL VALUES TO PLOT
        # Initially, all channels are shown
        self.chose_channel = self.channels[0] #[True for i in range(self.n_channels)]

        # Intially, the axis are not fixed
        self.chose_fix_x = False
        self.chose_fix_y = False
        self.chose_fix_z = False

        # First time-point to plot is 0
        self.chose_tp = 0
        # Initial y value
        self.chose_y = y_axis_values.value
        # Save the value and if the user does not change it, do not re-calculate all the y values
        self.chose_y_save = y_axis_values.value

        # The initial colors to show are some pre-selected by me ;)
        #self.chose_color = [initial_color[i] for i in range(self.n_channels)]
        # Default value for normalization defined when creating Dropbox widget
        self.chose_norm = dropbox2.value

        # Scatter plot parameters
        self.size = 80
        self.cmap = 'jet'
        self.alpha = 1

        #out1 = widgets.interactive_output(self.update_channel, self.checkbox1_arg_dict)

        # Initial y and x lim value
        self.x_lim = [0, x_max]
        self.y_lim = [0, y_max]
        self.z_lim = [0, z_max]
        self.int_lim = [0, np.max(self.max_val)]

        controls1 = widgets.HBox([widgets.VBox([int_slider, radio_buttons, dropbox2]), y_axis_values],
                                 layout={'width': '100%', 'height': '200px'})
        controls2 = widgets.HBox([widgets.VBox([int_range_slider1, int_range_slider2, int_range_slider3]),
                                  widgets.VBox([checkbox_xaxis, checkbox_yaxis, checkbox_zaxis]),
                                  widgets.VBox([text_xlabel, text_ylabel, text_zlabel, text_title])],
                                  layout={'width': '100%', 'height': '200px'})
        controls3 = widgets.HBox([widgets.VBox([alpha_slider, size_slider, cmap_picker]),
                                  widgets.VBox([int_range_slider_int, checkbox_int])],
                                 layout={'width': '100%', 'height': '200px'})

        #controls.layout = make_box_layout()
        #out_box = widgets.Box([output]) d
        #output.layout = make_box_layout()

        tab = widgets.Tab([controls1, controls2, controls3], layout={'height': '200px'})
        tab.set_title(0, 'Plotting values')
        tab.set_title(1, 'Plotting axes')
        tab.set_title(2, 'Scatter plot')

        self.children = [widgets.VBox([tab, output])]

        self.plot_lines()

        # callback functions

    def update_alpha(self, change):
        """Update the alpha value chosen with the slider by the user"""
        self.alpha = change.new
        self.plot_lines()

    def update_size(self, change):
        """Update the size of scatter spots chosen with the slider by the user"""
        self.size = change.new
        self.plot_lines()

    def update_cmap(self, change):
        """Update the colormap chosen with the dropdown menu by the user"""
        self.cmap = change.new
        self.plot_lines()

    def update(self, change):
        self.chose_tp = change.new
        self.plot_lines()

    def update_y(self, change):
        self.chose_y = change.new
        self.plot_lines()

    def plot_lines(self):

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        if self.cb1 is None:
            pass
        else:
            self.cb1.remove()
            self.cb2.remove()
            self.cb3.remove()

        # Selected channels by the user
        inds_channel = int(np.where(np.array(self.channels) == self.chose_channel)[0]) #np.where(np.array(self.chose_channel))[0]

        # Selected time-point by the user
        tp = self.chose_tp


        # Use the measure the user chose
        y_value = self.chose_y

        if self.chose_y_save != y_value:
            self.y = []
            #for i in inds_channel:
            for i in range(self.n_channels):
                self.y.append(self.spots_all.data_df_sorted[f'{y_value} ch{i+1}'])


        if self.chose_norm == 'Yes':

            # Find the maximum value to normalize
            self.max_val = np.zeros(self.n_channels)
            # Calculate the new max intensity value (depending on the intensity measure chosen by user)
            c = inds_channel
            for idx in np.unique(self.spots_all.data_df_sorted['Frames']):

                i = self.spots_all.data_df_sorted['Frames'] == idx

                if np.max(self.y[c][i].to_numpy()) > self.max_val[c]:
                    self.max_val[c] = np.max(self.y[c][i].to_numpy())

            c =  inds_channel
            # Indices of all the spots in frame selected by user
            inds = self.spots_all.data_df_sorted['Frames'] == tp

            # X, Y, Z coordinates
            x_coords = self.spots_all.data_df_sorted['X'].values[inds]
            y_coords = self.spots_all.data_df_sorted['Y'].values[inds]
            z_coords = self.spots_all.data_df_sorted['Z'].values[inds]

            self.line1 = self.ax1.scatter(x_coords, y_coords, s=self.size, alpha=self.alpha, cmap=self.cmap,
                                          edgecolor='k', c=self.y[c].values[inds]/self.max_val[c])
            self.line2 = self.ax2.scatter(x_coords, z_coords, s=self.size, alpha=self.alpha, cmap=self.cmap,
                                          edgecolor='k', c=self.y[c].values[inds]/self.max_val[c])
            self.line3 = self.ax3.scatter(y_coords, z_coords, s=self.size, alpha=self.alpha, cmap=self.cmap,
                                          edgecolor='k', c=self.y[c].values[inds]/self.max_val[c])

            self.cb1 = self.fig.colorbar(self.line1, ax=[(self.ax1, self.ax2, self.ax3)], location='bottom')
            self.cb2 = self.fig.colorbar(self.line2, ax=[(self.ax1, self.ax2, self.ax3)], location='bottom')
            self.cb3 = self.fig.colorbar(self.line3, ax=[(self.ax1, self.ax2, self.ax3)], location='bottom')

            if self.chose_fix_int:
                self.cb1.mappable.set_clim(self.int_lim[0], self.int_lim[1])
                self.cb2.mappable.set_clim(self.int_lim[0], self.int_lim[1])
                self.cb3.mappable.set_clim(self.int_lim[0], self.int_lim[1])

            self.fig.canvas.draw_idle()


        else:
            c = inds_channel
            # Indices of all the spots in frame selected by user
            inds = self.spots_all.data_df_sorted['Frames'] == tp

            # X, Y, Z coordinates
            x_coords = self.spots_all.data_df_sorted['X'].values[inds]
            y_coords = self.spots_all.data_df_sorted['Y'].values[inds]
            z_coords = self.spots_all.data_df_sorted['Z'].values[inds]

            self.line1 = self.ax1.scatter(x_coords, y_coords, s=self.size, alpha=self.alpha, cmap=self.cmap,
                                          edgecolor='k', c=self.y[c].values[inds])
            self.line2 = self.ax2.scatter(x_coords, z_coords, s=self.size, alpha=self.alpha, cmap=self.cmap,
                                          edgecolor='k', c=self.y[c].values[inds])
            self.line3 = self.ax3.scatter(y_coords, z_coords, s=self.size, alpha=self.alpha, cmap=self.cmap,
                                          edgecolor='k', c=self.y[c].values[inds])

            self.cb1 = self.fig.colorbar(self.line1, ax=[(self.ax1, self.ax2, self.ax3)], location='bottom')
            self.cb2 = self.fig.colorbar(self.line2, ax=[(self.ax1, self.ax2, self.ax3)], location='bottom')
            self.cb3 = self.fig.colorbar(self.line3, ax=[(self.ax1, self.ax2, self.ax3)], location='bottom')

            if self.chose_fix_int:
                self.cb1.mappable.set_clim(self.int_lim[0], self.int_lim[1])
                self.cb2.mappable.set_clim(self.int_lim[0], self.int_lim[1])
                self.cb3.mappable.set_clim(self.int_lim[0], self.int_lim[1])

            self.fig.canvas.draw_idle()


        # Fix y and x axis according to whether the fix Checkbox was checked
        if self.chose_fix_x:
            self.ax1.set_xlim([self.x_lim[0], self.x_lim[1]])
            self.ax2.set_xlim([self.x_lim[0], self.x_lim[1]])
        if self.chose_fix_y:
            self.ax1.set_ylim([self.y_lim[0], self.y_lim[1]])
            self.ax3.set_xlim([self.y_lim[0], self.y_lim[1]])
        if self.chose_fix_z:
            self.ax2.set_ylim([self.z_lim[0], self.z_lim[1]])
            self.ax3.set_ylim([self.z_lim[0], self.z_lim[1]])

        self.ax1.set_xlabel(self.x_label)
        self.ax2.set_xlabel(self.x_label)
        self.ax2.set_ylabel(self.y_label)
        self.ax3.set_xlabel(self.y_label)
        self.ax2.set_ylabel(self.z_label)
        self.ax3.set_ylabel(self.z_label)
        self.ax2.set_title(self.title_label)

    def update_int(self, change):
        """Update the intensity slider chosen with the slider by the user"""
        self.int_lim = change.new
        self.plot_lines()

    def update_fix_int(self, change):
        """Fix intensity selected by user"""
        self.chose_fix_int = change.new

    def update1(self, change):
        """Update all the x-coords axes"""
        self.ax1.set_xlim([change.new[0], change.new[1]])
        self.ax2.set_xlim([change.new[0], change.new[1]])
        self.x_lim = change.new
        #fig.canvas.draw()
        self.fig.canvas.draw_idle()

    def update2(self, change):
        """Update all the y-coords axes"""
        self.ax1.set_ylim([change.new[0], change.new[1]])
        self.ax3.set_xlim([change.new[0], change.new[1]])
        self.y_lim = change.new
        #fig.canvas.draw()
        self.fig.canvas.draw_idle()

    def update3(self, change):
        """Update all the z-coords axes"""
        self.ax2.set_ylim([change.new[0], change.new[1]])
        self.ax3.set_ylim([change.new[0], change.new[1]])
        self.z_lim = change.new
        #fig.canvas.draw()
        self.fig.canvas.draw_idle()

    def update_xaxis(self, change):
        self.chose_fix_x = change.new

    def update_yaxis(self, change):
        self.chose_fix_y = change.new

    def update_zaxis(self, change):
        self.chose_fix_z = change.new

    def update_channel(self, change):
        #self.chose_channel = []
        #for key in kwargs:
        #    self.chose_channel.append(kwargs[key])
        #self.plot_lines()

        self.chose_channel = change.new
        self.plot_lines()

    def update_norm(self, change):
        self.chose_norm = change.new
        self.plot_lines()

    def update_xlabel(self, change):
        self.x_label = change.new
        self.ax1.set_xlabel(change.new, fontsize=20)
        self.ax2.set_xlabel(change.new, fontsize=20)

    def update_ylabel(self, change):
        self.y_label = change.new
        self.ax1.set_ylabel(change.new, fontsize=20)
        self.ax3.set_xlabel(change.new, fontsize=20)

    def update_zlabel(self, change):
        self.z_label = change.new
        self.ax2.set_ylabel(change.new, fontsize=20)
        self.ax3.set_ylabel(change.new, fontsize=20)

    def update_title(self, change):
        self.title_label = change.new
        self.ax2.set_title(change.new, fontsize=25)






