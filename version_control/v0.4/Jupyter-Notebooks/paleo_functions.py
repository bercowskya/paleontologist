import numpy as np
from scipy import signal, spatial
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# Smoothing function with a defined window length and type of window
def smoothing_filter(x, window_len, window='hanning'):
    """
    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    return y


# Calculate the peaks using a series of rules:
class peak_detection:
    def __init__(self, x, window, th_val, th_val_, dist_val, dist_val_, prominence_val, prominence_val_, width_val,
                 width_val_):

        # Initialize the variables to save the data analysis results
        T_plus_final = []
        T_minus_final = []
        A_plus_final = []
        A_minus_final = []
        cycles_final = []
        self.n_cycles = []

        # Initialize the variables to save the data analysis results that will be used to plot
        self.cycles_T_minus_plot = []
        self.cycles_T_plus_plot = []
        self.cycles_A_minus_plot = []
        self.cycles_A_plus_plot = []

        # Counter in case there are no peaks detected it will go from 0 to 1
        self.count = 0

        def pair_wise_dist(r):
            dists = spatial.distance.pdist(r[:, None], 'cityblock')
            return dists

        def distance_neighbors(r):
            pwd_peaks = pair_wise_dist(r)
            peaks_neighbors = np.zeros(len(r) - 1)
            count = len(r) - 2
            aux = 0
            for i in range(len(r) - 1):
                peaks_neighbors[i] = pwd_peaks[i + aux]  # Only take distance of neighboring peals
                aux = aux + count
                count -= 1
            return peaks_neighbors

        cA = smoothing_filter(x, window)  # To find Maxima
        cA_ = cA * (-1)  # To find minima

        # Maxima
        self.peaks = []
        self.peaks, _ = signal.find_peaks(cA, prominence=prominence_val, width=width_val, distance=dist_val,
                                          threshold=th_val)
        self.peaks = np.unique(self.peaks)
        # Minima
        self.peaks_ = []
        self.peaks_, _ = signal.find_peaks(cA_, prominence=prominence_val_, width=width_val_, distance=dist_val_,
                                           threshold=th_val_)
        self.peaks_ = np.unique(self.peaks_)

        # If there is no minima or no maxima, then let the user know
        if np.size(self.peaks) == 0 or np.size(self.peaks_) == 0:
            print("No peaks detected")
            self.count += 1

        else:
            if self.peaks[0] < self.peaks_[0]:
                self.peaks_ = np.concatenate((np.array([0]), self.peaks_))  # Add a 0 at the beginning
            if self.peaks_[-1] < self.peaks[-1]:
                self.peaks_ = np.concatenate((self.peaks_, np.array([len(cA) - 1])))  # Add the last point
            # Distance between the values next to each other
            # Indices where the distance between points is smaller than a threshold --> They must be the same point!

            # Maxima
            ind_avg = np.where(distance_neighbors(self.peaks) < th_val)[0]
            while len(ind_avg) > 0:
                peaks_final = list(self.peaks)
                for i, val in enumerate(ind_avg):
                    peaks_final[i] = np.mean([peaks_final[i], peaks_final[i + 1]])

                peaks_final = [peaks_final[i] for i in range(len(peaks_final)) if i - 1 not in ind_avg]

                peaks_final = np.array(peaks_final, dtype=int)
                ind_avg = np.where(distance_neighbors(peaks_final) < th_val)[0]
                # If there are still some other values to average, restart the cycle
                self.peaks = peaks_final

            # Minima
            ind_avg_ = np.where(distance_neighbors(self.peaks_) < th_val_)[0]
            while len(ind_avg_) > 0:
                peaks_final_ = list(self.peaks_)
                for i, val in enumerate(ind_avg_):
                    peaks_final_[i] = np.mean([peaks_final_[i], peaks_final_[i + 1]])

                peaks_final_ = [peaks_final_[i] for i in range(len(peaks_final_)) if i - 1 not in ind_avg_]

                peaks_final_ = np.array(peaks_final_, dtype=int)
                ind_avg_ = np.where(distance_neighbors(peaks_final_) < th_val_)[0]
                # If there are still some other values to average, restart the cycle
                self.peaks_ = peaks_final_

            # There has to be min - max - min - max - min --> Always start and end with min
            peaks_aux = list(self.peaks_)
            for i in range(len(self.peaks) - 1):
                ind_del = np.where((self.peaks_ > self.peaks[i]) & (self.peaks_ < self.peaks[i + 1]))[0]
                if len(ind_del) > 1:
                    peaks_aux.remove(max(self.peaks_[ind_del]))

            self.peaks_ = np.array(peaks_aux)  # We had to convert to list to apply the remove funcion

            # Add the first value of the signal in case it does not start with a minimum
            if self.peaks_[0] > self.peaks[0]:
                self.peaks_ = np.concatenate((np.array([int(cA[0])]), self.peaks_))

            # Peaks analysis

            T_plus = []
            T_minus = []
            A_plus = []
            A_minus = []
            for i in range(len(self.peaks)):
                T_plus.append(self.peaks[i] - self.peaks_[i])
                T_minus.append(self.peaks_[i + 1] - self.peaks[i])
                A_plus.append(abs(cA[self.peaks[i]] - cA[self.peaks_[i]]))
                A_minus.append(abs(cA[self.peaks_[i + 1]] - cA[self.peaks[i]]))

            T_plus_final.append(np.array(T_plus))
            T_minus_final.append(np.array(T_minus))
            A_plus_final.append(np.array(A_plus))
            A_minus_final.append(np.array(A_minus))
            cycles_final.append(len(self.peaks))

            # Analysis of T+ and T-
            # 1. According to the cycle number:
            cycles_T_minus = {key: [] for key in np.arange(1, max(cycles_final) + 1)}
            cycles_T_plus = {key: [] for key in np.arange(1, max(cycles_final) + 1)}
            cycles_A_minus = {key: [] for key in np.arange(1, max(cycles_final) + 1)}
            cycles_A_plus = {key: [] for key in np.arange(1, max(cycles_final) + 1)}

            for i in range(len(T_minus_final)):
                for j, val in enumerate(T_minus_final[i]):
                    cycles_T_minus[j + 1].append(val)
                    cycles_T_plus[j + 1].append(T_plus_final[i][j])
                    cycles_A_minus[j + 1].append(A_minus_final[i][j])
                    cycles_A_plus[j + 1].append(A_plus_final[i][j])

            self.cycles_T_minus_plot = [cycles_T_minus[i + 1] for i in range(len(cycles_T_minus))]
            self.cycles_T_plus_plot = [cycles_T_plus[i + 1] for i in range(len(cycles_T_plus))]
            self.cycles_A_minus_plot = [cycles_A_minus[i + 1] for i in range(len(cycles_A_minus))]
            self.cycles_A_plus_plot = [cycles_A_plus[i + 1] for i in range(len(cycles_A_plus))]

            self.n_cycles = max(np.unique(cycles_final))


# Show all the peaks detected from the parameters chosen by the user
class peak_detection_plots:

    def __init__(self, tracks, tr_min):
        """Constructor method
        """
        super().__init__()

        self.fig = None

        # Initialize the figure size
        self.n_cols = 5
        self.n_cells = tracks.n_tracks_divs
        self.n_rows = np.ceil((self.n_cells + 1) / self.n_cols)
        self.size_fig = self.n_rows * (10 / self.n_cols)

        self.tracks = tracks
        self.tr_min = tr_min

        # Peak detection parameters
        self.threshold_val = 0  # Threshold value for maxima
        self.prominence_val = 0  # Prominence value for maxima
        self.width_val = 0  # Width value for maxima
        self.distance_val = 1  # Distance value for maxima
        self.window = 0  # Averaging window size

        # Peak detection parameters chosen by user
        # ----------------------------------------
        style = {'description_width': 'initial'}
        self.window_slider = widgets.IntSlider(value=self.window, min=0, max=100, step=1,
                                               description='Average window size', layout=dict(width='90%'), style=style)
        self.threshold_slider = widgets.FloatSlider(value=self.threshold_val, min=0, max=10, step=0.1,
                                                    description='Threshold', layout=dict(width='90%'), style=style)
        self.prominence_slider = widgets.FloatSlider(value=self.prominence_val, min=0, max=50, step=0.1,
                                                     description='Prominence', layout=dict(width='90%'), style=style)
        self.width_slider = widgets.FloatSlider(value=self.width_val, min=0, max=20, step=0.1, description='Width',
                                                layout=dict(width='90%'), style=style)
        self.distance_slider = widgets.FloatSlider(value=self.distance_val, min=1, max=20, step=0.1,
                                                   description='Distance', layout=dict(width='90%'), style=style)
        # Plot the results
        self.plot_button = widgets.Button(description='Plot', disabled=False, button_style='',
                                          tooltip='Plot the tracks', icon='')

        # Calculate the results only
        self.calculate_button = widgets.Button(description='Calculate', disabled=False, button_style='',
                                          tooltip='Calculate the peaks without plotting the traces', icon='')

        # Connect callbacks and traits
        # -----------------------------
        self.window_slider.observe(self.update_window, 'value')
        self.threshold_slider.observe(self.update_threshold, 'value')
        self.prominence_slider.observe(self.update_prominence, 'value')
        self.width_slider.observe(self.update_width, 'value')
        self.distance_slider.observe(self.update_distance, 'value')

        # Buttons plot
        buttons = widgets.HBox([self.plot_button, self.calculate_button])
        controls1 = widgets.VBox([self.window_slider, self.prominence_slider, self.width_slider, self.threshold_slider,
                                  self.distance_slider, buttons],
                                 layout={'width': '80%'})

        # Show the ipywidget
        display(controls1)

        def plot_on_click(b):
            self.plot_lines()

        def calculate_on_click(b):
            self.calculate_peaks()

        self.plot_button.on_click(plot_on_click)
        self.calculate_button.on_click(calculate_on_click)


    def update_window(self, change):
        """ Updates the value of the averaging window size."""
        self.window = change.new

    def update_prominence(self, change):
        """ Updates the value of the prominence for the peak detection."""
        self.prominence_val = change.new

    def update_threshold(self, change):
        """ Updates the value of the threshold for the peak detection."""
        self.threshold_val = change.new

    def update_distance(self, change):
        """ Updates the value of the distance for the peak detection."""
        self.distance_val = change.new

    def update_width(self, change):
        """ Updates the value of the width for the peak detection."""
        self.width_val = change.new

    def peak_detection(self, y_signal):

        # Smooth the curve
        cA = smoothing_filter(y_signal, self.window_slider.value)  # To find Maxima
        # Peak detection
        peaks, _ = signal.find_peaks(cA, prominence = self.prominence_slider.value, width = self.width_slider.value,
                                     distance = self.distance_slider.value, threshold = self.threshold_slider.value)
        peaks = np.unique(peaks)

        return peaks, cA

    #@self.plot_button.on_click
    def plot_lines(self):
        """ Plot all the tracks with the detected peaks using the parameters collected by user"""

        plot_num = 1

        # Initialize the figure size
        if self.fig is None:
            self.fig = plt.figure(figsize=[12, self.size_fig])
        else:
            self.fig.clf()

        self.peaks_all = []
        for i in range(self.n_cells):
            # Update the number of subplots
            #self.ax = plt.subplot(int(self.n_rows), self.n_cols, plot_num, aspect='auto')
            self.ax = self.fig.add_subplot(int(self.n_rows), self.n_cols, plot_num, aspect='auto')
            plot_num += 1

            peaks, cA = self.peak_detection(self.tracks.spots_features['Mean1'][i])

            # Save the peaks
            self.peaks_all.append(peaks)

            # Initial time-point where the trace starts
            init_t = self.tracks.spots_features['Frames'][i][0]

            # Plot the smoothed line and the peaks
            self.ax.plot(np.arange(init_t, len(cA) + init_t) * self.tr_min, cA, color='blue', linewidth=2)
            self.ax.plot((peaks + init_t) * self.tr_min, cA[peaks], 'xk', markersize=5)

            t_max = (len(cA) + init_t) * self.tr_min
            self.ax.set_title('Cell %d' % (i + 1), fontsize=16)
            self.ax.set_xlim([0, t_max])
            self.ax.set_ylim([min(cA), max(cA)])

        plt.tight_layout()


    def calculate_peaks(self):
        """ Calculate the peaks using the parameters collected by user"""

        self.peaks_all = []

        for i in range(self.n_cells):
            peaks, cA = self.peak_detection(self.tracks.spots_features['Mean1'][i])

            # Save the peaks
            self.peaks_all.append(peaks)

# Perform manual curation on some peaks
class manual_peak_curation:
    """ This class helps performing the manual curation of the peaks previously selected by the user."""

    def __init__(self, tracks, tr_min, curate_cells, exclude_cells, window, peaks):
        """Constructor method
        """
        super().__init__()

        self.peaks_curated = []

        for i in curate_cells:
            # Time-points and intensity
            frames = tracks.spots_features['Frames'][i]
            intensities = tracks.spots_features['Mean1'][i]

            # Smooth the curve
            cA = smoothing_filter(intensities, window)  # To find Maxima

            # Initial time-point where the trace starts
            init_t = frames[0]

            self.fig = plt.figure(figsize=[10, 9])

            # Plot the smoothed line and the peaks
            plt.plot(np.arange(init_t, len(cA) + init_t) * tr_min, cA,
                                      color='blue', linewidth=2)

            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('Time', fontsize=25)
            plt.ylabel('Intensity', fontsize=25)
            plt.title(f'Cell # {i:d}' , fontsize=30)

            points = plt.ginput(n=-1, timeout=0, mouse_add=1, mouse_pop=3, mouse_stop=2)

            # Re-arrange data to get peak times only
            peaks_aux = []
            for j in range(len(points)):
                peaks_aux.append(points[j][0])

            float_val = np.round((np.array(peaks_aux)-init_t)/tr_min)
            self.peaks_curated.append(np.array(list(map(int, float_val))))

            plt.close(self.fig)

        plt.clf()

        self.peaks = peaks

        # Change the values for the curated peaks
        for i, val in enumerate(curate_cells):
            self.peaks[val] = self.peaks_curated[i]

        # Exclude cells selected by user:
        for val in exclude_cells:
            self.peaks[val] = []

        # Cells to plot - only chose the cells with peaks, avoid the excluded cells by user
        self.peak_inds = [np.size(self.peaks[i]) for i in range(len(self.peaks))]

        # Number of tracks in total
        self.N = len([i for i, val in enumerate(self.peak_inds) if val > 0])

        # Plot the curated cells, the old cells and avoid the excluded cells
        self.n_cols = 5
        self.n_rows = np.ceil((self.N + 1) / self.n_cols)
        self.size_fig = self.n_rows * (10 / self.n_cols)
        plot_num = 1

        # Initialize the figure size
        self.fig = plt.figure(figsize=[12, self.size_fig])

        for i in self.peak_inds:
            # Update the number of subplots
            #self.ax = plt.subplot(int(self.n_rows), self.n_cols, plot_num, aspect='auto')
            self.ax = self.fig.add_subplot(int(self.n_rows), self.n_cols, plot_num, aspect='auto')
            plot_num += 1

            # Smooth signal
            cA = smoothing_filter(tracks.spots_features['Mean1'][i], window)

            # Initial time-point where the trace starts
            init_t = tracks.spots_features['Frames'][i][0]

            # Plot the smoothed line and the peaks
            self.ax.plot(np.arange(init_t, len(cA) + init_t) * tr_min, cA, color='blue', linewidth=2)
            self.ax.plot((self.peaks[i] + init_t) * tr_min, cA[self.peaks[i]], 'xk', markersize=5)

            t_max = (len(cA) + init_t) * tr_min
            self.ax.set_title('Cell %d' % (i + 1), fontsize=16)
            self.ax.set_xlim([0, t_max])
            self.ax.set_ylim([min(cA), max(cA)])

        plt.tight_layout()










