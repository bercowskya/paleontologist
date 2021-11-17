import numpy as np
from scipy import signal, spatial

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
#
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