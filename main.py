import math
import time

import scipy
import pandas as pd
from  matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import welch


from gen_epoch_windows import gen_STFT
from AttentionAlgorithms import AttentionAlgorithms
from util.data_processing import filter_with_PMD, exclude_outliers

# ---------------------------------------- #

# ---------------------------------------- #
def compute_attention(raw_data_, original_sampling_rate, resampling_rate, interval, algorithm):
    """
    compute attention

    """
    raw_data = raw_data_[:, :, 1]
    # raw_data = raw_data[:, :, 1] * 1e6  # convert to microvolts

    channels = raw_data.shape[0]
    data_len = raw_data.shape[1]
    new_frequency = resampling_rate  # ##
    original_frequency = 500
    resampling_factor = (new_frequency*interval) / (original_frequency*interval)

    # spectral_features_list = ['spectral_power']
    spectral_features_list = ['spectral_power_by_scipy']

    res = []
    att = []

    new_ = True  # 1-(0+2)/1

    for c in range(channels):
        # filter data
        raw_data[c] = filter_with_PMD(data=raw_data[c], Fs = original_sampling_rate, wn=6, lp=60)

        # resampling data
        raw_data_resampled = scipy.signal.resample(raw_data[c], int(data_len * resampling_factor)) if resampling_factor != 1 else raw_data[c]
        # resampled_data = resampled_data - np.mean(resampled_data)  # remove DC offset

        for feature_name in spectral_features_list:  # ['spectral_power_by_scipy']
            # compute sum of spectral power with respect to frequency bands
            # delta(0.5-4), theta(4-7), alpha(7-13), beta(13-30)
            # 0             1           2           3
            _ = spectral_features(raw_data_resampled, new_frequency, feature_name)
            res.append([.0, .0, .0, .0] if np.isnan(_).any() else _)
        if np.sum(res[-1]) != 0:

            if algorithm == AttentionAlgorithms.Ec:
                att.append(AttentionAlgorithms.compute_ec(res[c]))  # R=Beat/(Alpha +Theta)
            elif algorithm == AttentionAlgorithms.Proportion_of_Beta:
                att.append(AttentionAlgorithms.compute_proportion_of_beta(res[c]))  # R=Beat/(Alpha +Total)
            elif algorithm == AttentionAlgorithms.Beta_over_Alpha:
                att.append(AttentionAlgorithms.compute_beta_over_alpha(res[c]))
            elif algorithm == AttentionAlgorithms.XY_RATIO:
                att.append(AttentionAlgorithms.compute_xy_ratio(res[c]))
            else: # default
                att.append(0)
        else:
            att.append(0)
    if new_:  # att[1]-(att[0]+att[2])/2
        # return (
        #     att[0], np.max(att), np.min(att),
        #     [np.mean(np.stack(res)[:, 0]), np.mean(np.stack(res)[:, 1]), np.mean(np.stack(res)[:, 2]),
        #      np.mean(np.stack(res)[:, 3])]
        # )
        att = [_ for _ in att if _ != 0]
        res = [_ for _ in res if np.sum(_) != 0]
        return (
            np.mean(att),np.array(res)
        )
    return (
        np.mean(att), np.max(att), np.min(att),
        [np.mean(np.stack(res)[:, 0]), np.mean(np.stack(res)[:, 1]), np.mean(np.stack(res)[:, 2]),
         np.mean(np.stack(res)[:, 3])]
    )


def spectral_features(x, new_frequency, feature_name='spectral_power', params_st=[]):  # Fs = 1000, feature_name='spectral_power'
    if len(params_st) == 0:
        if 'spectral' in feature_name:
            params_st = SpectralParameters(method = 'periodogram')
            # params_st = SpectralParameters(method='welch_periodogram')
    freq_bands = params_st.freq_bands

    if len(x) < (params_st.L_window * new_frequency):
        print('SPECTRAL features: signal length < window length; set shorter L_window')
        featx = np.nan
        return featx
    if feature_name == 'spectral_power':
        # ---------------------------------------------------------------------
        # apply spectral analysis with periodogram or welch periodogram
        # ---------------------------------------------------------------------
        pxx, itotal_bandpass, f_scale, fft_length, fp = gen_spectrum(x, new_frequency, params_st, 1)
        pxx = pxx * new_frequency  # shape(230401,)
        point_num_dc2nyquist = pxx.shape[0]  # 230401
        if feature_name == 'spectral_relative_power':
            pxx_total = np.sum(pxx[itotal_bandpass]) / fft_length
        else:
            pxx_total = 1

        # spec_pow = np.full([1, freq_bands.shape[0]], np.nan)  # shape(1, 4)

        res_li = []
        for p in range(freq_bands.shape[0]):
            ibandpass = np.arange(np.ceil(freq_bands[p, 0] * f_scale), np.floor(freq_bands[p, 1]) * f_scale,
                                  dtype=int)  # 231-1843
            ibandpass = ibandpass + 1  # ## ? 这里似乎应该是ibandpass = np.append(ibandpass, ibandpass[-1]+1)
            ibandpass[ibandpass < 1] = 0
            ibandpass[ibandpass > point_num_dc2nyquist] = point_num_dc2nyquist

            # spec_pow[0, p] = np.sum(pxx[ibandpass]) / (fft_length * pxx_total)
        # return spec_pow[0]

            res_li.append(np.sum(pxx[ibandpass]) / (fft_length * pxx_total))
        return np.array(res_li)


    elif feature_name == 'spectral_power_by_scipy':
        frequencies, psd = welch(x, new_frequency, nperseg=new_frequency*2, noverlap=new_frequency//2)
        delta_power = calculate_band_power(frequencies, psd, params_st.freq_bands[0])
        theta_power = calculate_band_power(frequencies, psd, params_st.freq_bands[1])
        alpha_power = calculate_band_power(frequencies, psd, params_st.freq_bands[2])
        beta_power = calculate_band_power(frequencies, psd, params_st.freq_bands[3])
        return np.array([delta_power, theta_power, alpha_power, beta_power])

def calculate_band_power(f, Pxx_den, band):
    indices = np.logical_and(f >= band[0], f <= band[1])
    band_power = np.trapz(Pxx_den[indices], f[indices])  # integral
    return band_power


def gen_spectrum(x, Fs, params_st, SCALE_PSD=0):
    """
    变换成频谱
    """

    # remove NaNs
    x[np.isnan(x)] = []

    if params_st.method.lower() == 'periodogram':
        # ---------------------------------------------------------------------
        # Periodogram
        # frequencies, psd = scipy.signal.periodogram(data)
        # ---------------------------------------------------------------------
        X = np.abs(np.fft.fft(x)) ** 2

        # positive frequencies only:
        N = len(X)
        Nh = int(np.floor(N / 2))
        Nfreq = N
        X = X[:Nh + 1]  # including DC and Nyquist frequencies
        pxx = X / (Fs * N)  # normalize by Fs and N, where Fs is the new frequency and N is the number of new data points
        frequencies = np.fft.fftfreq(N, 1 / Fs)[:Nh + 1]


        # ##
        # X_ = (np.abs(np.fft.fft(x)) ** 2)[Nh + 1 :]
        # nxx = X_ / (Fs * N)
        #
        # freqs = np.fft.fftfreq(N, 1 / Fs)[:Nh + 1]
        # freqs_n = np.fft.fftfreq(N, 1 / Fs)[Nh + 1:]
        # import matplotlib.pyplot as plt
        # # 绘制功率谱密度
        # plt.figure(figsize=(8, 6))
        # plt.plot(freqs, pxx)
        # plt.plot(freqs_n, nxx)
        # plt.title('Power Spectral Density (Periodogram)')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Power Spectral Density')
        # plt.grid(True)
        # plt.show()

    elif params_st.method.lower() == 'welch_periodogram':
        # ---------------------------------------------------------------------
        # Welch periodogram
        # frequencies, psd = scipy.signal.welch(data)
        # ---------------------------------------------------------------------
        S_stft, Nfreq, f_scale, win_epoch = gen_STFT(x, params_st.L_window, params_st.window_type, params_st.overlap, Fs)

        # average over time:
        pxx = np.nanmean(S_stft, 0)

        N = len(pxx)
        # normalise (so similar to pwelch):
        E_win= np.sum(np.abs(win_epoch)**2) / Nfreq
        pxx=(pxx/(Nfreq*E_win*Fs))

    else:
        print(f'unknown spectral method "{params_st.method}"; check spelling\n')
        pxx = np.nan
        itotal_bandpass = np.nan
        f_scale = np.nan
        fp = np.nan

    # in order to conserve the total power of both positive and negative frequencies
    if SCALE_PSD:
        pscale = np.ones([1, len(pxx)]) + 1
        if Nfreq % 2:
            pscale[0, 0] = 1
        else:
            pscale[0, 0], pscale[0, -1] = 1, 1

        pxx = pxx * pscale
        pxx = pxx[0]

    N = pxx.shape[0]
    f_scale = Nfreq / Fs  # points/sampling

    # for plotting only:
    fp = np.arange(N) / f_scale

    if hasattr(params_st, 'total_freq_bands'):
        total_freq_bands = params_st.total_freq_bands
        # b) limit to frequency band of interest:
        # print('test total freq bands', total_freq_bands[1])
        total_freq_bands_low = np.ceil(total_freq_bands[0] * f_scale)
        total_freq_bands_high = np.floor(total_freq_bands[1] * f_scale)
        itotal_bandpass = np.arange(total_freq_bands_low + 1, total_freq_bands_high + 2, dtype=int)

        itotal_bandpass[itotal_bandpass < 1] = 0
        itotal_bandpass[itotal_bandpass > N] = N

    else:
        itotal_bandpass = np.nan

    return pxx, itotal_bandpass, f_scale, Nfreq, fp


class SpectralParameters:
    def __init__(self, method='PSD', L_window=2, window_type='hamm', overlap=50,
                 freq_bands=np.array([[0.5, 4], [4, 7], [7, 13], [13, 30]]),
                 total_freq_bands=None, SEF=0.95):
        # 初始化与频谱分析相关的参数
        # how to estimate the spectrum for 'spectral_flatness', 'spectral_entropy',
        # spectral_edge_frequency features:
        # 1) PSD: estimate power spectral density (e.g. Welch periodogram)
        # 2) robust-PSD: median (instead of mean) of spectrogram
        # 3) periodogram: magnitude of the discrete Fourier transform

        self.method = method

        # length of time - domain analysis window and overlap:
        # (applies to 'spectral_power', 'spectral_relative_power',
        # 'spectral_flatness', and 'spectral_diff' features)
        self.L_window = L_window  # in seconds
        self.window_type = window_type  # type of window
        self.overlap = overlap  # overlap in percentage
        self.freq_bands = freq_bands
        if total_freq_bands is None:
            total_freq_bands = [0.5, 30]
        self.total_freq_bands = total_freq_bands
        self.SEF = SEF  # spectral edge frequency


def normalize(value, min_old=0.0, max_old=2.0, min_new=0, max_new=100):
    value_clipped = np.clip(value, min_old, max_old)
    normalized_value = (value_clipped - min_old) / (max_old - min_old) * (max_new - min_new) + min_new
    return normalized_value




# if __name__ == '__main__':
#
#     data_dir = Path('./data')
#     data_path = data_dir/'chb1.mat'
#     data = scipy.io.loadmat(str(data_path))
#
#     interictal_data = data['Interictal_data']  # 14x23x460800 double
#     # preictal_data = data['Preictal_data']  # 5x23x460800 double
#
#     # 23x460800 double
#     data_ = interictal_data[0]
#     _ = []
#     interval = 4000
#     for i in range(0, data_.shape[1], interval):
#         start_time = time.time()
#
#         # AttentionAlgorithms.Ec or AttentionAlgorithms.XY_RATIO
#         att_avg, att_max, att_min, interested_band = compute_attention(data_[:, i:min((i + interval), data_.shape[1])],
#                                                                        64, AttentionAlgorithms.XY_RATIO)
#
#         end_time = time.time()
#         _.append(att_avg)
#         score_norm = normalize(att_avg, -0.2, 0.7, 0, 100)
#         print(
#             f'Time: {end_time - start_time:5.4f}, '
#             f'avg_org attention score: {att_avg:6.3f}, '
#
#             # f'avg_normalized: {normalize(att_avg, 0.0, 1.0, 0, 100):6.3f}, '  # rescale for Ec
#
#             f'avg_normalized attention score: {score_norm:6.3f}, '  # rescale for x y ratio
#             # f'max: {att_max:6.3f}, min: {att_min:6.3f}, '
#
#             f'relaxation score: {100-score_norm:6.3f}, '
#
#             f'delta-> {interested_band[0]:9.4f}, '
#             f'theta-> {interested_band[1]:9.4f}, '
#             f'alpha-> {interested_band[2]:9.4f}, '
#             f'beta-> {interested_band[3]:9.4f}, '
#
#             f'testing data frame: {(i, min((i + interval), data_.shape[1]))}'
#         )
#
#     print(f'avg: {np.mean(_):5.3f}, max: {np.max(_):5.3f}, min: {np.min(_):5.3f}')
#
#     exit()

def plot_score_matplotlib(data):
    import matplotlib.pyplot as plt
    import numpy as np

    vector = data

    indices = np.arange(len(vector))

    plt.plot(indices, vector, marker='o')

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)

    plt.show()

def plot_score(data):

    import plotly.graph_objects as go

    vector = data
    indices = list(range(len(vector)))

    fig = go.Figure(data=[go.Scatter(x=indices, y=vector, mode='markers+lines', marker=dict(size=5))])


    # marker_li = [24.8540796, 45.238474200000006, 69.81087579999999, 92.10353370000001, 121.1371983, 141.5345128, 166.4873682,
    #         189.1481633, 213.264333, 235.9293005, 258.3703464, 278.4493198, 301.5013298, 324.3326149, 347.1754775,
    #         368.82531109999996, 389.018268, 409.2409788, 435.98605729999997, 457.4862642]
    #
    # for marker in marker_li:
    #     fig.add_shape(
    #         type="line",
    #         x0=marker, y0=min(vector), x1=marker, y1=max(vector),
    #         line=dict(color="Red", dash="dash")
    #     )

    fig.update_layout(
        title='Interactive Plot',
        xaxis_title='Index',
        yaxis_title='Value',
        xaxis=dict(range=[indices[0], indices[-1]]),
        yaxis=dict(range=[min(vector), max(vector)])
    )

    fig.show()


def read_marker_add_rest_marker(file_path):
    res_li = []
    # read marker
    with open(file_path, 'r') as file:
        line_count = 0
        for line in file:
            line_count += 1
            if line_count == 5:
                res_li = line.strip().split()
                break
    marker_li = []
    for mark in res_li:
        mark_ = float(mark)
        marker_li.append(mark_)
        marker_li.append(mark_+10.0)
    return marker_li


def read_data(dir_, file_, eeg_channel_num_):
    res_li = []
    # compute warm up time
    warm_up_time_ = 10000.0
    for channel_num in range(eeg_channel_num_):
        _ = pd.read_csv(dir_ / f'{file_}{channel_num}.csv').iloc[0, 0]
        if warm_up_time_ > _:
            warm_up_time_ = _
    warm_up_time_ = math.ceil(warm_up_time_)

    for channel_num in range(eeg_channel_num_):
        df_ = pd.read_csv(dir_ / f'{file_}{channel_num}.csv')
        val_ = df_[df_.iloc[:, 0] > warm_up_time_].values
        val_[:, 0] = val_[:, 0] - warm_up_time_
        res_li.append(val_)
    return res_li


def plot_bands(interested_bands):
    bands_arr = np.stack(interested_bands, axis=2)
    plt.figure(figsize=(20, 10))
    num_channels = bands_arr.shape[0]  # 通道数
    num_bands = bands_arr.shape[1]  # 频带数
    total_plots = num_channels * num_bands
    for channel in range(num_channels):
        for band in range(num_bands):
            plot_index = band * num_bands + channel + 1
            plt.subplot(4, 4, plot_index)
            plt.plot(bands_arr[channel, band, :])
            plt.title(f'Channel {channel + 1}, Band {band + 1}')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    file_name = '24-10-10-21-35'
    data_dir = f'./data/fNIRS-EEG/{file_name}'
    file_prefix = 'EEG_000001_0'
    eeg_channel_num = 4

    data_dir_ = Path(data_dir)

    #################
    #### setting ####
    original_sampling_rate = 500
    resampling_rate = 64
    interval = 4
    algorithm_ = AttentionAlgorithms.Ec
    #### setting ####
    #################

    data = read_data(data_dir_, file_prefix, eeg_channel_num)

    # marker_file_name = f'MentalCalculation_mark-2.txt'
    # marker_li_ = read_marker_add_rest_marker(f'{data_dir}/{marker_file_name}')

# pcf 频差法
    _normalize = []
    task_rest_avg_li = []
    __ = []
    count=0

    data_ = [[] for _ in range(eeg_channel_num)]
    final_time = 1e06
    for i in data:
        if i[-1, 0] < final_time:
            final_time = i[-1, 0]

    interested_bands = []
    attention_li = []
    for i in range(interval, math.ceil(final_time)+1 ):
        start_time = time.time()

        min_len_ = 1000*interval
        for channel in range(len(data)):
            data_[channel] = data[channel][(data[channel][:, 0] < i) & (data[channel][:, 0] > (i - interval))]
            if len(data_[channel]) < min_len_: min_len_ = len(data_[channel])
            # data_[j] = data[j][:, 1].reshape(1, -1)
        for channel in range(len(data)):
            data_[channel] = data_[channel][:min_len_]

        # AttentionAlgorithms.Ec or AttentionAlgorithms.XY_RATIO
        # att_avg, att_max, att_min, interested_band = compute_attention(np.stack(data_), new_freq, interval, algorithm_)
        data_arr = np.stack(data_)
        att_score, bands_integral_c_x_bands_per_second = compute_attention(data_arr, original_sampling_rate, resampling_rate, interval, algorithm_)
        attention_li.append(att_score)
        interested_bands.append(bands_integral_c_x_bands_per_second)

        end_time = time.time()

        # if i< marker_li_[count]:
        #     __.append(att_avg)
        # else:
        #     task_rest_avg_li.append(np.mean(__))
        #     __ = []
        #     __.append(att_avg)
        #     count += 1

        # score_norm = normalize(att_avg, -2.0, 4.0, 0, 100)
        # _normalize.append(score_norm)
        print(
            f'Time: {end_time - start_time:5.4f}, '
            f'avg_org attention score: {att_score:6.3f}, '

            # f'avg_normalized: {normalize(att_avg, 0.0, 1.0, 0, 100):6.3f}, '  # rescale for Ec

            # f'avg_normalized attention score: {score_norm:6.3f}, '  # rescale for x y ratio
            # f'max: {att_max:6.3f}, min: {att_min:6.3f}, '

            # f'relaxation score: {100 - score_norm:6.3f}, '

            # f'delta-> {interested_band[0]:9.4f}, '
            # f'theta-> {interested_band[1]:9.4f}, '
            # f'alpha-> {interested_band[2]:9.4f}, '
            # f'beta-> {interested_band[3]:9.4f}, '

            # f'testing data frame: {(i, min((i + interval), data_.shape[1]))}'
        )

    # plot_bands(interested_bands)

    attention_li = exclude_outliers(attention_li)
    print(f'avg: {np.mean(attention_li):5.3f}, max: {np.max(attention_li):5.3f}, min: {np.min(attention_li):5.3f}')

    # plot_score(_)
    # plot_score(_normalize)
    # flag = False
    # for i in task_rest_avg_li:
    #     if flag:
    #         print(f'task: {i:5.3f}', end=' ')
    #         flag = False
    #     else:
    #         print(f'rest: {i:5.3f}')
    #         flag = True



