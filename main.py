import time

import scipy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from gen_epoch_windows import gen_STFT
from AttentionAlgorithms import AttentionAlgorithms
# ---------------------------------------- #

# ---------------------------------------- #
def compute_attention(raw_data, new_frequency_, algorithm):
    """
    compute attention

    """
    raw_data = raw_data * 1e6  # convert to microvolts

    channels = raw_data.shape[0]
    data_len = raw_data.shape[1]
    new_frequency = new_frequency_  # ##
    original_frequency = 1000
    resampling_factor = new_frequency / original_frequency
    spectral_features_list = ['spectral_power']

    res = []
    att = []

    for c in range(channels):
        raw_data_resampled = scipy.signal.resample(raw_data[c], int(data_len * resampling_factor)) if resampling_factor != 1 else raw_data[c]
        # resampled_data = resampled_data - np.mean(resampled_data)  # remove DC offset
        for feature_name in spectral_features_list:  # ['spectral_power']
            # compute sum of spectral power with respect to frequency bands
            # delta(0.5-4), theta(4-7), alpha(7-13), beta(13-30)
            # 0             1           2           3
            _ = spectral_features(raw_data_resampled, new_frequency, feature_name)
            res.append([.0, .0, .0, .0] if np.isnan(_).any() else _)  # R=Beat/(Alpha +Theta)
        if np.sum(res[-1]) != 0:

            if algorithm == AttentionAlgorithms.Ec:
                att.append(AttentionAlgorithms.compute_ec(res[c]))  # R=Beat/(Alpha +Theta)
            elif algorithm == AttentionAlgorithms.XY_RATIO:
                att.append(AttentionAlgorithms.compute_xy_ratio(res[c]))
            else: # default
                att.append(0)
        else:
            att.append(0)
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

        spec_pow = np.full([1, freq_bands.shape[0]], np.nan)  # shape(1, 4)

        for p in range(freq_bands.shape[0]):
            ibandpass = np.arange(np.ceil(freq_bands[p, 0] * f_scale), np.floor(freq_bands[p, 1]) * f_scale,
                                  dtype=int)  # 231-1843
            ibandpass = ibandpass + 1
            ibandpass[ibandpass < 1] = 0
            ibandpass[ibandpass > point_num_dc2nyquist] = point_num_dc2nyquist

            spec_pow[0, p] = np.sum(pxx[ibandpass]) / (fft_length * pxx_total)

        return spec_pow[0]


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
    f_scale = Nfreq / Fs  # 460800/1000

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




if __name__ == '__main__':


    data = scipy.io.loadmat('./data/chb1.mat')

    interictal_data = data['Interictal_data']  # 14x23x460800 double
    # preictal_data = data['Preictal_data']  # 5x23x460800 double

    # 23x460800 double
    data_ = interictal_data[0]
    _ = []
    interval = 4000
    for i in range(0, data_.shape[1], interval):
        start_time = time.time()
        att_avg, att_max, att_min, interested_band = compute_attention(data_[:, i:min((i + interval), data_.shape[1])],
                                                                       64, AttentionAlgorithms.XY_RATIO)
        # att_avg, att_max, att_min = compute_attention(data_[:, i:min((i + interval), data_.shape[1])])
        end_time = time.time()
        _.append(att_avg)
        score_norm = normalize(att_avg, -0.2, 0.7, 0, 100)
        print(
            f'Time: {end_time - start_time:5.4f}, '
            f'avg_org attention score: {att_avg:6.3f}, '
            
            # f'avg_normalized: {normalize(att_avg, 0.0, 1.0, 0, 100):6.3f}, '  # rescale for Ec
            
            f'avg_normalized attention score: {score_norm:6.3f}, '  # rescale for x y ratio
            f'max: {att_max:6.3f}, min: {att_min:6.3f}, '
            
            f'relaxation score: {100-score_norm:6.3f}, '
            
            f'delta-> {interested_band[0]:9.4f}, '
            f'theta-> {interested_band[1]:9.4f}, '
            f'alpha-> {interested_band[2]:9.4f}, '
            f'beta-> {interested_band[3]:9.4f}, '
            
            f'testing data frame: {(i, min((i + interval), data_.shape[1]))}'
        )

    print(f'avg: {np.mean(_):5.3f}, max: {np.max(_):5.3f}, min: {np.min(_):5.3f}')

    exit()

