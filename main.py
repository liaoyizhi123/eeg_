import time

import scipy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from gen_epoch_windows import gen_epoch_window


def compute_attention(raw_data, new_frequency_):
    """
    compute attention

    parameters:
    raw_data: 2d array, shape=(channels, times)

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
            _ = spectral_features(raw_data_resampled, new_frequency, feature_name)
            res.append([.0, .0, .0, .0] if np.isnan(_).any() else _)  # R=Beat/(Alpha +Theta)
        if np.sum(res[-1]) != 0:
            att.append(res[c][3] / (res[c][2] + res[c][1]))  # R=Beat/(Alpha +Theta)
        else:
            att.append(0)
    return (
        np.mean(att), np.max(att), np.min(att),
        [np.mean(np.stack(res)[:, 0]), np.mean(np.stack(res)[:, 1]), np.mean(np.stack(res)[:, 2]),
         np.mean(np.stack(res)[:, 3])]
    )


def spectral_features(x, Fs, feature_name='spectral_power', params_st=[]):  # Fs = 1000, feature_name='spectral_power'
    if len(params_st) == 0:
        if 'spectral' in feature_name:
            # params_st = SpectralParameters(method = 'periodogram')
            params_st = SpectralParameters(method='welch_periodogram')
    freq_bands = params_st.freq_bands

    if (len(x) < (params_st.L_window * Fs)):
        print('SPECTRAL features: signal length < window length; set shorter L_window')
        featx = np.nan
        return featx
    if feature_name == 'spectral_power':
        # ---------------------------------------------------------------------
        # use periodogram to estimate spectral power, not walch
        # ---------------------------------------------------------------------
        pxx, itotal_bandpass, f_scale, data_num_resampled, fp = gen_spectrum(x, Fs, params_st, 1)
        pxx = pxx * Fs  # shape(230401,)
        point_num_dc2nyquist = pxx.shape[0]  # 230401
        if feature_name == 'spectral_relative_power':
            pxx_total = np.sum(pxx[itotal_bandpass]) / data_num_resampled
        else:
            pxx_total = 1

        spec_pow = np.full([1, freq_bands.shape[0]], np.nan)  # shape(1, 4)

        for p in range(freq_bands.shape[0]):
            ibandpass = np.arange(np.ceil(freq_bands[p, 0] * f_scale), np.floor(freq_bands[p, 1]) * f_scale,
                                  dtype=int)  # 231-1843
            ibandpass = ibandpass + 1
            ibandpass[ibandpass < 1] = 0
            ibandpass[ibandpass > point_num_dc2nyquist] = point_num_dc2nyquist

            spec_pow[0, p] = np.sum(pxx[ibandpass]) / (data_num_resampled * pxx_total)

        return spec_pow[0]



def gen_STFT(x,L_window,window_type,overlap,Fs, STFT_OR_SPEC=0):
    """
    %-------------------------------------------------------------------------------
    % gen_STFT: Short-time Fourier transform (or spectrogram)
    %
    % Syntax: [S_stft,Nfreq,f_scale,win_epoch]=gen_STFT(x,L_window,window_type,overlap,Fs)
    %
    % Inputs:
    %     x            - input signal
    %     L_window     - window length
    %     window_type  - window type
    %     overlap      - percentage overlap
    %     Fs           - sampling frequency (Hz)
    %     STFT_OR_SPEC - return short-time Fourier transform (STFT) or spectrogram
    %                    (0=spectrogram [default] and 1=STFT)
    %
    % Outputs:
    %     S_stft     - spectrogram
    %     Nfreq      - length of FFT
    %     f_scale    - frequency scaling factor
    %     win_epoch  - window
    %
    % Example:
    %     Fs=64;
    %     data_st=gen_test_EEGdata(32,Fs,1);
    %     x=data_st.eeg_data(1,:);
    %
    %     L_window=2;
    %     window_type='hamm';
    %     overlap=80;
    %
    %     S_stft=gen_STFT(x,L_window,window_type,overlap,Fs);
    %
    %     figure(1); clf; hold all;
    %     imagesc(S_stft); axis('tight');
    %
    %-------------------------------------------------------------------------------
    """
    L_hop,L_epoch,win_epoch = gen_epoch_window(overlap, L_window, window_type, Fs, 1)
    N = len(x)
    N_epochs = int(np.ceil((N-(L_epoch-L_hop))/L_hop))

    if N_epochs < 1:
        N_epochs = 1
    nw = list(range(0,L_epoch))
    Nfreq = L_epoch

    # ---------------------------------------------------------------------
    #  generate short-time FT on all data:
    # ---------------------------------------------------------------------
    K_stft = np.zeros([N_epochs,L_epoch])
    for k in range(N_epochs):
        nf = np.mod(nw+k*L_hop,N)
        nf = nf.astype(int)

        K_stft[k,:] = x[nf] * win_epoch

    f_scale = Nfreq/Fs

    if STFT_OR_SPEC:
        S_stft = np.fft.fft(K_stft, Nfreq, axis=1)
    else:
        S_stft = np.abs(np.fft.fft(K_stft, Nfreq, axis=1))**2

    S_stft = S_stft[:, :Nfreq//2+1]

    return S_stft, Nfreq, f_scale, win_epoch


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
    interval = 4000
    for i in range(0, data_.shape[1], interval):
        start_time = time.time()
        att_avg, att_max, att_min, interested_band = compute_attention(data_[:, i:min((i + interval), data_.shape[1])],
                                                                       64)
        # att_avg, att_max, att_min = compute_attention(data_[:, i:min((i + interval), data_.shape[1])])
        end_time = time.time()
        print(
            f'Time: {end_time - start_time:5.4f}, '
            f'avg_org: {att_avg:5.3f}, '
            f'avg_normalized: {normalize(att_avg, 0.0, 1.0, 0, 100):5.3f}, '
            f'max: {att_max:5.3f}, min: {att_min:5.3f}, '

            f'delta: {interested_band[0]:5.3f}, '
            f'theta: {interested_band[1]:5.3f}, '
            f'alpha: {interested_band[2]:5.3f}, '
            f'beta: {interested_band[3]:5.3f}, '

            f'data: {(i, min((i + interval), data_.shape[1]))}'
        )
    exit()

