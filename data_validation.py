import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, freqz, iirnotch
import matplotlib.pyplot as plt

from main import read_data
from util.data_processing import filter_with_PMD

def get_data():
    frequency = 50  # 50 Hz
    duration = 2  # 20 seconds
    sampling_rate = 500  # 1000 samples per second for smooth signal
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    # Generate sine wave
    sine_wave = np.sin(2 * np.pi * frequency * t)  # 50
    sine_wave_2 = np.sin(2 * np.pi * frequency*2 * t) # 100

    return sine_wave, sine_wave_2, t



if __name__ == '__main__':

    file_name = '24-10-10-21-35'
    data_dir = f'./data/fNIRS-EEG/{file_name}'
    file_prefix = 'EEG_000001_0'

    data_dir_ = Path(data_dir)

    #################
    #### setting ####
    eeg_channel_num = 4
    interval = 4
    sampling_rate = 500  # 采样频率
    nyquist = 0.5 * sampling_rate
    #### setting ####
    #################

    data = read_data(data_dir_, file_prefix, eeg_channel_num)

    # ##1
    x = data[0][:, 1]
    # ##2
    # x_1, x_2, t = get_data()
    # x = x_1 + x_2

    filtered_x = filter_with_PMD(data=x, lp=40)

    # # high-pass filter parameters
    # low_cut = 10
    # low = low_cut / nyquist
    # # low-pass filter parameters
    # high_cut = 32
    # high = high_cut / nyquist
    #
    # # ###################1
    # b, a = butter(4, low, btype="highpass")
    # filtered_x = filtfilt(b, a, x)
    # b, a = butter(4, high, btype="lowpass")
    # filtered_x = filtfilt(b, a, filtered_x)


    # ###################2
    # b,a = butter(4, [low, high], btype='bandpass')
    # filtered_x = filtfilt(b, a, x)

    # 计算原始数据和滤波后数据的频谱
    x_fft = np.fft.fft(x)
    filtered_x_fft = np.fft.fft(filtered_x)
    frequencies = np.fft.fftfreq(len(x), d=1 / sampling_rate)

    # # 只取正频率部分
    frequencies = frequencies[:len(x) // 2 ]
    x_fft = x_fft[:len(x) // 2]
    filtered_x_fft = filtered_x_fft[:len(x) // 2]

    plt.figure(figsize=(16, 8))

    # 绘制原始数据
    plt.subplot(221)
    plt.plot(x, label='Original Data', alpha=0.5, color='blue')
    # plt.ylim((-300, 300))
    plt.title('Original Data')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    # 绘制滤波后的数据
    plt.subplot(222)
    plt.plot(filtered_x, label='Filtered Data', alpha=0.5, color='red')
    plt.title('Filtered Data')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    # 绘制原始数据的频谱
    plt.subplot(223)
    plt.plot(frequencies[100:], np.abs(x_fft)[100:], label='Original Data Spectrum')
    plt.title('Original Data Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    plt.xlim((0, sampling_rate / 2))  # 只显示正频率部分
    plt.legend()

    plt.subplot(224)
    plt.plot(frequencies, np.abs(filtered_x_fft), label='Filtered Data Spectrum', linewidth=2)
    plt.title('Filtered Data Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim((0, sampling_rate / 2))  # 只显示正频率部分
    plt.ylim((0,100000))
    plt.legend()

    plt.tight_layout()
    plt.show()

