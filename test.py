import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz

# 获取数据
def get_data():
    frequency = 50  # 50 Hz
    duration = 2  # 2 seconds
    sampling_rate = 500  # 500 samples per second
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    # Generate sine wave
    sine_wave = np.sin(2 * np.pi * frequency * t)  # 50 Hz
    sine_wave_2 = np.sin(2 * np.pi * frequency * 2 * t)  # 100 Hz

    return sine_wave, sine_wave_2, t

# 带通滤波器设计
def bandpass_filter(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# 应用滤波器
def apply_filter(data, lowcut, highcut, fs):
    b, a = bandpass_filter(lowcut, highcut, fs)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# 绘制频谱
def plot_spectrum(data, fs):
    freqs = np.fft.fftfreq(len(data), 1/fs)
    fft_spectrum = np.fft.fft(data)
    plt.plot(freqs[:len(freqs)//2], np.abs(fft_spectrum[:len(freqs)//2]))
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.show()

# 主程序
sine_wave, sine_wave_2, t = get_data()
sampling_rate = 500  # 采样率
combined_signal = sine_wave + sine_wave_2  # 组合信号 (50Hz + 100Hz)

# 滤波参数
lowcut = 20  # 带通下限
highcut = 40  # 带通上限

# 对信号应用滤波器
filtered_signal = apply_filter(combined_signal, lowcut, highcut, sampling_rate)

# 绘制原始信号和滤波后的信号
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, combined_signal)
plt.title("Original Signal (50Hz + 100Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(t, filtered_signal)
plt.title("Filtered Signal (20-50 Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# 绘制滤波后的频谱图
plot_spectrum(filtered_signal, sampling_rate)