import numpy as np
from scipy.signal import butter, filtfilt
from PyEMD import EMD, Visualisation  # 可视化

def exclude_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [_ for _ in data if lower_bound < _ < upper_bound]

def filter_with_PMD(data, Fs, wn=6, lp=70, hp=0.04, sta=5, ):

    length = data.shape[0]

    # b, a = iirnotch(50, 30, 500)
    # lpdata = filtfilt(b, a, data)

    # [b,a] = signal.butter(wn, [hp,lp], btype='band', analog=False, output='ba', fs=Fs)
    # bpdata = signal.filtfilt(b, a, data, axis=0)

    [b1, a1] = butter(wn, lp, btype='lowpass', analog=False, output='ba', fs=Fs)
    # [b2, a2] = signal.butter(wn, hp, btype='highpass', analog=False, output='ba', fs=Fs)
    lpdata = filtfilt(b1, a1, data, axis=0)
    # bpdata = signal.filtfilt(b2, a2, lpdata, axis=0)

    res = np.copy(lpdata)

    ##载入时间序列数据
    S = lpdata
    t = np.arange(0, len(S), 1)  # t 表示横轴的取值范围
    # Extract imfs and residue
    # In case of EMD
    emd = EMD()
    emd.emd(S)

    # 获得分量+残余分量
    imfs, ress = emd.get_imfs_and_residue()
    # 分量的个数
    # print(len(imfs))
    # vis = Visualisation()
    # 分量可视化
    # vis.plot_imfs(imfs=imfs, residue=ress, t=t , include_residue=True)
    # vis.show()

    res = res - ress
    res = res - np.mean(res[:100], axis=0)

    return res


