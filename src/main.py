import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def draw(filename, t, signal, fs, with_window=True):
    n_samples = signal.size

    # 应用汉宁窗
    if with_window:
        hanning_window = np.hanning(n_samples)
        signal_windowed = signal * hanning_window
    else:
        signal_windowed = signal

    # 使用 NumPy 的 fft 函数
    fft_result = np.fft.fft(signal_windowed)

    # 获取频率轴
    freqs = np.fft.fftfreq(n_samples, 1/fs)

    # 取 FFT 结果的模（幅值）
    fft_magnitude = np.abs(fft_result)

    # 由于 FFT 结果是对称的，只取前一半
    half_n = int(np.ceil(n_samples / 2.0))
    fft_magnitude = fft_magnitude[:half_n]
    freqs = freqs[:half_n]

    # 转换为 dB 格式
    max_magnitude = np.max(fft_magnitude)
    if max_magnitude == 0:
        ref = 1e-12  # 或根据场景调整
    else:
        ref = max_magnitude
    fft_db = 20 * np.log10((fft_magnitude + 1e-12)/ ref)


    os.makedirs(f"./example/{filename}", exist_ok=True)

    # 创建子图
    fig = make_subplots(rows=3, cols=1, subplot_titles=("Time Domain Signal", "Frequency Magnitude", "Frequency Spectrum in dB"))

    # 添加时域图
    fig.add_trace(go.Scatter(x=t, y=signal, mode='lines', name="Signal"), row=1, col=1)
    time_domain_data = pd.DataFrame({
        "Time (s)": t,
        "Amplitude": signal
    })
    time_domain_data.to_csv(f"./example/{filename}/time_domain.csv", index=False)

    # 添加振幅图
    fig.add_trace(go.Scatter(x=freqs, y=fft_magnitude, mode='lines', name="Frequency Magnitude"), row=2, col=1)
    time_domain_data = pd.DataFrame({
        "Frequency (Hz)": freqs,
        "Magnitude": fft_magnitude
    })
    time_domain_data.to_csv(f"./example/{filename}/frequency_magnitude.csv", index=False)

    # 添加频域图
    fig.add_trace(go.Scatter(x=freqs, y=fft_db, mode='lines', name="Frequency Spectrum"), row=3, col=1)
    time_domain_data = pd.DataFrame({
        "Frequency (Hz)": freqs,
        "Magnitude (db)": fft_db
    })
    time_domain_data.to_csv(f"./example/{filename}/frequency_spectrum.csv", index=False)

    # 设置图表标题和轴标签
    fig.update_layout(
        title="Time Domain and Frequency Domain Analysis",
        xaxis1_title="Time (s)",
        yaxis1_title="Amplitude",
        xaxis2_title="Frequency (Hz)",
        yaxis2_title="Magnitude",
        xaxis3_title="Frequency (Hz)",
        yaxis3_title="Magnitude (dB)",
        yaxis1=dict(range=[-1, 1]),  # 限制 y 轴范围
        xaxis2=dict(range=[0, fs / 2]),  # 限制 x 轴范围到 Nyquist 频率
        yaxis2=dict(range=[np.min(fft_magnitude), np.max(fft_magnitude)]),  # 限制 y 轴范围
        xaxis3=dict(range=[0, fs / 2]),  # 限制 x 轴范围到 Nyquist 频率
        yaxis3=dict(range=[np.min(fft_db), np.max(fft_db)])  # 限制 y 轴范围
    )

    # 保存为 HTML 文件
    fig.write_html(f"./example/{filename}/index.html")


def main(): 
    # 参数设置
    fs = 1000  # 采样率 (Hz)
    n_samples = 1000  # 采样点数
    f = 100  # 信号频率 (Hz)

    # 时间轴
    t = np.linspace(0, (n_samples - 1) / fs, n_samples)
    # 创建正弦波信号
    signal = np.sin(2 * np.pi * f * t)
    draw("sine_wave_1000fs_1000n_100hz", t, signal, fs)
    draw("sine_wave_1000fs_1000n_100hz_no_window", t, signal, fs, False)
    
    # 设置 [100, 200) 范围内的采样值为 0
    signal[100:200] = 0
    draw("sine_wave_1000fs_1000n_100hz_[100-200]", t, signal, fs)
    draw("sine_wave_1000fs_1000n_100hz_[100-200]_no_window", t, signal, fs, False)

    signal = np.sin(2 * np.pi * f * t)
    signal[98:198] = 0
    draw("sine_wave_1000fs_1000n_100hz_[98-198]_no_window", t, signal, fs, False)

    signal[0:1000] = 1
    draw("1_1000fs_1000n_100hz_no_window", t, signal, fs, False)

    signal[0:100] = 0
    signal[100:1000] = 1
    draw("u-100_1000fs_1000n_100hz_no_window", t, signal, fs, False)

    signal[0:500] = 0
    signal[500:1000] = 1
    draw("u-500_1000fs_1000n_100hz_no_window", t, signal, fs, False)

    signal[0:1000] = 1
    signal[100:200] = 0
    draw("1_1000fs_1000n_100hz_[100-200]_no_window", t, signal, fs, False)

    signal[0:1000] = 1
    signal[75:175] = 0
    draw("1_1000fs_1000n_100hz_[75-175]_no_window", t, signal, fs, False)

    signal[0:1000] = 1
    signal[100:300] = 0
    draw("1_1000fs_1000n_100hz_[100-300]_no_window", t, signal, fs, False)

    signal[0:1000] = 1
    signal[100:105] = 0
    draw("1_1000fs_1000n_100hz_[100-105]_no_window", t, signal, fs, False)

    signal[0:1000] = 1
    signal[500:600] = 0
    draw("1_1000fs_1000n_100hz_[500-600]_no_window", t, signal, fs, False)

    signal[0:1000] = 1
    signal[250:750] = 0
    draw("1_1000fs_1000n_100hz_[250-750]_no_window", t, signal, fs, False)

    signal[0:1000] = 1
    signal[0:100] = -1
    draw("sgn-100_1000fs_1000n_100hz_no_window", t, signal, fs, False)

    signal[0:1000] = 0
    signal[100:200] = 1
    draw("1_1000fs_1000n_100hz_[0-100,200-1000]_no_window", t, signal, fs, False)

    signal[0:1000] = 0
    signal[100:900] = 1
    draw("1_1000fs_1000n_100hz_[0-100,900-1000]_no_window", t, signal, fs, False)


    # 参数设置
    fs = 48000  # 采样率 (Hz)
    n_samples = 48000  # 采样点数
    f = 100  # 信号频率 (Hz)
    # 时间轴
    t = np.linspace(0, (n_samples - 1) / fs, n_samples)
    # 创建正弦波信号
    signal = np.sin(2 * np.pi * f * t)
    # 设置 [4800, 9600) 范围内的采样值为 0
    signal[4800:9600] = 0
    draw("sine_wave_48000fs_48000n_100hz_[4800-9600]_no_window", t, signal, fs, False)


    # 参数设置
    fs = 48000  # 采样率 (Hz)
    n_samples = 48000  # 采样点数
    f = 10000  # 信号频率 (Hz)
    # 时间轴
    t = np.linspace(0, (n_samples - 1) / fs, n_samples)
    # 创建正弦波信号
    signal = 0.01 * np.sin(2 * np.pi * f * t)
    draw("sine_wave_48000fs_48000n_10000hz_no_window", t, signal, fs, False)

if __name__ == "__main__":
    main()