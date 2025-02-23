import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def draw(filename, t, signal, fs):
    n_samples = signal.size

    # 应用汉宁窗
    hanning_window = np.hanning(n_samples)
    signal_windowed = signal * hanning_window

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
    fft_db = 20 * np.log10(fft_magnitude / np.max(fft_magnitude))

    # 创建子图
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Time Domain Signal", "Frequency Spectrum in dB"))

    # 添加时域图
    fig.add_trace(go.Scatter(x=t, y=signal, mode='lines', name="Signal"), row=1, col=1)

    # 添加频域图
    fig.add_trace(go.Scatter(x=freqs, y=fft_db, mode='lines', name="Frequency Spectrum"), row=2, col=1)

    # 设置图表标题和轴标签
    fig.update_layout(
        title="Time Domain and Frequency Domain Analysis",
        xaxis1_title="Time (s)",
        yaxis1_title="Amplitude",
        xaxis2_title="Frequency (Hz)",
        yaxis2_title="Magnitude (dB)",
        xaxis2=dict(range=[0, fs / 2]),  # 限制 x 轴范围到 Nyquist 频率
        yaxis2=dict(range=[np.min(fft_db), 0])  # 限制 y 轴范围
    )

    # 保存为 HTML 文件
    fig.write_html(f"{filename}.html")


def main(): 
    # 参数设置
    fs = 1024  # 采样率 (Hz)
    n_samples = 1024  # 采样点数
    f = 100  # 信号频率 (Hz)

    # 时间轴
    t = np.linspace(0, (n_samples - 1) / fs, n_samples)

    # 创建正弦波信号
    signal = np.sin(2 * np.pi * f * t)

    draw("sine_wave_1024fs_1024n_100hz", t, signal, fs)
    
    # 设置 [100, 200) 范围内的采样值为 0
    signal[100:200] = 0

    draw("sine_wave_1024fs_1024n_100hz_[100-200]", t, signal, fs)

if __name__ == "__main__":
    main()