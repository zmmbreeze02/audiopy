import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wave


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
    half_n = int(np.ceil(n_samples / 2.0)) + 1
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
        yaxis1=dict(range=[-1.25, 1.25]),  # 限制 y 轴范围
        xaxis2=dict(range=[0, fs / 2]),  # 限制 x 轴范围到 Nyquist 频率
        yaxis2=dict(range=[np.min(fft_magnitude), np.max(fft_magnitude)]),  # 限制 y 轴范围
        xaxis3=dict(range=[0, fs / 2]),  # 限制 x 轴范围到 Nyquist 频率
        yaxis3=dict(range=[np.min(fft_db), np.max(fft_db)])  # 限制 y 轴范围
    )

    # 保存为 HTML 文件
    fig.write_html(f"./example/{filename}/index.html")

    return fft_result


def compute_magnitude_rate(n_samples, fft_result, number=1000, rect_frequency=10, sin_frequency=1000, mask_range=300):
    half_n = int(np.ceil(n_samples / 2.0)) + 1
    fft_result = fft_result[:half_n]
    fft_magnitude = np.abs(fft_result)

    right_range = range(
        np.max([sin_frequency + mask_range, sin_frequency + 1]),
        number * rect_frequency + sin_frequency + 1,
        rect_frequency
    )
    harmonic_wave_magnitude = np.sum([fft_magnitude[i] for i in right_range])

    left_range = range(0, np.min([sin_frequency - mask_range + 1, sin_frequency]), rect_frequency)
    harmonic_wave_magnitude += np.sum([fft_magnitude[i] for i in left_range])

    return harmonic_wave_magnitude / np.sum(fft_magnitude)


# 创造矩形波，通过傅立叶级数公式实现
# 三角函数版本
# Rect(t) = aA + 2aA\sum_{n=1}^{\infty} \text{sinc}(an) \cos(2\pi n f t)
# signal = 0.25 * 1
# for n in range(1, 101):
#     signal += 1 * 2 * 0.25 * np.sinc(0.25 * n) * np.cos(2 * np.pi * n * 10 * t)
# 指数函数版本
# Rect(t) = aA + 2aA\sum_{n=1}^{\infty} \text{sinc}(an) e^{2\pi n f t}
#
# 参数：
# amplitude: 振幅
# duty_rate: 占空比
# frequency: 频率
# t: 时间轴
# number: 级数的数量
# time_offset: 时间偏移量
def create_rect_wave(amplitude, duty_rate, frequency, t, number, time_offset=0):
    signal = duty_rate * amplitude
    for n in range(1, number+1):
        signal += amplitude * 2 * duty_rate * np.real(np.sinc(duty_rate * n) * np.exp(1j * 2 * np.pi * n * frequency * (t - time_offset)))
    return signal

def save_wav_file(filename, signal, fs):
    signal = np.int16(signal * 32767)
    # 创建 WAV 文件
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16 位采样宽度
        wav_file.setframerate(fs)  # 采样率
        wav_file.writeframes(signal.tobytes())


def create_real_rect_wave_demo(filename, number, t, n_samples, fs):
    signal = create_rect_wave(1, 0.9, 1, t, number, -0.45+0.1) * np.sin(2 * np.pi * 100 * t)
    save_wav_file(f"./example/{filename}.wav", signal, fs)
    fft_result = draw(filename, t, signal, fs, False)
    r = compute_magnitude_rate(n_samples, fft_result, number, 1, 100, 0)
    print(f"{filename}: ", r)


def create_real_rect_wave_demo2(filename, number, t, n_samples, fs):
    signal = create_rect_wave(1, 0.9, 1, t, number, -0.45+0.1-0.0025) * np.sin(2 * np.pi * 100 * t)
    save_wav_file(f"./example/{filename}.wav", signal, fs)
    fft_result = draw(filename, t, signal, fs, False)
    r = compute_magnitude_rate(n_samples, fft_result, number, 1, 100, 0)
    print(f"{filename}: ", r)


def create_real_rect_wave_demo3(filename, number, t, n_samples, fs):
    signal = create_rect_wave(1, 0.9, 1, t, number, -0.45+0.1) * np.sin(2 * np.pi * 800 * t)
    save_wav_file(f"./example/{filename}.wav", signal, fs)
    fft_result = draw(filename, t, signal, fs, False)
    r = compute_magnitude_rate(n_samples, fft_result, number, 1, 100, 0)
    print(f"{filename}: ", r)


def create_real_rect_wave_demo4(filename, number, t, n_samples, fs):
    signal = create_rect_wave(1, 0.9, 1, t, number, -0.45+0.1) * np.sin(2 * np.pi * 500 * t)
    save_wav_file(f"./example/{filename}.wav", signal, fs)
    fft_result = draw(filename, t, signal, fs, False)
    r = compute_magnitude_rate(n_samples, fft_result, number, 1, 100, 0)
    print(f"{filename}: ", r)

def create_real_rect_wave_demo4_2(filename, number, t, n_samples, fs):
    signal = create_rect_wave(1, 0.9, 1, t, number, -0.45+0.1+0.0005) * np.sin(2 * np.pi * 500 * t)
    save_wav_file(f"./example/{filename}.wav", signal, fs)
    fft_result = draw(filename, t, signal, fs, False)
    r = compute_magnitude_rate(n_samples, fft_result, number, 1, 100, 0)
    print(f"{filename}: ", r)



def main(): 
    # 参数设置
    fs = 48000  # 采样率 (Hz)
    n_samples = 48000  # 采样点数
    f = 10  # 信号频率 (Hz)
    t = np.linspace(0, (n_samples - 1) / fs, n_samples)

    # # 方波
    signal = 0
    for n in range(1, 100, 2):
        signal += np.sin(np.pi * f * t * n) / (np.pi * n)
    fft_result = draw("fang_bo", t, signal, fs, False)


    # 完美矩形波
    signal = np.ones(48000)
    loop_width = 4800
    zero_width = int(loop_width / 4)
    half_zero_width = int(zero_width / 2)
    one_width = half_zero_width + loop_width - zero_width
    # 直接画出矩形波
    signal[half_zero_width:one_width] = 0
    signal[loop_width + half_zero_width:loop_width + one_width] = 0
    signal[2*loop_width + half_zero_width:2*loop_width + one_width] = 0
    signal[3*loop_width + half_zero_width:3*loop_width + one_width] = 0
    signal[4*loop_width + half_zero_width:4*loop_width + one_width] = 0
    signal[5*loop_width + half_zero_width:5*loop_width + one_width] = 0
    signal[6*loop_width + half_zero_width:6*loop_width + one_width] = 0
    signal[7*loop_width + half_zero_width:7*loop_width + one_width] = 0
    signal[8*loop_width + half_zero_width:8*loop_width + one_width] = 0
    signal[9*loop_width + half_zero_width:9*loop_width + one_width] = 0
    fft_result = draw("perfect_rect", t, signal, fs, False)


    # 完美直线
    signal = np.ones(48000)
    fft_result = draw("perfect_one", t, signal, fs, False)
    save_wav_file(f"./example/perfect_one.wav", signal, fs)


    # # 矩形波
    # signal = create_rect_wave(1, 0.25, 10, t, 100)
    # fft_result = draw("rect", t, signal, fs, False)



    # # 矩形波+正弦波
    # signal = create_rect_wave(1, 0.25, 10, t, 1000) * np.sin(2 * np.pi * 1000 * t)
    # fft_result = draw("rect10_sin1000", t, signal, fs, False)
    # r = compute_magnitude_rate(n_samples, fft_result, 1000, 10, 1000)
    # print("rect10_sin1000: ", r)


    # # 矩形波+正弦波
    # signal = create_rect_wave(1, 0.25, 10, t, 1000, 0.00025) * np.sin(2 * np.pi * 1000 * t)
    # fft_result = draw("rect10_sin1000_2", t, signal, fs, False)
    # r = compute_magnitude_rate(n_samples, fft_result, 1000, 10, 1000)
    # print("rect10_sin1000_2: ", r)


    # 矩形波+正弦波 n次谐波
    # create_real_rect_wave_demo("rect1_sin100_100", 100, t, n_samples, fs)
    # create_real_rect_wave_demo("rect1_sin100_300", 300, t, n_samples, fs)
    # create_real_rect_wave_demo("rect1_sin100_400", 400, t, n_samples, fs)
    # create_real_rect_wave_demo("rect1_sin100_600", 600, t, n_samples, fs)
    # create_real_rect_wave_demo("rect1_sin100_800", 800, t, n_samples, fs)
    # create_real_rect_wave_demo("rect1_sin100_1000", 1000, t, n_samples, fs)
    # create_real_rect_wave_demo("rect1_sin100_10000", 10000, t, n_samples, fs)
    # create_real_rect_wave_demo2("rect1_sin100_10000_2", 10000, t, n_samples, fs)
    # create_real_rect_wave_demo("rect1_sin100_20000", 20000, t, n_samples, fs)
    # create_real_rect_wave_demo2("rect1_sin100_20000_2", 20000, t, n_samples, fs)
    # create_real_rect_wave_demo3("rect1_sin800_20000", 20000, t, n_samples, fs)
    # create_real_rect_wave_demo4("rect1_sin500_20000", 20000, t, n_samples, fs)
    # create_real_rect_wave_demo4_2("rect1_sin500_20000_2", 20000, t, n_samples, fs)



if __name__ == "__main__":
    main()