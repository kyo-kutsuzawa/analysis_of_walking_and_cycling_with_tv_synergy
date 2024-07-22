"""
Class 角度をDelsys IMUから取得するクラスを作成する。

＜格納変数＞
time_magやself.emg1・・・・生データ
C_mag・・・地磁気の軌道の中心
S_mag・・・地磁気の初期位置
angle_mag・・・クランクの角度
time_bottom・・・0度の時の時間
cycle_time,emg・・・リアルな時間での1サイクルごとのサンプリング
percent_time,emg・・・時間を正規化した1サイクルごとのサンプリング
rpm・・・各サイクルごとのrpm
Each_Sample_Number・・・このcsvから取得したサンプル数
"""

import numpy as np
from numpy import convolve
import matplotlib.pyplot as plt
import csv
import time
import math
import datetime
from scipy import interpolate
from scipy import signal


class DetectAngle:

    def __init__(self, name, x=0, Min_sample=2, Max_sample=162):
        # Read time and mag signal data from CSV file
        if x == 0:
            (self.time_mag, self.mag_x, self.mag_y) = np.genfromtxt(
                name,
                dtype=float,
                delimiter=",",
                converters=0,
                usecols=(0, 1, 2),
                unpack=True,
                filling_values=(0, 0, 0),
            )
        if x == 1:
            (self.time_mag, self.mag_x, self.mag_y) = np.genfromtxt(
                name,
                dtype=float,
                delimiter=",",
                converters=0,
                usecols=(14, 15, 17),
                unpack=True,
                filling_values=(0, 0, 0),
            )
        if x == 2:
            (self.time_mag, self.mag_x, self.mag_y) = np.genfromtxt(
                name,
                dtype=float,
                delimiter=",",
                converters=0,
                usecols=(14, 15, 17),
                unpack=True,
                filling_values=(0, 0, 0),
            )

        self.time_emg, self.emg = Load_EMG_data(name)

        self.emg = bandpass(self.emg)
        print("bandpass is finished")
        self.emg = highpass(self.emg)
        print("highpass is finished")
        for X in range(len(self.emg)):
            self.emg[X] = abs(self.emg[X])
        print("ABS is finished")
        self.emg = lowpass(self.emg)
        print("lowpass is finished")

        # 中心点と初期位置の計算からクランクアングル計算
        self.C_mag = [
            (MAX(self.mag_x) + MIN(self.mag_x)) / 2,
            (MAX(self.mag_y) + MIN(self.mag_y)) / 2,
        ]
        # スタート位置を最大の点で統一すべき。
        self.S_mag = start_position(self.time_mag, self.mag_x, self.mag_y)
        # self.S_mag=[(MAX(self.mag_x)+MIN(self.mag_x))/2,MAX(self.mag_y)]
        self.angle_mag = []
        for i in range(len(self.time_mag)):
            self.angle_mag.append(
                Angle(self.C_mag, self.S_mag, self.mag_x[i], self.mag_y[i])
            )
        # 0度の時の時間
        # self.time_bottom=Bottom(self.time_mag,self.angle_mag)
        # self.time_top=(self.time_mag,self.angle_mag)

        self.time_angle_resample, self.angle_resample = Angle_resample(
            self.time_mag, self.angle_mag, self.time_emg
        )

        self.new_time_bottom = Bottom(self.time_angle_resample, self.angle_resample)
        self.new_time_top = Top(self.time_angle_resample, self.angle_resample)

        self.time360, self.angle360 = change_360_scale(
            self.time_angle_resample,
            self.angle_resample,
            self.new_time_top,
            self.new_time_bottom,
        )
        (
            self.cycle_time,
            self.cycle_angle,
            self.cycle_emg,
            self.rpm,
            self.Each_Sample_Number,
            self.percent_time,
            self.percent_angle,
            self.percent_emg,
        ) = Data_for_Synergy_Analyzer(
            self.time_emg,
            self.new_time_bottom,
            self.emg,
            self.time360,
            self.angle360,
            Min_sample,
            Max_sample,
        )

    def plot_xymag(self):
        print("Magnetのxとyをプロットします")
        plt.plot(self.mag_x, self.mag_y)
        plt.show()

    def plot_angle(self):
        print("アングルをプロットします")
        plt.plot(self.time_mag, self.angle_mag)
        plt.show()

    def plot_rpm(self):
        plt.plot(self.rpm)
        plt.show()


def Load_EMG_data(name):
    time = np.genfromtxt(
        name,
        dtype=float,
        delimiter=",",
        converters=0,
        usecols=(20),
        unpack=True,
        filling_values=(0),
    )
    EMG = []
    for X in range(6):
        EMG.append([])
        EMG[X] = np.genfromtxt(
            name,
            dtype=float,
            delimiter=",",
            converters=0,
            usecols=(41 + X * 20),
            unpack=True,
            filling_values=(0),
        )
    return time, EMG


def Ave(Data):  # 0をデータ数に含まずに、平均をとる
    Sum = 0
    T = 0
    for i in range(len(Data)):
        if Data[i] != 0:
            Sum = Sum + Data[i]
            T = T + 1
    return Sum / T


def MAX(Data):  # 0が最大にならないように回避するMAX　defend 0 is the max of deta
    Max = -1000000
    for i in range(len(Data)):
        if Max < Data[i] and Data[i] != 0:
            Max = Data[i]
    return Max


def MIN(Data):  # 0が最小にならないように回避するMIN　defend 0 is the min of deta
    Min = 1000000
    for i in range(len(Data)):
        if Min > Data[i] and Data[i] != 0:
            Min = Data[i]
    return Min


def start_position(
    t, x, y
):  # 開始三秒間における位置の特定(この点を初期位置角度＝０度として角度計算をする)　decide start position in 3s
    A = 0
    xs = 0
    xT = 0
    ys = 0
    yT = 0

    for i in range(len(t)):

        if abs(t[i] - 1) < 0.1:
            A = i
            break
    for j in range(A):
        if x[j] != 0:
            xs = xs + x[j]
            xT = xT + 1
        if y[j] != 0:
            ys = ys + y[j]
            yT = yT + 1
    return [xs / xT, ys / yT]


def Angle(
    C, S, x, y
):  # 初期位置を基準とした角度計算　C＝Center[x,y] S=Start[x,y]  のことである
    vec_0 = [S[0] - C[0], S[1] - C[1]]
    vec = [x - C[0], y - C[1]]

    n = vec_0[0] * vec[0] + vec_0[1] * vec[1]
    size_0 = (vec_0[0] ** 2 + vec_0[1] ** 2) ** 0.5
    size = (vec[0] ** 2 + vec[1] ** 2) ** 0.5

    angle = np.arccos(n / (size * size_0)) * 180 / np.pi
    return angle


def Angle_resample(angle_time, angle, emg_time):
    new_angle_time = []
    new_angle = []
    for i in range(np.argmax(angle_time)):
        new_angle_time.append(angle_time[i])
        new_angle.append(angle[i])

    t0 = 0  # 初期時間[s]
    tf = np.max(angle_time)  # 終了時間[s]
    print("終了時間", tf)
    dt = 0.0009  # 時間刻み[s]
    # f = 1    # 周波数[Hz}
    t = np.array(angle_time)  # 時間軸
    y = np.array(angle)

    # 補間関数fを作成
    f = interpolate.interp1d(t, y, kind="linear")

    # 補間した結果からリサンプリング波形を生成
    num = np.max(angle_time) / 0.0009

    t_resample = np.linspace(t0, tf, int(num))
    y_resample = f(t_resample)

    print(t_resample)
    return t_resample, y_resample


def cos_sim(v1, v2):  # cos類似度の計算
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def SAVE_CSV(FILE, x):
    f = open(FILE, "a")
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(x)  # xの記録
    f.close()


def Bottom(time_mag, angle_mag):
    # 極大値の時間をとる　方法としては160度を超えた時の時間を記録し、ピークごとにその平均の時間を0度に至った点とする。
    time_20 = []
    for k in range(len(time_mag)):
        if angle_mag[k] < 10:
            time_20.append(time_mag[k])
    time_bottom = []
    bottom = 0
    counter = 0
    pointer = []
    for l in range(1, len(time_20)):
        if time_20[l] - time_20[l - 1] <= 0.05:
            bottom += time_20[l]
            counter += 1
        # データの値がたまに飛んでることがあって、それを防止するため。
        elif counter == 0:
            print("カウントが０です。")
        else:
            time_bottom.append(bottom / counter)
            pointer.append(-1)
            bottom = 0
            counter = 0

    del time_bottom[0]
    del pointer[0]

    return time_bottom


def Top(time_mag, angle_mag):
    # 極大値の時間をとる　方法としては160度を超えた時の時間を記録し、ピークごとにその平均の時間を0度に至った点とする。
    time_20 = []
    for k in range(len(time_mag)):
        if angle_mag[k] > 170:
            time_20.append(time_mag[k])

    time_bottom = []
    bottom = 0
    counter = 0
    pointer = []
    for l in range(1, len(time_20)):
        if time_20[l] - time_20[l - 1] <= 0.05:
            bottom += time_20[l]
            counter += 1
        # データの値が飛んでいるときのため
        elif counter == 0:
            print("カウントが0です。")
        else:
            time_bottom.append(bottom / counter)
            pointer.append(-1)
            bottom = 0
            counter = 0

    del time_bottom[0]
    del pointer[0]

    return time_bottom


def change_360_scale(angle_time, angle, top, bottom):
    flag = 0
    flag_count = []
    time360 = []
    angle360 = []
    len_bottom_top = []
    len_bottom_top.append(len(top))
    len_bottom_top.append(len(bottom))

    length = np.max(len_bottom_top)
    print(len_bottom_top, length)

    count = 1
    for i in range(len(angle_time)):
        for j in range(len(top)):
            if abs(top[j] - angle_time[i]) < 0.0005:
                flag = 1
        for k in range(len(bottom)):
            if abs(bottom[k] - angle_time[i]) < 0.0005:
                flag = 0
        if flag == 1:
            time360.append(angle_time[i])
            angle360.append(360 - angle[i])
        else:
            time360.append(angle_time[i])
            angle360.append(angle[i])
        flag_count.append(flag)

    return angle_time, angle360


def bandpass(
    emg,
    sampling_frequency=1111.11,
    filter_order=4,
    lower_band_limit=20,
    upper_band_limit=450,
):
    # Normalise cut-off frequencies by sampling frequency
    lower_band_limit = lower_band_limit / (sampling_frequency / 2)
    upper_band_limit = upper_band_limit / (sampling_frequency / 2)

    # Create bandpass filter
    # Prosses EMG signal: bandpassed
    b1, a1 = signal.butter(
        filter_order, [lower_band_limit, upper_band_limit], btype="bandpass"
    )
    bandpass_filtered = signal.filtfilt(b1, a1, emg)
    return bandpass_filtered


def lowpass(emg, sampling_frequency=1111.11, filter_order=4, cutoff_frequency=5):
    # Create lowpass filter and apply to rectified signal to get EMG envelope
    low_pass_band = cutoff_frequency / (sampling_frequency / 2)

    # 引数は（フィルターの次数,正規化したカットオフ周波数,タイプ）
    b2, a2 = signal.butter(filter_order, low_pass_band, btype="lowpass")
    lowpass_filtered = signal.filtfilt(b2, a2, emg)
    return lowpass_filtered


def highpass(emg, sampling_frequency=1111.11, filter_order=4, cutoff_frequency=10):
    # Create lowpass filter and apply to rectified signal to get EMG envelope
    high_pass_band = cutoff_frequency / (sampling_frequency / 2)

    # 引数は（フィルターの次数,正規化したカットオフ周波数,タイプ）
    b3, a3 = signal.butter(filter_order, high_pass_band, btype="highpass")
    highpass_filtered = signal.filtfilt(b3, a3, emg)
    return highpass_filtered


def substruct_average(emg):
    for i in range(len(emg)):
        emg[i] = emg[i] - np.mean(emg[i])
        for j in range(len(emg[i])):
            if emg[i][j] <= 0:
                emg[i][j] = 0
    return emg


def Normalize(emg, mvcs):
    for i in range(len(emg)):
        emg[i] = emg[i] / mvcs[i]
    return emg


# time_bottom を参照して、emgtime行列を作る。
def Data_for_Synergy_Analyzer(
    time_emg, time_bottom, emg, time360, angle360, Min_sample, Max_sample
):
    cycle_time = []
    cycle_emg = []
    cycle_angle = []
    percent_time = []
    percent_emg = []
    percent_angle = []
    Each_Sample_Number = 0
    rpm = []

    cycle_time_each = []
    cycle_angle_each = []
    # cycle_emg1_each=[]
    cycle_emg2_each = []
    cycle_emg3_each = []
    cycle_emg4_each = []
    cycle_emg5_each = []
    cycle_emg6_each = []
    cycle_emg7_each = []

    save_angle_time_index = 0
    percent_index = 0

    for m in range(Min_sample, Max_sample):  # 何試行分取ってくるか。2，15だと13サイクル
        print("今どこにいるか", m)
        for o in range(len(time_emg)):
            if time_bottom[m - 1] <= time_emg[o] <= time_bottom[m]:
                # timeの中に角度を入れる
                cycle_time_each.append(0.0009 * len(cycle_time_each))
                for len_angle in range(save_angle_time_index, len(time360)):
                    if abs(time_emg[o] - time360[len_angle]) < 0.0008:
                        cycle_angle_each.append(angle360[len_angle])
                        save_angle_time_index = len_angle
                        break
                # cycle_emg1_each.append(emg1[o])
                cycle_emg2_each.append(emg[0][o])
                cycle_emg3_each.append(emg[1][o])
                cycle_emg4_each.append(emg[2][o])
                cycle_emg5_each.append(emg[3][o])
                cycle_emg6_each.append(emg[4][o])
                cycle_emg7_each.append(emg[5][o])
        rpm.append(60 * 1 / cycle_time_each[-1])  # rpmの計算

        percent_time_each = []
        percent_angle_each = []
        # percent_emg1_each=[]
        percent_emg2_each = []
        percent_emg3_each = []
        percent_emg4_each = []
        percent_emg5_each = []
        percent_emg6_each = []
        percent_emg7_each = []
        near_no = [0]

        for p in range(1000):
            near = 10000
            near_kousin = 0
            percent_time_each.append(p / 1000)
            for q in range(near_no[-1], len(cycle_time_each)):
                if abs(q / len(cycle_time_each) - p / 1000) < near:
                    near = abs(q / len(cycle_time_each) - p / 1000)
                    near_kousin = q
            near_no.append(near_kousin)

            percent_angle_each.append(cycle_angle_each[near_no[-1]])
            # percent_emg1_each.append(cycle_emg1_each[near_no[-1]])
            percent_emg2_each.append(cycle_emg2_each[near_no[-1]])
            percent_emg3_each.append(cycle_emg3_each[near_no[-1]])
            percent_emg4_each.append(cycle_emg4_each[near_no[-1]])
            percent_emg5_each.append(cycle_emg5_each[near_no[-1]])
            percent_emg6_each.append(cycle_emg6_each[near_no[-1]])
            percent_emg7_each.append(cycle_emg7_each[near_no[-1]])

        cycle_time.append(cycle_time_each)
        cycle_angle.append(cycle_angle_each)
        # cycle_emg.append([cycle_emg1_each,cycle_emg2_each,cycle_emg3_each,cycle_emg4_each,cycle_emg5_each,cycle_emg6_each,cycle_emg7_each])
        cycle_emg.append(
            [
                cycle_emg2_each,
                cycle_emg3_each,
                cycle_emg4_each,
                cycle_emg5_each,
                cycle_emg6_each,
                cycle_emg7_each,
            ]
        )

        percent_time.append(percent_time_each)
        percent_angle.append(percent_angle_each)
        # percent_emg.append([percent_emg1_each,percent_emg2_each,percent_emg3_each,percent_emg4_each,percent_emg5_each,percent_emg6_each,percent_emg7_each])
        percent_emg.append(
            [
                percent_emg2_each,
                percent_emg3_each,
                percent_emg4_each,
                percent_emg5_each,
                percent_emg6_each,
                percent_emg7_each,
            ]
        )

        cycle_time_each = []
        cycle_angle_each = []
        # cycle_emg1_each=[]
        cycle_emg2_each = []
        cycle_emg3_each = []
        cycle_emg4_each = []
        cycle_emg5_each = []
        cycle_emg6_each = []
        cycle_emg7_each = []
    Each_Sample_Number = len(cycle_time)
    return (
        cycle_time,
        cycle_angle,
        cycle_emg,
        rpm,
        Each_Sample_Number,
        percent_time,
        percent_angle,
        percent_emg,
    )
