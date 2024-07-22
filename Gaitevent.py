"""
Detect gait event
歩行イベントにより筋電図時間を正規化する

<格納変数>
gyrxやemf...ジャイロやEMGの生データ
msw,ic,to...それぞれの歩行イベントのタイミング(NaNとその時の時間が格納)
to_time...toの時間
Each_Sample_Number・・・このcsvから取得したサンプル数
cycle_time,emg・・・リアルな時間での1サイクルごとのサンプリング
percent_time,emg・・・時間を正規化した1サイクルごとのサンプリング
"""

import numpy as np
from numpy import convolve
import matplotlib.pyplot as plt
import csv
import math
import datetime
from numpy.core.numeric import NaN
from numpy.lib.shape_base import expand_dims
from scipy import signal
import statistics

# Heuristic Gait Event Detection
# Detecting Mid Swing(MSw), Initial Contact(IC), Toe-Off(TO)


class GaitEvent:

    def __init__(self, name, LorR, Min_sample=5, Max_sample=35):

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

        self.time_gyrx, self.gyrx = Load_Gyro_data(name)
        self.gyrx = lowpass1(self.gyrx)
        msw, msw_t, self.ic, ic_t, self.to, to_t, time_gyrx = hged(
            self.gyrx, self.time_gyrx
        )
        self.to_time = To_resample(self.to, self.time_gyrx)
        self.ic_time = Ic_resample(self.ic, self.time_gyrx)

        (
            self.cycle_time,
            self.cycle_emg,
            self.Each_Sample_Number,
            self.percent_time,
            self.percent_emg,
        ) = Data_for_Synergy_Analyzer(
            self.time_emg, self.emg, self.ic_time, Min_sample, Max_sample
        )


##MSw,IC,TOを求める
def hged(gyrx, time_gyrx):
    gyrx = gyrx
    time = time_gyrx

    # Define threshold and initialization
    walkingthres = [-120, 30]
    msw = np.zeros(len(gyrx))
    ic = np.zeros(len(gyrx))
    to = np.zeros(len(gyrx))
    msw_t = np.zeros(len(gyrx))
    ic_t = np.zeros(len(gyrx))
    to_t = np.zeros(len(gyrx))
    msw_i = []
    ic_i = []
    to_i = []
    gyrxdot = np.zeros(len(gyrx))
    imin = 0

    for i in range(1, len(gyrx)):
        # calculate angular acceleration
        imin = i - 1
        gyrxdot[i] = gyrx[i] - gyrx[imin]

        # MidSwing rule
        if gyrx[i] < walkingthres[0]:
            if (
                gyrxdot[i] > 0
                and gyrxdot[imin] < 0
                and (np.count_nonzero(msw_i) == 0 or msw_i == 0)
            ):
                msw[imin] = gyrx[imin]
                msw_t[i] = msw_t[imin]
                msw_i = 1
                hndlrmsw = imin
                continue

            if gyrxdot[i] > 0 and gyrxdot[imin] < 0 and msw_i == 1:
                approxmsw = round((hndlrmsw + (imin)) / 2)
                msw[hndlrmsw] = 0
                msw[approxmsw] = gyrx[approxmsw]
                msw_t[-1] = msw_t[imin]
                continue

        # Initial Contact rule
        if gyrx[i] > walkingthres[1] and np.count_nonzero(msw_i) != 0:
            if gyrxdot[i] < 0 and gyrxdot[imin] > 0 and msw_i == 1:
                ic[imin] = gyrx[imin]
                ic_t[i] = ic_t[imin]
                ic_i = 1
                msw_i = 0
                continue

        # Toe Off rule
        if gyrx[i] > walkingthres[1] and np.count_nonzero(ic_i) != 0:
            if gyrxdot[i] < 0 and gyrxdot[imin] > 0 and ic_i == 1:
                to[imin] = gyrx[imin]
                to_t[i] = to_t[imin]
                to_i = 1
                ic_i = 0
                hndlrto = imin
                continue

            if gyrxdot[i] < 0 and gyrxdot[imin] > 0 and to_i == 1:
                to[hndlrto] = 0
                to[imin] = gyrx[imin]
                to_t[-1] = to_t[imin]
                continue

    for i in range(len(gyrx)):
        if msw[i] == 0:
            msw[i] = NaN
        if ic[i] == 0:
            ic[i] = NaN
        if to[i] == 0:
            to[i] = NaN
    """
    ##画像として出力
    #fig = plt.figure()
    plt.scatter(time,gyrx,s = 2,label="Gyr.X")
    plt.scatter(time,msw,s = 15,label="MSw")
    plt.scatter(time,ic,s = 20,color="k",label="IC")
    plt.scatter(time,to,s = 20,color="r",label="TO")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Angular velocity (deg/s)")
    plt.title("H-GED based on IMU placed on around Ankle/Heel")
    plt.show()
    #fig.savefig("img.png")
    """

    return msw, msw_t, ic, ic_t, to, to_t, time_gyrx


##ジャイロ用の4次のバターワースフィルタ
def lowpass1(
    emg,
    sampling_frequency=100,
    filter_order=4,
    cutoff_frequency=6,
):
    # Create lowpass filter and apply to rectified signal to get EMG envelope
    low_pass_band = cutoff_frequency / (sampling_frequency / 2)
    # 引数は（フィルターの次数,正規化したカットオフ周波数,タイプ）
    b2, a2 = signal.butter(filter_order, low_pass_band, btype="lowpass")
    lowpass_filtered = signal.filtfilt(b2, a2, emg)

    return lowpass_filtered


##時間とEMGデータを読み取る
def Load_EMG_data(name):
    time = np.genfromtxt(
        name,
        dtype=float,
        delimiter=",",
        converters=0,
        usecols=(0),
        unpack=True,
        filling_values=(0),
    )
    EMG = []
    for X in range(8):
        EMG.append([])
        EMG[X] = np.genfromtxt(
            name,
            dtype=float,
            delimiter=",",
            converters=0,
            usecols=(1 + X * 20),
            unpack=True,
            filling_values=(0),
        )
    return time, EMG


##時間とジャイロを読み取る
def Load_Gyro_data(name):
    time_gyrx = np.genfromtxt(
        name,
        dtype=float,
        delimiter=",",
        converters=0,
        usecols=168,
        unpack=True,
        filling_values=0,
    )
    gyrx = np.genfromtxt(
        name,
        dtype=float,
        delimiter=",",
        converters=0,
        usecols=169,
        unpack=True,
        filling_values=0,
    )
    return time_gyrx, gyrx


##バンドパスフィルター
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


##ローパスフィルター
def lowpass(
    emg,
    sampling_frequency=1111.11,
    filter_order=4,
    cutoff_frequency=5,
):
    # Create lowpass filter and apply to rectified signal to get EMG envelope
    low_pass_band = cutoff_frequency / (sampling_frequency / 2)
    # 引数は（フィルターの次数,正規化したカットオフ周波数,タイプ）
    b2, a2 = signal.butter(filter_order, low_pass_band, btype="lowpass")
    lowpass_filtered = signal.filtfilt(b2, a2, emg)

    return lowpass_filtered


##ハイパスフィルター
def highpass(
    emg,
    sampling_frequency=1111.11,
    filter_order=4,
    cutoff_frequency=20,
):
    # Create lowpass filter and apply to rectified signal to get EMG envelope
    high_pass_band = cutoff_frequency / (sampling_frequency / 2)
    # 引数は（フィルターの次数,正規化したカットオフ周波数,タイプ）
    b3, a3 = signal.butter(filter_order, high_pass_band, btype="highpass")
    highpass_filtered = signal.filtfilt(b3, a3, emg)

    return highpass_filtered


def remove_artifact(data, div_value=10, mean_factor=15):
    mean_value = statistics.mean(abs(data))
    max_value = np.max(abs(data))
    print("mean: " + str(mean_value) + "\t max: " + str(max_value))
    for i in range(len(data)):
        if abs(data[i]) > (mean_value * mean_factor):
            # data[i] = data[i]/div_value
            data[i] = mean_value
            print("Wow!")
    return data


##TOのタイミング
def To_resample(to, time_gyrx):
    to_time = []
    for i in range(len(to)):
        if math.isnan(to[i]) == False:
            to_time.append(time_gyrx[i])
    return to_time


##ICのタイミング
def Ic_resample(ic, time_gyrx):
    ic_time = []
    for i in range(len(ic)):
        if math.isnan(ic[i]) == False:
            ic_time.append(time_gyrx[i])
    return ic_time


# toを参照し、emgtime行列を作る
def Data_for_Synergy_Analyzer(time_emg, emg, ic_time, Min_sample=2, Max_sample=131):
    cycle_time = []
    cycle_emg = []
    # cycle_angle=[]
    percent_time = []
    percent_emg = []
    # percent_angle=[]
    Each_Sample_Number = 0
    # rpm=[]

    cycle_time_each = []
    # cycle_angle_each=[]
    cycle_emg1_each = []
    cycle_emg2_each = []
    cycle_emg3_each = []
    cycle_emg4_each = []
    cycle_emg5_each = []
    cycle_emg6_each = []
    cycle_emg7_each = []
    cycle_emg8_each = []

    save_angle_time_index = 0
    percent_index = 0

    for m in range(Min_sample, Max_sample):  # 何試行分取ってくるか。2，15だと13サイクル
        # print("今どこにいるか", m)
        for o in range(len(time_emg)):
            """
            if time_bottom[m-1]<=time_emg[o]<=time_bottom[m]:
                #timeの中に角度を入れる
                cycle_time_each.append(0.0009*len(cycle_time_each))
                for len_angle in range(save_angle_time_index , len(time360)):
                    if abs(time_emg[o]-time360[len_angle])<0.0008:
                        cycle_angle_each.append(angle360[len_angle])
                        save_angle_time_index=len_angle
                        break
            """
            if ic_time[m - 1] <= time_emg[o] <= ic_time[m]:
                cycle_time_each.append(0.0009 * len(cycle_time_each))
                cycle_emg1_each.append(emg[0][o])
                cycle_emg2_each.append(emg[1][o])
                cycle_emg3_each.append(emg[2][o])
                cycle_emg4_each.append(emg[3][o])
                cycle_emg5_each.append(emg[4][o])
                cycle_emg6_each.append(emg[5][o])
                cycle_emg7_each.append(emg[6][o])
                cycle_emg8_each.append(emg[7][o])

        percent_time_each = []
        # percent_angle_each=[]
        percent_emg1_each = []
        percent_emg2_each = []
        percent_emg3_each = []
        percent_emg4_each = []
        percent_emg5_each = []
        percent_emg6_each = []
        percent_emg7_each = []
        percent_emg8_each = []

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

            # percent_angle_each.append(cycle_angle_each[near_no[-1]])
            percent_emg1_each.append(cycle_emg1_each[near_no[-1]])
            percent_emg2_each.append(cycle_emg2_each[near_no[-1]])
            percent_emg3_each.append(cycle_emg3_each[near_no[-1]])
            percent_emg4_each.append(cycle_emg4_each[near_no[-1]])
            percent_emg5_each.append(cycle_emg5_each[near_no[-1]])
            percent_emg6_each.append(cycle_emg6_each[near_no[-1]])
            percent_emg7_each.append(cycle_emg7_each[near_no[-1]])
            percent_emg8_each.append(cycle_emg8_each[near_no[-1]])

        cycle_time.append(cycle_time_each)
        # cycle_angle.append(cycle_angle_each)
        # cycle_emg.append([cycle_emg1_each,cycle_emg2_each,cycle_emg3_each,cycle_emg4_each,cycle_emg5_each,cycle_emg6_each,cycle_emg7_each])
        cycle_emg.append(
            [
                cycle_emg1_each,
                cycle_emg2_each,
                cycle_emg3_each,
                cycle_emg4_each,
                cycle_emg5_each,
                cycle_emg6_each,
                cycle_emg7_each,
                cycle_emg8_each,
            ]
        )

        percent_time.append(percent_time_each)
        # percent_angle.append(percent_angle_each)
        # percent_emg.append([percent_emg1_each,percent_emg2_each,percent_emg3_each,percent_emg4_each,percent_emg5_each,percent_emg6_each,percent_emg7_each])
        percent_emg.append(
            [
                percent_emg1_each,
                percent_emg2_each,
                percent_emg3_each,
                percent_emg4_each,
                percent_emg5_each,
                percent_emg6_each,
                percent_emg7_each,
                percent_emg8_each,
            ]
        )

        cycle_time_each = []
        # cycle_angle_each=[]
        cycle_emg1_each = []
        cycle_emg2_each = []
        cycle_emg3_each = []
        cycle_emg4_each = []
        cycle_emg5_each = []
        cycle_emg6_each = []
        cycle_emg7_each = []
        cycle_emg8_each = []

    Each_Sample_Number = len(cycle_time)

    return cycle_time, cycle_emg, Each_Sample_Number, percent_time, percent_emg
