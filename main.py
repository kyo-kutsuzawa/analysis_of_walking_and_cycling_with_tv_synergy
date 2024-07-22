"""
 Synergy demo: extraction of TV and TVTS
 
 EMG data from walking data
 (c)Eiji inomata, Takumi matsumura - Tohoku University, Sendai, Japan -
 
 Date: 20201030, 20210909

"""

from SynergyAnalyser_cycling import *
from Syn_cycling import find_mp
import numpy as np
from numpy import convolve
import matplotlib.pyplot as plt
import csv
import time
import math
import datetime

# from DetectAngle import *
from Gaitevent import *
import statistics


def SAVE_CSV(FILE, x):
    f = open(FILE, "a")
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(x)  # xの記録
    f.close()


def remove_artifact(data, div_value=10, mean_factor=20):
    mean_value = statistics.mean(abs(data))
    max_value = np.max(abs(data))
    print("mean: " + str(mean_value) + "\t max: " + str(max_value))
    for i in range(len(data)):
        if abs(data[i]) > (mean_value * mean_factor):
            data[i] = data[i] / div_value
    return data


#################################################################################################################################
# walkingの被験者
data_file = "./subject/walking/"
name = [data_file + "walking_4km_0.csv"]

# cyclingの被験者
# data_file="./subject/cycling/"
# name=[data_file+"cycling_480cycle.csv"]

# 計算結果の保存先ファイル名、mkdirで同名のファイルを作成する
file_name = "TEST"
print("file_name", file_name)

RR = [2]  # シナジーいくつ抽出するか。[2,3,4]とすればNsyn＝２～４までが計算される。
LorR = 1

# 保存するデータのfile名。時刻＋ファイルの名前
# print(data_file)
print(name)
dt_now = datetime.datetime.now()
FILE = str("./")
FILE += str(file_name)
FILE += str("/")
FILE += str(dt_now.hour)
FILE += str("時")
FILE += str(dt_now.minute)
FILE += str("分")
if LorR == 1:
    FILE += str("Right")
if LorR == 2:
    FILE += str("Left")
# FILE+=str(".csv")
TVTS_CSV = []
W_CSV = []
TVTSW = []
TVTSREC = []
TVW = []
TVREC = []

for Write in range(7):
    W_CSV.append(FILE + str("WW") + str(Write) + str(".csv"))
    TVTS_CSV.append(FILE + str("Params") + str(Write) + str(".csv"))
    TVTSW.append(FILE + str("TVTSW") + str(Write) + str(".png"))
    TVW.append(FILE + str("TVW") + str(Write) + str(".png"))
    TVREC.append(FILE + str("TVREC") + str(Write) + str(".png"))
    TVTSREC.append(FILE + str("TVTSW") + str(Write) + str(".png"))

save_TVTS_R = FILE + str("TVTS_R") + str(".csv")
save_TV_R = FILE + str("TV_R") + str(".csv")
save_Data_After_Fil_TVTS = FILE + str("Data_After_filtering_TVTS") + str(".csv")
save_Data_After_Fil_TV = FILE + str("Data_After_filtering_TV") + str(".csv")

# save_Time_Angle_TVTS=[]
##save_Time_Angle_TV=[]
# for Write_A in range(len(name)):
#    save_Time_Angle_TVTS.append(FILE+str("Time_and_angle_TVTS")+str(Write_A)+str(".csv"))
#    save_Time_Angle_TV=FILE+str("Time_and_angle_TV")+str(Write_A)+str(".csv")
save_Time_Angle_TV = FILE + str("Time_Angle") + str(".csv")
# TVTSの画像ファイル
TVTSR = "TVTS_R.png"

# TVの画像ファイル
TVR = "TV_R.png"
###################################################################################################################################
FILE = "C.csv"
personN = []
person = [0]

# name=["50.csv"]

x = len(name)

# for TVTS
cycle_emg = []
cycle_time = []
cycle_angle = []

# for T-V
percent_emg = []
percent_time = []
percent_angle = []

# データの取得
rpm = []
Each_Sample_Number = [0]
# 各csvファイルから読み込み
# percent_emg=時間正規化　cycle_emg=現実の時間スケール
for Data_N in range(x):
    GE = GaitEvent(name[Data_N], LorR)
    for j in range(GE.Each_Sample_Number):
        # print(GE.Each_Sample_Number)
        # 現実時間スケールデータの取得
        cycle_emg.append(GE.cycle_emg[j])
        cycle_time.append(GE.cycle_time[j])
        # cycle_angle.append(GE.cycle_angle[j])
        # plt.plot(GE.cycle_time[j],GE.cycle_angle[j])
        # plt.show()

        # 正規化時間スケールデータの取得
        percent_emg.append(GE.percent_emg[j])
        percent_time.append(GE.percent_time[j])
        # percent_angle.append(GE.percent_angle[j])

        # 速度の取得
        # rpm.append(GE.rpm[j])
    Each_Sample_Number.append(Each_Sample_Number[-1] + GE.Each_Sample_Number)
    print("ここまでの合計サンプル数", Each_Sample_Number[-1])

time.sleep(3)

TVTS_save_time = ["TVTS_time"]
# TVTS_save_angle = ["TVTS_angle"]
TV_save_time = ["TV_time"]
# TV_save_angle = ["TV_angle"]
cycle_count = 0
percent_count = 0
for time in range(len(cycle_time)):
    for sample in range(len(cycle_time[time])):

        TVTS_save_time.append(cycle_time[time][sample] + cycle_count)
        # TVTS_save_angle.append(cycle_angle[time][sample])
    cycle_count += cycle_time[time][-1]

for Ttime in range(len(percent_time)):
    for Tsample in range(len(percent_time[Ttime])):
        TV_save_time.append(percent_time[Ttime][Tsample] + percent_count)
        # TV_save_angle.append(percent_angle[Ttime][Tsample])
    percent_count += percent_time[Ttime][-1]

print("TVTSのセーブ終了")
print("TVのセーブ終了")

# load data from file
print("Loading raw EMG data (reaching to 8 target in frontal and sagittal planes)")
dataTVTS = Data().getFromCSVDATA(percent_time, percent_emg, len(percent_time))
dataTV = Data().getFromCSVDATA(percent_time, percent_emg, len(percent_time))


# どのデータを取得するか
# emgchannels = ["GMAX","GMED","VL","ST","TA","SOL","GM","PB"]
emgchannels = ["GMAX", "BF", "ST", "RF", "VL", "TA", "GM", "SOL"]
emgchannels = np.array([emgchannels[i] for i in range(8)])

##RRに格納された数のシナジーを抽出する
for No_of_W in range(len(RR)):
    # create object and preprocess EMG data
    print("今計算しているシナジー数は", RR[No_of_W])
    # time.sleep(2)
    TVTS = SynergyAnalyser(dataTVTS, emgchannels)
    TVTS.opt.verbose = 1

    TV = SynergyAnalyser(dataTV, emgchannels)
    TV.opt.verbose = 1

    ind = np.array(0)

    # filter and resample

    #####################################################################################
    # 信号を各サイクルの筋肉の最大筋力で正規化
    TVTS.CSV_NORMALIZE1(len(cycle_time), 8)
    TV.CSV_NORMALIZE1(len(percent_time), 8)

    # 信号を各サイクルの筋肉の平均値を減算する
    TVTS.CSV_Average(len(cycle_time), 8)
    TV.CSV_Average(len(percent_time), 8)

    # フィルタリングを行う。
    TVTS.opt.emgFilter.type = "butter"
    TVTS.opt.emgFilter.par = np.array(
        [4, 0.01]
    )  # 20 Hz @ 1KHz EMG sampling rate% 2f_l/f_s
    TVTS.opt.emgFilter.resample = 1  # 1
    TVTS.opt.emgFilter.resample_period = np.array([0.01])  # resampling period [s]
    TVTS.emgFilter()
    print("TVTSリサンプリング後のサイズは", np.size(TVTS.emg.obj[0].data[0]))

    TV.opt.emgFilter.type = "butter"
    TV.opt.emgFilter.par = np.array(
        [4, 0.999]
    )  # 20 Hz @ 1KHz EMG sampling rate% 2f_l/f_s
    TV.opt.emgFilter.resample = 1  # 1
    TV.opt.emgFilter.resample_period = np.array([0.01])  # resampling period [s]
    TV.emgFilter()

    #    ind = np.array(0)
    #    TVTS.emg.plot(ind)
    #   TVTS.SAVE_data(save_Data_After_Fil_TVTS,len(cycle_time),6)
    TV.SAVE_data(save_Data_After_Fil_TV, len(percent_time), 8)

    #########################################Find Synergy###################################################
    # TVTS.opt.find.type = 'spatiotemporal'
    # TVTS.opt.find.N = np.array([RR[No_of_W]]) #### SELECT THE NUMBER OF SYNERGIES ####
    # TVTS.opt.find.updateS = 1  #### CHANGE THE MINIMUM NUMBER OF SAMPLES IN Syn.py->find()->ntime ####
    #### 0-> TV    1-> TVTS ####
    ##TVTS.opt.find.nrep = 1
    # TVTS.opt.find.niter = np.array([5, 5, 1e-4])
    # TVTS.opt.find.plot = 0
    # TVTS.opt.find.N_trial = Each_Sample_Number[-1]
    # TVTS.findSynergies()
    print("次はTVです")
    # time.sleep(3)
    TV.opt.find.type = "spatiotemporal"
    TV.opt.find.N = np.array([RR[No_of_W]])  #### SELECT THE NUMBER OF SYNERGIES ####
    TV.opt.find.updateS = (
        0  #### CHANGE THE MINIMUM NUMBER OF SAMPLES IN Syn.py->find()->ntime ####
    )
    #### 0-> TV    1-> TVTS ####
    TV.opt.find.nrep = 5
    TV.opt.find.niter = np.array([5, 5, 1e-4])
    TV.opt.find.N_trial = Each_Sample_Number[-1]
    TV.opt.find.plot = 0
    TV.findSynergies()

    #####################################TVの画像とCSV保存############################################################

    #####パラメータの保存#######################
    TVTS_I = ["TVTS_I"]
    TVTS_T = ["TVTS_T"]
    TVTS_C = ["TVTS_C"]
    TVTS_S = ["TVTS_S"]

    TV_I = ["TV_I"]
    TV_T = ["TV_T"]
    TV_C = ["TV_C"]
    TV_S = ["TV_S"]
    saST_RPM = ["RPM"]

    for JJJ in range(len(rpm)):
        saST_RPM.append(rpm[JJJ])
    SAVE_CSV(TVTS_CSV[RR[No_of_W] - 1], saST_RPM)

    for III in range(np.size(TV.syn.C[0][0])):
        TV_I.append(TV.syn.I[0][0][III])
        TV_T.append(TV.syn.T[0][0][III])
        TV_C.append(TV.syn.C[0][0][III][0])
        TV_S.append(TV.syn.S[0][0][III])

    SAVE_CSV(TVTS_CSV[RR[No_of_W] - 1], TVTS_I)
    SAVE_CSV(TVTS_CSV[RR[No_of_W] - 1], TVTS_T)
    SAVE_CSV(TVTS_CSV[RR[No_of_W] - 1], TVTS_C)
    SAVE_CSV(TVTS_CSV[RR[No_of_W] - 1], TVTS_S)
    SAVE_CSV(TVTS_CSV[RR[No_of_W] - 1], TV_I)
    SAVE_CSV(TVTS_CSV[RR[No_of_W] - 1], TV_T)
    SAVE_CSV(TVTS_CSV[RR[No_of_W] - 1], TV_C)
    SAVE_CSV(TVTS_CSV[RR[No_of_W] - 1], TV_S)

    ####################################シナジーの保存######################################

    DAY = str(datetime.datetime.now)
    SAVE_CSV(W_CSV[RR[No_of_W] - 1], DAY)

    #    print("TVTS_Wのシェイプ",np.shape(TVTS.syn.W[0][0]))
    # for RRR in range(RR):
    #    W_for_save=[[],[],[],[],[],[],[]]
    #    for SSS in range(np.size(TVTS.syn.W[0][0][:,0])):
    #        for KKK in range(np.size(TVTS.syn.W[0][0][0,:])):
    #            W_for_save[SSS].append(TVTS.syn.W[0][0][SSS,KKK])
    #        SAVE_CSV(W_CSV[RR[No_of_W]-1],W_for_save[SSS])

    SAVE_CSV(W_CSV[RR[No_of_W] - 1], "TVのシナジー")
    print("TVTS_Wのシェイプ", np.shape(TV.syn.W[0][0]))
    W_for_save = [[], [], [], [], [], [], [], [], []]
    for SSS in range(np.size(TV.syn.W[0][0][:, 0])):
        for KKK in range(np.size(TV.syn.W[0][0][0, :])):
            W_for_save[SSS].append(TV.syn.W[0][0][SSS, KKK])
        SAVE_CSV(W_CSV[RR[No_of_W] - 1], W_for_save[SSS])

    #####################################TVの画像とCSV保存############################################################
    if No_of_W == len(RR) - 1:
        TV.opt.plot.type = "rsq"
        # saST.opt.plot.isect = ind
        TV.SAVE_plot(TVR, TVW[0], TVREC[0])

        TV.opt.plot.N = int(RR[No_of_W])
        TV.opt.plot.type = "W"
        TV.SAVE_plot(TVR, TVW[RR[No_of_W] - 1], TVREC[RR[No_of_W] - 1])

        TV.opt.plot.type = "rec"
        TV.opt.plot.isect = np.arange(480)  #### CHANGE THE NUMBER OF TRIALS ####
        # saST.opt.plot.isect = ind
        TV.SAVE_plot(TVR, TVW[RR[No_of_W] - 1], TVREC[RR[No_of_W] - 1])

    print(TV.syn.R)
    sa_R = []
    for XX in range(len(TV.opt.find.N)):
        sa_R.append(TV.syn.R[XX][0])
    print(sa_R)
    SAVE_CSV(save_TV_R, sa_R)

    # plt.figure()

    TVTS = None
    TV = None
