"""
SynergyAnalyser: class to extract muscle synergies from EMG data

 (c) Felipe Moreira Ramos - Tohoku University, Sendai, Japan -
 

 Date: 20180910
"""

from abc import ABC
import numpy as np
import warnings
import matplotlib.pyplot as plt
import csv

from Syn_cycling import Syn
from Syn_cycling import getShiftedMatrix
from Data import Data
from EmgData import EmgData
from Traces import Traces
from Bars import Bars
from SizBox import SizBox
from ArrayBox import ArrayBox
import statistics


def SAVE_CSV(FILE, x):
    f = open(FILE, "a")
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(x)
    f.close()


class SynergyAnalyser:
    # Properties
    emg = None
    syn = None
    isect = None
    ssect = None
    info = None
    opt = None

    # Methods
    """
     creates object
    
     obj = SynergyAnalyser(data)
     obj = SynergyAnalyser(data,chlabels)
    
     data must be a structure array with .emg and .emgtime fields
      
    """

    def __init__(self, data=None, chlabels=None):
        if data is None:
            return

        if chlabels is None:
            self.emg = EmgData(data)  # create EmgData
        else:
            self.emg = EmgData(data, [], chlabels)  # create EmgData

        self.syn = Syn()  # create Syn object
        self.info = getInfo(data)  # create getInfo
        self.opt = Opt().getDefOpt()  # create getDefOpt

    """
    Filter emg data
    """
    """
    Filter emg data
    """

    def CSV_ABS(self, trialCSV):
        for i in range(trialCSV):
            self.emg.obj[i].data = np.abs(self.emg.obj[i].data)
        return self

    def CSV_NORMALIZE2(
        self, trialCSV, NoofCHCSV, MAX
    ):  # 外から持ってきた値で正規化する方法
        for k in range(trialCSV):
            for l in range(NoofCHCSV):
                self.emg.obj[k].data[l] = self.emg.obj[k].data[l] / MAX[l]
        return self

    def CSV_NORMALIZE1(self, trialCSV, NoofCHCSV):
        MAX = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(trialCSV):
            for j in range(NoofCHCSV):
                if MAX[j] < np.max(self.emg.obj[i].data[j]):
                    MAX[j] = np.max(
                        self.emg.obj[i].data[j]
                    )  # 各サイクルの各筋肉のMAXをとる

        for k in range(trialCSV):
            for l in range(NoofCHCSV):
                self.emg.obj[k].data[l] = self.emg.obj[k].data[l] / MAX[l]
        return self

    def CSV_NORMALIZE3(self, trialCSV, NoofCHCSV):
        MAX = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(trialCSV):
            for j in range(NoofCHCSV):
                if MAX[j] < np.max(self.emg.obj[i].data[j]):
                    MAX[j] = np.max(
                        self.emg.obj[i].data[j]
                    )  # 各サイクルの各筋肉のMAXをとる

        for k in range(trialCSV):
            for l in range(NoofCHCSV):
                self.emg.obj[k].data[l] = self.emg.obj[k].data[l] / MAX[l]
        return self

    def CSV_PERSONEL_NORMALIZE(self, person, NoofCHCSV):
        for No_person in range(len(person) - 1):
            MAX = [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
            for i in range(person[No_person], person[No_person + 1]):
                for j in range(NoofCHCSV):
                    if MAX[j] < np.max(self.emg.obj[i].data[j]):
                        MAX[j] = np.max(
                            self.emg.obj[i].data[j]
                        )  # 各サイクルの各筋肉のMAXをとる

            for k in range(person[No_person], person[No_person + 1]):
                for l in range(NoofCHCSV):
                    self.emg.obj[k].data[l] = self.emg.obj[k].data[l] / MAX[l]
        return self

    def CSV_NORMALIZE_MVC(self, trialCSV, NoofCHCSV, MVC):
        for i in range(trialCSV):
            for j in range(NoofCHCSV):
                self.emg.obj[i].data[j] = self.emg.obj[i].data[j] / MVC[j]
        return self

    def CSV_Average(self, trialCSV, NoofCHCSV):
        nsect = self.info.shape[0]
        gr = []
        if not self.opt.average.gr:
            if not self.opt.average.grcond:
                gr.append(range(nsect))  # a single group with all trials
            else:
                self.groupTrials(self.opt.average.grcond)
        else:
            gr = self.opt.average.gr

        isect = np.unique(gr)  # all included sections
        for i in range(trialCSV):
            for j in range(NoofCHCSV):
                self.emg.obj[i].data[j] = self.emg.obj[i].data[j] - np.mean(
                    self.emg.obj[i].data[j]
                )
                for k in range(self.emg.obj[i].time.size):
                    if self.emg.obj[i].data[j][k] <= 0:
                        self.emg.obj[i].data[j][k] = 0.001
        return self

    def CSV_Average2(self, trialCSV, NoofCHCSV):
        nsect = self.info.shape[0]
        gr = []
        if not self.opt.average.gr:
            if not self.opt.average.grcond:
                gr.append(range(nsect))  # a single group with all trials
            else:
                self.groupTrials(self.opt.average.grcond)
        else:
            gr = self.opt.average.gr

        isect = np.unique(gr)  # all included sections
        for i in range(trialCSV):
            for j in range(NoofCHCSV):
                self.emg.obj[i].data[j] = self.emg.obj[i].data[j] - np.mean(
                    self.emg.obj[i].data[j]
                )
                for k in range(self.emg.obj[i].time.size):
                    if self.emg.obj[i].data[j][k] <= 0:
                        self.emg.obj[i].data[j][k] = 0.0000001
        return self

    ### 除去しきれなかったアーチファクトの除去・平均のmeanfactor倍の値をdiv_valueで割る
    def remove_artifact(self, trial, N, x, div_value=10, mean_factor=15):
        B = np.empty((0, 8))
        for i in range(trial):
            # print("Filtering後のセーブしている試行数",i)
            for k in range(self.emg.obj[i].time.size):
                A = []
                for j in range(N):
                    # self.emg.obj[i].data[j]=self.emg.obj[i].data[j]-np.mean(self.emg.obj[i].data[j])
                    A.append(self.emg.obj[i].data[j][k])
                B = np.append(B, np.array([A]), axis=0)

        mean_value = np.mean(abs(B[:, x]))
        max_value = np.max(abs(B[:, x]))
        print("mean: " + str(mean_value) + "\t max: " + str(max_value))
        for i in range(trial):
            for k in range(self.emg.obj[i].time.size):
                if abs(self.emg.obj[i].data[x][k]) > (mean_value * mean_factor):
                    self.emg.obj[i].data[x][k] = self.emg.obj[i].data[x][k] / div_value
        return self

    def SAVE_data(self, FILE, trial, N):
        for i in range(trial):
            # print("Filtering後のセーブしている試行数",i)
            for k in range(self.emg.obj[i].time.size):
                A = []
                A.append(self.emg.obj[i].time[k])
                for j in range(N):
                    # self.emg.obj[i].data[j]=self.emg.obj[i].data[j]-np.mean(self.emg.obj[i].data[j])
                    A.append(self.emg.obj[i].data[j][k])
                SAVE_CSV(FILE, A)

    """
    Filter emg data
    """

    def emgFilter(self):
        if self.opt.verbose:
            print("filtering EMG data using %s filter" % (self.opt.emgFilter.type))
        self.emg.filter(self.opt.emgFilter)

        if self.opt.emgFilter.resample:
            if self.opt.verbose:
                print(
                    "resampling EMG data using %5.3f [s] period"
                    % (self.opt.emgFilter.resample_period)
                )

            self.emg.mean(dt=self.opt.emgFilter.resample_period)

        return self

    """
    normalize EMG data amplitude
    """

    def emgNormalize(self):
        if not self.opt.emgNormalize.isect.size:
            self.opt.emgNormalize.isect = np.arange(len(self.emg))
        self.emg.normalize(self.opt.emgNormalize)

    """
       subtract tonic activity (based on linear ramp from onset to end) to
       extract phasic activity
    """

    def emgPhasic(self):
        # if self.info[0].events is None:
        #    warnings.warn('Movement onset and end events required for phasic EMGs',stacklevel=2)
        #    return

        nsect = self.info.shape[0]

        t_onset = []
        t_end = []
        for i in range(nsect):
            ind = np.where(self.info[i].events.code == 13)  # onset
            t_onset.append(self.info[i].events.time[ind[0]][0])
            ind = np.where(self.info[i].events.code == 14)
            t_end.append(self.info[i].events.time[ind[0]][0])
        t_onset = np.array(t_onset)
        t_end = np.array(t_end)

        self.emg.subtract(t=[t_onset, t_end])

    def emgDemean(self):
        print("emgDemean not implemented")

    def average(self):
        # trial section

        nsect = self.info.shape[0]
        gr = []
        if not self.opt.average.gr:
            if not self.opt.average.grcond:
                gr.append(range(nsect))  # a single group with all trials
            else:
                self.groupTrials(self.opt.average.grcond)
        else:
            gr = self.opt.average.gr

        isect = np.unique(gr)  # all included sections
        if not isect.size:  # is empty
            warnings.warn("No groups selected!", stacklevel=2)
            return
        if self.opt.verbose:
            print("averaging trials using %i groups" % (len(gr)))

        # tref
        tref = np.zeros((nsect, 1))  # assumed data have been aligned

        # trange
        trange = self.opt.average.trange

        # average emg data over trials
        self.emg.average(grOpt=[gr, tref, trange])  # changes the size of self.emg

        # update info
        ngr = len(gr)
        info = np.array([Info() for i in range(ngr)])
        for i in range(ngr):
            info[i].id = i + 1
            info[i].type = i + 1
            info[i].selected = 1
        self.info = info

    """
       group trial number according to conditions
      
         gr = groupTrials(obj[('condtype',condval),('condtype2',condval),...])
      
         gr -> list {ngr}
      
         each condition must have exactely ngr componets
      
         contype     condval        notes
                  [nrowls,ncols]
         ----------------------------------------------------------
         'type'     [ngr,ntype]     matches the info.type vector
         'type1'    [ngr,1]         matches the info.type(1) value
         'type2'    [ngr,1]         matches the info.type(2) value
         'type3'    [ngr,1]         matches the info.type(3) value
         'typei'    [ngr,2] (i,val) matches the info.type(i) value
         'selected' [ngr,1]         matches the info.selected value
         'id'       {ngr}           matches info.id value
    """

    def groupTrials(self, conditions=None):
        if conditions is None:
            ncond = 0
        else:
            ncond = len(conditions)

        ninfo = self.info.shape[0]

        ngr = conditions[0][1].shape[0]

        gr = []
        for k in range(ngr):
            gr.append([])
            for i in range(ninfo):
                ind_j = []
                for j in range(ncond):
                    condtype = conditions[j][0]
                    condval = conditions[j][1]

                    if condtype is "type":
                        if (self.info[i].type == condval[k]).all():
                            ind_j.append(j)
                    elif condtype is "type1":
                        if self.info[i].type[0] == condval[k]:
                            ind_j.append(j)
                    elif condtype is "type2":
                        if self.info[i].type[1] == condval[k]:
                            ind_j.append(j)
                    elif condtype is "type3":
                        if self.info[i].type[2] == condval[k]:
                            ind_j.append(j)
                    elif condtype is "typei":
                        if self.info[i].type[condval[k, 0]] == condval[k, 1]:
                            ind_j.append(j)
                    elif condtype is "selected":
                        if self.info[i].selected == condval[k]:
                            ind_j.append(j)
                    elif condtype is "id":
                        if self.info[i].id in condval[k]:
                            ind_j.append(j)

                # end j
                if len(ind_j) == ncond:
                    gr[k].append(i)
            # end i
        # end k
        return gr

    """
    % find trial number according to conditions
      
         ind = findTrials(obj,[('condtype',condval),('condtype2',condval),...])
      
         contype     condval        notes
                   [nrowls,ncols]
         ----------------------------------------------------------
         'id'       [nsect,1]       matches the info.id
         'type'     [nclass,ntype]  matches the info.type vector
         'type1'    [nclass,1]      matches the info.type(1) value
         'type2'    [nclass,1]      matches the info.type(2) value
         'type3'    [nclass,1]      matches the info.type(3) value
         'selected' [nclass,1]      matches the info.selected value
      
          logical OR between classes for each condition
          logical AND between conditions
    """

    def findTrials(self, conditions=None):
        if conditions is None:
            ncond = 0
        else:
            ncond = len(conditions)

        ntrial = self.info.shape[0]
        ind = []
        for i in range(ntrial):
            ind_j = []
            for j in range(ncond):
                condtype = conditions[j][0]
                condval = conditions[j][1]

                if condtype is "id":
                    if self.info[i].id in condval:
                        ind_j.append(j)
                elif condtype is "type":
                    nclass = condval.shape[0]
                    for k in range(nclass):
                        if (self.info[i].type == condval[k]).all():
                            ind_j.append(j)
                elif condtype is "type1":
                    nclass = condval.shape[0]
                    for k in range(nclass):
                        if self.info[i].type[0] == condval[k]:
                            ind_j.append(j)
                elif condtype is "type2":
                    nclass = condval.shape[0]
                    for k in range(nclass):
                        if self.info[i].type[1] == condval[k]:
                            ind_j.append(j)
                elif condtype is "type3":
                    nclass = condval.shape[0]
                    for k in range(nclass):
                        if self.info[i].type[2] == condval[k]:
                            ind_j.append(j)
                elif condtype is "typei":
                    ii = condval[0]
                    classes = condval[1]
                    nclass = condval.shape[0]
                    for k in range(nclass):
                        if self.info[i].type[ii] == classes[k]:
                            ind_j.append(j)
                elif condtype is "selected":
                    if self.info[i].selected == condval:
                        ind_j.append(j)

            # end j
            if len(ind_j) == ncond:
                ind.append(i)
        # end i
        return ind

    """
    run synergy extraction algorithm
    """

    def findSynergies(self):
        # select sections
        if self.isect is None:
            self.isect = self.findTrials([("selected", 1)])

        # prepare data matrix
        findtype = self.opt.find.type
        data, inds, nch = self.emg.getData(findtype, self.isect, self.ssect)

        # create Syn object
        self.syn = Syn(data, findtype, inds, nch, self.emg[0].chlabels)

        #  run algorithm
        if self.opt.verbose:
            print(
                "extracting %s synergies using %s alogithm"
                % (self.opt.find.type, self.opt.find.algo)
            )

        self.syn.find(self.opt.find)

    """
    plot synergies, coefficient, and EMG data reconstruction
    """

    def plot(self):
        if self.syn is None:
            return

        opt = self.opt.plot

        if opt.type == "rsq":
            # plot R^2 vs. N
            plt.figure()
            # prepare figure
            h_fig = plt.figure(
                num="R^2 vs. N for %s synergies (%s algorithm)"
                % (self.syn.opt.type, self.syn.opt.algo)
            )
            h_fig.patch.set_facecolor((0.9, 0.9, 0.9))
            # Paper type is defined when saving the figure
            # it is only possible to disable menubar for all figures
            h_axes = h_fig.add_subplot(111)

            h_axes.plot(self.syn.num(), self.syn.R, "k.-")
            h_axes.set_xlabel("Number of synergies")
            h_axes.set_ylabel("$R^2$")

            # h_fig.canvas.draw()
            # h_fig.canvas.flush_events()

        elif opt.type == "W":
            # plot one set of synergies in one figure
            plt.figure()
            # select repetition with N elements with max R^2 (if opt.N is specified)
            if not opt.N is None:
                opt.iset, opt.irep = self.syn.indmaxR(np.array([opt.N]))

            # self.opt.plot.isort = self.syn.sort(opt)
            # opt.isort = self.opt.plot.isort

            # specific options
            posaxes = np.array([0.08, 0.05, 0.9, 0.9])  # axes box

            # prepare figure
            opt.h_fig = plt.figure(
                num="%s synergies (%s algorithm)"
                % (self.syn.opt.type, self.syn.opt.algo)
            )
            # Paper type is defined when saving the figure
            # it is only possible to disable menubar for all figures

            # call syn plotting method
            self.syn.plot(opt, posaxes)

            opt.h_fig.canvas.draw()
            opt.h_fig.canvas.flush_events()

        elif opt.type == "rec":
            # plot the original data and the reconstruction as synerergy combinations for opt.isect sections
            plt.figure()
            # select specific sections
            if opt.isect is None:
                opt.isect = self.isect
            # select sections used for extraction
            opt.isect = np.intersect1d(opt.isect, self.isect)
            if not opt.isect.size:  # is empty
                warnings.warn("Specified sections not available", stacklevel=2)
                return

            # prepare figure
            opt.h_fig = plt.figure(
                num="%s syn. reconstruction (%s algorithm) (v%d)"
                % (self.syn.opt.type, self.syn.opt.algo, opt.isect.sum())
            )
            opt.h_fig.patch.set_facecolor((0.9, 0.9, 0.9))
            # Paper type is defined when saving the figure
            # it is only possible to disable menubar for all figures

            # set specific options
            opt.subtype = ""
            posaxes = np.array(
                [[0.08, 0.40, 0.9, 0.55], [0.08, 0.05, 0.9, 0.30]]
            )  # 2 axes box (1 for traces and 1 for coefs)
            autoscale = 0  # autoscale traces ylim

            # select repetition with N elements with max R^2 (if opt.N is specified)
            if not opt.N is None:
                opt.iset, opt.irep = self.syn.indmaxR(np.array([opt.N]))

            # compute reconstruction of selected sections

            trec = self.reconstruct(opt)
            # np.save('datahat_v%d.npy'%(opt.isect.sum()),trec[0].data)
            # np.save('data_v%d.npy'%(opt.isect.sum()),trec[1].data)

            # plot data and reconstruction
            tcmin = []
            tcmax = []
            for i in range(2):
                tcmin.append(trec[i].data.min())
                tcmax.append(trec[i].data.max())
            tcmin = np.array(tcmin)
            tcmax = np.array(tcmax)
            tcylim = np.array([tcmin.min(), tcmax.max()])  # if autoscale is off
            color = ["blue", "black"]
            fill = [0, 1]
            for i in range(2):
                trec[i].prop.color = color[i]
                trec[i].opt.fill = fill[i]
                trec[i].opt.autoscale = autoscale
                trec[i].opt.ylim = tcylim
                trec[i].timeunits = "samples"

            # pos_i = SizBox(posaxes[0],np.matrix([1]),dur,0,i,np.array([.02, .02]))
            pos_i = posaxes[0]
            hsect = opt.h_fig.add_subplot(111, label="emg0")
            hsect.set_position(pos_i)
            trec[1].plot(0, hsect)
            trec[0].plot(0, hsect)

            """
            emg = self.emg[opt.isect]
            emgrec = self.reconstruct(opt)
            dur = self.emg.duration(emg)
            
            #plot data and reconstruction
            optemg = EmgData.PlotOpt().getDefPlotOpt(self.emg[opt.isect])
            optemg.figure.append(opt.h_fig)
            optemg.color = 'black'
            optemg.fill = 1
            for i in range(nsect):
                pos_i = SizBox(posaxes[0],np.matrix([1]),dur,0,i,np.array([.02, .02]))
                hsect = optemg.figure[0].add_subplot(111, label='emg%d'%(i+1)) 
                hsect.set_position(pos_i)
                optemg.axes.append(hsect)
            self.emg.plot(opt.isect,optemg) 
            
            optemg.color = 'red'
            optemg.fill = 0
            optemg.usetitle = 0
            optemg.ylim = self.emg.emglim(self.emg[opt.isect])
            emgrec.plot(np.arange(nsect),optemg) 
            """
            # compute coefficeint time courses

            tcoef = self.coeftrace(opt)
            self.syn.contribution(opt)  # print contribution of synergies (%)

            # plot coefs on second axes
            if self.syn.type == "spatiotemporal":
                nt = tcoef.size
                tcmin = []
                tcmax = []
                for i in range(nt):
                    tcmin.append(tcoef[i].data.min())
                    tcmax.append(tcoef[i].data.max())
                tcmin = np.array(tcmin)
                tcmax = np.array(tcmax)
                # tcylim = np.array([tcmin.min(), tcmax.max()]) #if autoscale is off
                tcylim = np.array([0.0, 1.0])
                for i in range(nt):
                    tcoef[i].prop.color = "blue"
                    tcoef[i].prop.linewidth = 1.5
                    tcoef[i].opt.fill = 1
                    tcoef[i].opt.autoscale = autoscale
                    tcoef[i].opt.ylim = tcylim
                    tcoef[i].timeunits = "samples"
                    # tcoef[i].opt.profile = 1
                    # pos_i = SizBox(posaxes[1],np.matrix([1]),dur,0,i,np.array([.02, .02]))
                    pos_i = posaxes[1]
                    hsect = opt.h_fig.add_subplot(111, label="c%d" % (i + 1))
                    hsect.set_position(pos_i)
                    tcoef[i].plot(i, hsect)

            opt.h_fig.canvas.draw()
            opt.h_fig.canvas.flush_events()

    def SAVE_plot(self, fileR, fileW, fileREC):
        if self.syn is None:
            return

        opt = self.opt.plot

        if opt.type == "rsq":
            # plot R^2 vs. N

            # prepare figure
            h_fig = plt.figure(
                num="R^2 vs. N for %s synergies (%s algorithm)"
                % (self.syn.opt.type, self.syn.opt.algo)
            )
            h_fig.patch.set_facecolor((0.9, 0.9, 0.9))
            # Paper type is defined when saving the figure
            # it is only possible to disable menubar for all figures
            h_axes = h_fig.add_subplot(111)

            h_axes.plot(self.syn.num(), self.syn.R, "k.-")
            h_axes.set_xlabel("Number of synergies")
            h_axes.set_ylabel("$R^2$")

            plt.savefig(fileR)
            # h_fig.canvas.draw()
            # h_fig.canvas.flush_events()
            # h_fig=None

        elif opt.type == "W":
            # plot one set of synergies in one figure

            # select repetition with N elements with max R^2 (if opt.N is specified)
            if not opt.N is None:
                opt.iset, opt.irep = self.syn.indmaxR(np.array([opt.N]))

            # self.opt.plot.isort = self.syn.sort(opt)
            # opt.isort = self.opt.plot.isort

            # specific options
            posaxes = np.array([0.08, 0.05, 0.9, 0.9])  # axes box

            # prepare figure
            opt.h_fig = plt.figure(
                num="%s synergies (%s algorithm)"
                % (self.syn.opt.type, self.syn.opt.algo)
            )
            # Paper type is defined when saving the figure
            # it is only possible to disable menubar for all figures

            # call syn plotting method
            self.syn.plot(opt, posaxes)
            plt.savefig(fileW)
            # opt.h_fig.canvas.draw()
            # opt.h_fig.canvas.flush_events()

        elif opt.type == "rec":
            # plot the original data and the reconstruction as synerergy combinations for opt.isect sections

            # select specific sections
            if opt.isect is None:
                opt.isect = self.isect
            # select sections used for extraction
            opt.isect = np.intersect1d(opt.isect, self.isect)
            if not opt.isect.size:  # is empty
                warnings.warn("Specified sections not available", stacklevel=2)
                return

            # prepare figure
            opt.h_fig = plt.figure(
                num="%s syn. reconstruction (%s algorithm) (v%d)"
                % (self.syn.opt.type, self.syn.opt.algo, opt.isect.sum())
            )
            opt.h_fig.patch.set_facecolor((0.9, 0.9, 0.9))
            # Paper type is defined when saving the figure
            # it is only possible to disable menubar for all figures

            # set specific options
            opt.subtype = ""
            posaxes = np.array(
                [[0.08, 0.40, 0.9, 0.55], [0.08, 0.05, 0.9, 0.30]]
            )  # 2 axes box (1 for traces and 1 for coefs)
            autoscale = 0  # autoscale traces ylim

            # select repetition with N elements with max R^2 (if opt.N is specified)
            if not opt.N is None:
                opt.iset, opt.irep = self.syn.indmaxR(np.array([opt.N]))

            # compute reconstruction of selected sections

            trec = self.reconstruct(opt)
            # np.save('datahat_v%d.npy'%(opt.isect.sum()),trec[0].data)
            # np.save('data_v%d.npy'%(opt.isect.sum()),trec[1].data)

            # plot data and reconstruction
            tcmin = []
            tcmax = []
            for i in range(2):
                tcmin.append(trec[i].data.min())
                tcmax.append(trec[i].data.max())
            tcmin = np.array(tcmin)
            tcmax = np.array(tcmax)
            tcylim = np.array([tcmin.min(), tcmax.max()])  # if autoscale is off
            color = ["blue", "black"]
            fill = [0, 1]
            for i in range(2):
                trec[i].prop.color = color[i]
                trec[i].opt.fill = fill[i]
                trec[i].opt.autoscale = autoscale
                trec[i].opt.ylim = tcylim
                trec[i].timeunits = "samples"

            # pos_i = SizBox(posaxes[0],np.matrix([1]),dur,0,i,np.array([.02, .02]))
            pos_i = posaxes[0]
            hsect = opt.h_fig.add_subplot(111, label="emg0")
            hsect.set_position(pos_i)
            trec[1].plot(0, hsect)
            trec[0].plot(0, hsect)

            """
            emg = self.emg[opt.isect]
            emgrec = self.reconstruct(opt)
            dur = self.emg.duration(emg)
            
            #plot data and reconstruction
            optemg = EmgData.PlotOpt().getDefPlotOpt(self.emg[opt.isect])
            optemg.figure.append(opt.h_fig)
            optemg.color = 'black'
            optemg.fill = 1
            for i in range(nsect):
                pos_i = SizBox(posaxes[0],np.matrix([1]),dur,0,i,np.array([.02, .02]))
                hsect = optemg.figure[0].add_subplot(111, label='emg%d'%(i+1)) 
                hsect.set_position(pos_i)
                optemg.axes.append(hsect)
            self.emg.plot(opt.isect,optemg) 
            
            optemg.color = 'red'
            optemg.fill = 0
            optemg.usetitle = 0
            optemg.ylim = self.emg.emglim(self.emg[opt.isect])
            emgrec.plot(np.arange(nsect),optemg) 
            """
            # compute coefficeint time courses

            tcoef = self.coeftrace(opt)
            self.syn.contribution(opt)  # print contribution of synergies (%)

            # plot coefs on second axes
            if self.syn.type == "spatiotemporal":
                nt = tcoef.size
                tcmin = []
                tcmax = []
                for i in range(nt):
                    tcmin.append(tcoef[i].data.min())
                    tcmax.append(tcoef[i].data.max())
                tcmin = np.array(tcmin)
                tcmax = np.array(tcmax)
                # tcylim = np.array([tcmin.min(), tcmax.max()]) #if autoscale is off
                tcylim = np.array([0.0, 1.0])
                for i in range(nt):
                    tcoef[i].prop.color = "blue"
                    tcoef[i].prop.linewidth = 1.5
                    tcoef[i].opt.fill = 1
                    tcoef[i].opt.autoscale = autoscale
                    tcoef[i].opt.ylim = tcylim
                    tcoef[i].timeunits = "samples"
                    # tcoef[i].opt.profile = 1
                    # pos_i = SizBox(posaxes[1],np.matrix([1]),dur,0,i,np.array([.02, .02]))
                    pos_i = posaxes[1]
                    hsect = opt.h_fig.add_subplot(111, label="c%d" % (i + 1))
                    hsect.set_position(pos_i)
                    tcoef[i].plot(i, hsect)
            plt.savefig(fileREC)
            # opt.h_fig.canvas.draw()
        # opt.h_fig.canvas.flush_events()

    """
    synergy reconstruct of EMG data
    """

    def reconstruct(self, opt):
        trec = []

        datahat, inds = self.syn.reconstruct(opt)
        t = Traces(datahat, np.array([]))
        trec.append(t)

        # Nsect = inds.shape[0]
        # Ksample = int(datahat.shape[1]/Nsect)

        # data = np.zeros((self.syn.data.shape[0],Nsect*Ksample))
        # tt_time = np.zeros(nsect*Ksample)
        # for ii in range(Nsect):

        # data[:,ii*Ksample:(ii+1)*Ksample] = self.syn.data[:,inds[ii]]
        # tt_time[ii*Ksample:(ii+1)*Ksample] = self.emg[i].time
        data = self.syn.data[:, inds]
        t = Traces(data, np.array([]))

        nch = data.shape[0]
        t.chlabels = np.array(["$C_{%i}$" % (j + 1) for j in range(nch)])
        # t.chlabels = self.emg[0].chlabels

        trec.append(t)
        trec = np.array(trec)

        return trec

    """
    synergy coefficient traces
    """

    def coeftrace(self, opt):
        isect = opt.isect
        nsect = isect.size

        W = self.syn.W[opt.iset][opt.irep]
        C = self.syn.C[opt.iset][opt.irep]
        I = self.syn.I[opt.iset][opt.irep]
        T = self.syn.T[opt.iset][opt.irep]
        S = self.syn.S[opt.iset][opt.irep]
        ####Edition For Continuous Data##########
        Nsyn = self.syn.opt.N[opt.iset]
        ########################################
        Ninst = I.size
        Qsample = int(W.shape[1] / Nsyn)
        Ksample = int(self.syn.data.shape[1])

        c_data = np.zeros((Nsyn, Ksample))
        for i in range(Ninst):
            syn = I[i]

            center = T[i] + Qsample / 2
            init = center - S[i] / 2
            end = init + S[i]

            instance = np.ones(S[i])

            c_data[syn, np.maximum(int(init), 0) : np.minimum(int(end), Ksample)] += (
                C[i]
                * instance[
                    np.maximum(-int(init), 0) : S[i] + np.minimum(Ksample - int(end), 0)
                ]
            )
        c_data /= c_data.max()

        # sort synergies
        if not opt.isort.size or not opt.isort.size == Nsyn:
            isort = np.arange(Nsyn)
        else:
            isort = opt.isort
            c_data = c_data[isort, :]

        tt = []
        chstr = np.array(["$H_{%i}$" % (j + 1) for j in range(Nsyn)])
        ind = np.array([], dtype=int)
        # tt_data = np.zeros((Nsyn,nsect*Ksample))
        # tt_time = np.zeros(nsect*Ksample)
        for ii in range(nsect):
            i = opt.isect[ii]
            ind = np.append(ind, self.syn.inds[i])

        tt_data = c_data[:, ind]
        # tt_data[:,ii*Ksample:(ii+1)*Ksample] = c_data[:,self.syn.inds[i]]
        # tt_time[ii*Ksample:(ii+1)*Ksample] = self.emg[i].time

        t = Traces(tt_data, np.array([]))
        t.chlabels = chstr
        if len(opt.syncolor) and len(opt.syncolor) == Nsyn:
            t.opt.chcolor = opt.syncolor

        tt.append(t)
        tt = np.array(tt)
        return tt


"""
------------------------------------------------------------------
 subfunctions
------------------------------------------------------------------
"""
"""
 get trial type information contained into data

 for reaching data assumes data.info has plane, start, and target codes
  plane: 1 -> frontal, 2 -> sagittal
  start, target: 0 -> center, 1 -> medial/back, 3 -> down, 5 -> lateral/forward, 7-> up
"""


def getInfo(data):
    ntrial = len(data)

    # if not 'info' in data[0] or not 'plane' in data[0]['info'] or not 'start' in data[0]['info'] or not 'target' in data[0]['info']:
    #    warnings.warn('missing information in data',stacklevel=2)
    #    info = Info()
    #    return info

    info = np.array([Info() for i in range(ntrial)])

    for i in range(ntrial):
        # id
        info[i].id = i + 1

        # type
        info[i].type[0] = data[i].info.plane

        if data[i].info.start == 0:  # center-out/out-center, direction
            info[i].type[1] = 1
            info[i].type[2] = data[i].info.target
        else:
            info[i].type[1] = 0
            info[i].type[2] = np.mod(data[i].info.start + 3, 8) + 1  # assumes 8 targets

        # selected
        info[i].selected = 1

        # events
        info[i].events.code = np.array([13, 14])
        info[i].events.time = np.array([data[i].info.t_onset, data[i].info.t_end])

    return info


class Info:
    class Events:
        code = None
        time = None

    id = None
    type = None
    selected = None
    events = None

    def __init__(self):
        self.id = -1
        self.type = np.array([0, 0, 0])
        self.selected = -1
        self.events = self.Events()
        self.events.code = np.array([0, 0])
        self.events.time = np.array([0.0, 0.0])


"""
Inner classes: https://pythonspot.com/inner-classes/
"""

"""
get default options
"""


class Opt:
    class EmgFilter:
        type = None
        par = None
        resample = None
        resample_period = None

    class EmgNormalize:
        type = None
        isect = None
        normdata = None

    class Average:
        gr = None
        grcond = None
        trange = None

    class Find:
        type = None
        algo = None
        N = None
        nrep = None
        # bestrsqrep = None
        niter = None
        plot = None
        updateW = None
        updateT = None
        updateS = None

    class Plot:
        type = None
        subtype = None
        N = None
        iset = None
        irep = None
        isort = None
        isect = None
        h_fig = None
        syncolor = None
        papertype = None

    emgFilter = None
    emgNormalize = None
    average = None
    find = None
    plot = None
    verbose = None

    def __init__(self):
        self.emgFilter = self.EmgFilter()
        self.emgNormalize = self.EmgNormalize()
        self.average = self.Average()
        self.find = self.Find()
        self.plot = self.Plot()

        self.emgFilter.type = "fir1"
        self.emgFilter.par = np.array([50, 0.04])  # 20Hz @ 1Khz sampling
        self.emgFilter.resample = 1
        self.emgFilter.resample_period = np.array([0.01])  # resempling period [s]

        # self.emgNormalize.type = 1 #max absolute value of all channels
        self.emgNormalize.type = 2  # max absolute value of each channels
        self.emgNormalize.isect = np.array([])  # sections to use for computing max
        # self.emgNormalize.type = 0 # provided normalization vector (normdata)
        self.emgNormalize.normdata = []

        self.average.gr = []  # each cell contains the ids of the trials to average
        self.average.grcond = (
            []
        )  # group selection criteria type (e.g. ('type',[1 1 1;1 2 1]))
        self.average.trange = np.array([-0.5, 1])

        self.find.type = "spatial"
        self.find.algo = "nmf"
        self.find.N = np.arange(8) + 1
        self.find.nrep = 5
        # self.find.bestrsqrep = 1 #0-> save all reps; 1-> save only best rep
        self.find.niter = np.array([5, 5, 1e-4])
        self.find.plot = 0
        self.find.updateW = 1
        self.find.updateT = 1
        self.find.updateS = 1

        self.plot.type = "W"
        self.plot.subtype = "barh"  # plot subtype
        self.plot.N = None  # number of synergies to plot (choose rep with max R^2)
        self.plot.iset = 0  # first set
        self.plot.irep = 0  # first repetition
        self.plot.isort = np.array([], dtype=int)  # synergy sort order
        self.plot.isect = np.array([0])
        self.h_fig = None
        self.plot.syncolor = []
        self.plot.papertype = "a4"

        self.verbose = 0  # print additional messages

    def getDefOpt(self):
        self.emgFilter = self.EmgFilter()
        self.emgNormalize = self.EmgNormalize()
        self.average = self.Average()
        self.find = self.Find()
        self.plot = self.Plot()

        self.emgFilter.type = "fir1"
        self.emgFilter.par = np.array([50, 0.04])  # 20Hz @ 1Khz sampling
        self.emgFilter.resample = 1
        self.emgFilter.resample_period = np.array([0.01])  # resempling period [s]

        # self.emgNormalize.type = 1 #max absolute value of all channels
        self.emgNormalize.type = 2  # max absolute value of each channels
        self.emgNormalize.isect = np.array([])  # sections to use for computing max
        # self.emgNormalize.type = 0 # provided normalization vector (normdata)
        self.emgNormalize.normdata = []

        self.average.gr = []  # each cell contains the ids of the trials to average
        self.average.grcond = (
            []
        )  # group selection criteria type (e.g. {'type',[1 1 1;1 2 1]})
        self.average.trange = np.array([-0.5, 1])

        self.find.type = "spatial"
        self.find.algo = "nmf"
        self.find.N = np.arange(8) + 1
        self.find.nrep = 5
        # self.find.bestrsqrep = 1 #0-> save all reps; 1-> save only best rep
        self.find.niter = np.array([5, 5, 1e-4])
        self.find.plot = 0
        self.find.updateW = 1
        self.find.updateT = 1
        self.find.updateS = 1

        self.plot.type = "W"
        self.plot.subtype = "barh"  # plot subtype
        self.plot.N = None  # number of synergies to plot (choose rep with max R^2)
        self.plot.iset = 0  # first set
        self.plot.irep = 0  # first repetition
        self.plot.isort = np.array([])  # synergy sort order
        self.plot.isect = np.array([0])
        self.h_fig = None
        self.plot.syncolor = []
        self.plot.papertype = "a4"

        self.verbose = 0  # print additional messages

        return self


"""
data_S6.mat
    data: 1x160 struct
        emg: 18xN double
        emgtime: 1xN double
        pos: 3xM double 
        postime: 1xM double
        info: 1x1 struct
            plane: int
            start: int
            target: int
            t_onset: double
            t_end: double
    emgchannels: 18x1 string
    
    Example:
        Matlab: data(160).emg(18,:) -> emg signal 
            of electrode 18 of the trial 160
        Python: data['emg'][0][159][17,:] -> emg signal 
            of electrode 18 of the trial 160
"""


def convertDataToList(data):
    ntrial = data.size
    data_list = []

    for i in range(ntrial):
        info = {}
        info["plane"] = data["info"][0][i]["plane"][0][0][0][0]
        info["start"] = data["info"][0][i]["start"][0][0][0][0]
        info["target"] = data["info"][0][i]["target"][0][0][0][0]
        info["t_onset"] = data["info"][0][i]["t_onset"][0][0][0][0]
        info["t_end"] = data["info"][0][i]["t_end"][0][0][0][0]
        # info = {'plane': plane, 'start': start , 'target': target , 't_onset': t_onset, 't_end': t_end}

        trial = {}
        trial["emg"] = np.array(data["emg"][0][i])
        trial["emgtime"] = np.array(data["emgtime"][0][i][0])
        trial["pos"] = np.array(data["pos"][0][i])
        trial["postime"] = np.array(data["postime"][0][i][0])
        trial["info"] = info
        # trial = {'emg': emg, 'emgtime': emgtime, 'pos': pos, 'postime': postime, 'info': info}
        data_list.append(trial)

    # print(data_list[159]['info']['t_end'])
    return data_list
