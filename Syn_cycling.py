"""
 Syn: class to extract and plot muscle synergies

 (c) Felipe Moreira Ramos - Tohoku University, Sendai, Japan -
 

 Date: 20180910
"""

import numpy as np
from scipy import signal
from scipy.sparse.linalg import cg
from scipy.spatial import distance
from scipy import interpolate
import time

from ArrayBox import *
from Traces import Traces
from Bars import Bars


class Syn:
    type = None
    data = None
    inds = None
    nch = None
    W = None
    C = None
    I = None
    T = None
    S = None
    R = None
    R_default = None
    R_scaled = None
    Riter = None
    chlabels = None
    opt = None

    def __init__(self, data=None, syntype=None, inds=None, nch=None, chlabels=None):
        if not data is None:
            self.data = data
        if not syntype is None:
            self.type = syntype
        if not inds is None:
            self.inds = inds
        if not nch is None:
            self.nch = nch
        if not chlabels is None:
            self.chlabels = chlabels

    def find(self, opt):
        # set options or use defaults
        if opt is None:
            defopt = self.FindOpt().getDefFindOpt()
        else:
            defopt = self.FindOpt()
            defopt.type = opt.type
            defopt.algo = opt.algo
            defopt.N = opt.N
            defopt.nrep = opt.nrep
            defopt.niter = opt.niter
            defopt.plot = opt.plot
            defopt.updateW = opt.updateW
            defopt.updateT = opt.updateT
            defopt.updateS = opt.updateS
            defopt.N_trial = opt.N_trial  ###2020年6月18日　Ntrialの追加
        self.opt = defopt

        # loop on number of synergies and repetitions
        data = self.data

        # zero-clip negative data
        data = np.multiply(data, data > 0)
        # dmean data
        datam = np.mean(data, axis=1, keepdims=True)
        datazero = data - np.dot(datam, np.ones((1, data.shape[1])))
        if self.opt.zeromeandata:
            data = datazero

        self.W = []  # each iset has a W with shape( nch,N[iset]),
        # so it is not possible to store in a np.array
        self.C = []
        self.I = []
        self.T = []
        self.S = []
        self.R = []
        self.R_default = []
        self.R_scaled = []
        self.Riter = []

        nch, nsamp = data.shape
        ntrial = len(self.inds)
        if self.opt.updateS:  # TVTS
            """
            @TODO self.opt.synSamples
            """
            ntime = 50
        else:  # TV
            # ntime = int(nsamp/ntrial)
            ntime = 45
            print("ntimeは", ntime)
        N = self.opt.N
        nset = N.size
        for iset in range(nset):
            self.W.append([])
            self.C.append([])
            self.I.append([])
            self.T.append([])
            self.S.append([])
            self.R.append([])
            self.R_default.append([])
            self.R_scaled.append([])
            self.Riter.append([])
            for irep in range(self.opt.nrep):
                if self.opt.verbose:
                    print("synergy extraction: set %i, rep %i" % (iset + 1, irep + 1))

                if (
                    self.opt.algo == "nmf"
                ):  # Lee & Seung NMF algorithm with Eucledian cost
                    # initialize synergies and coefficients to uniform random
                    # values in [0 1]
                    Wini = np.random.rand(nch, N[iset] * ntime)
                    Nsyn = N[iset]

                    # run algorithm
                    W, C, I, T, S, R, R_default, R_scaled = find_nmf(
                        self.data, Wini, Nsyn, self.opt
                    )

                    if not self.opt.bestrsqrep or irep == 0:
                        self.W[iset].append(W)
                        self.C[iset].append(C)
                        self.I[iset].append(I)
                        self.T[iset].append(T)
                        self.S[iset].append(S)
                        self.R[iset].append(R[-1])
                        self.R_default[iset].append(R_default)
                        self.R_scaled[iset].append(R_scaled)
                        self.Riter[iset].append(R)
                    elif R[-1] > self.R[iset][0]:  # last extraction has higher Rsq
                        self.W[iset][0] = W
                        self.C[iset][0] = C
                        self.I[iset][0] = I
                        self.T[iset][0] = T
                        self.S[iset][0] = S
                        self.R[iset][0] = R[-1]
                        self.R_default[iset][0] = R_default
                        self.R_scaled[iset][0] = R_scaled
                        self.Riter[iset][0] = R
                elif self.opt.algo == "pca":  # Principal COmponent Analysis
                    print("PCA not implemented")

        self.R = np.array(self.R)
        self.R_default = np.array(self.R_default)
        self.R_scaled = np.array(self.R_scaled)

    class FindOpt:
        type = None
        algo = None
        N = None
        nrep = None
        clip = None
        zeromeandata = None

        niter = None
        nmaxiter = None
        bestrsqrep = None
        plot = None
        updateW = None
        updateT = None
        updateS = None
        iprint = None
        verbose = None
        N_trial = None  #########2020年6月18日　Ntrialの追加

        def __init__(self):
            self.type = "spatial"
            self.algo = "nmf"
            self.N = np.array([1])
            self.nrep = 1
            self.clip = 0
            self.zeromeandata = 0

            # specific options
            if self.algo == "nmf":
                self.niter = np.array(
                    [5, 5, 1e-4]
                )  # number of iterations or termination condition
                self.nmaxiter = 300
                self.bestrsqrep = 1  # 0-> save all reps; 1-> save only best rep
                self.plot = 1
                self.updateW = 1  #  0-> for data fitting
                self.updateT = 1  # Time-shift
                self.updateS = 1  # Time-scale
                self.clip = 1
                self.iprint = 0  # print message at each iteration
                self.N_trial = (
                    1  # 2020年6月18日　試行数×シナジー数で足切りするために使う。
                )
            elif self.algo == "pca":
                self.zeromeandata = 1

            self.verbose = 1

        """
        default extraction algorithm options
        """

        def getDefFindOpt(self, algo=None):
            if algo is None:
                algo = "nmf"

            self.type = "spatial"
            self.algo = algo
            self.N = np.array([1])
            self.nrep = 1
            self.clip = 0
            self.zeromeandata = 0

            # specific options
            if self.algo == "nmf":
                self.niter = np.array(
                    [5, 5, 1e-4]
                )  # number of iterations or termination condition
                self.nmaxiter = 700
                self.bestrsqrep = 1  # 0-> save all reps; 1-> save only best rep
                self.plot = 1
                self.updateW = 1  #  0-> for data fitting
                self.updateT = 1  # Time-shift
                self.updateS = 1  # Time-scale
                self.clip = 1
                self.iprint = 0  # print message at each iteration
                self.N_trial = (
                    20  # 2020年6月18日　試行数×シナジー数で足切りするために使う。
                )
            elif self.algo == "pca":
                self.zeromeandata = 1

            self.verbose = 1

    """
    recostruct EMG data with synergies
    """

    def reconstruct(self, opt=None):
        # set options or use defaults
        if opt is None:
            defopt = self.ReconstructOpt().getDefReconstructOpt()
        else:
            defopt = self.ReconstructOpt()
            defopt.iset = opt.iset
            defopt.irep = opt.irep
            defopt.isect = opt.isect
        opt = defopt

        if self.W is None:
            datahat = np.zeros(self.data.shape)
            warnings.warn("No synergies found", stacklevel=2)
            return datahat

        W = self.W[opt.iset][opt.irep]
        C = self.C[opt.iset][opt.irep]
        I = self.I[opt.iset][opt.irep]
        T = self.T[opt.iset][opt.irep]
        S = self.S[opt.iset][opt.irep]

        if (
            self.type == "spatial"
            or self.type == "temporal"
            or self.type == "spatiotemporal"
        ):
            # ind0 = 0
            ind = np.array([], dtype=int)
            # inds = []
            for ii in range(opt.isect.size):
                i = opt.isect[ii]
                # inds.append( ind0 + np.arange(self.inds[i].size) )
                # ind0 = ind0 + self.inds[i].size
                ind = np.append(ind, self.inds[i])
            # inds = np.array(inds,dtype=int)

            # Nsyn,Nsect = C.shape
            ####Edition For Continuous Data##########
            Nsyn = self.opt.N[opt.iset]
            ########################################
            Qsample = int(W.shape[1] / Nsyn)
            Ksample = self.data.shape[1]
            # H = getShiftedMatrix(I, T, C, Nsyn, Qsample, Ksample)
            datahat = getLinearCombination(W, I, T, S, C, Nsyn, Qsample, Ksample)
            # datahat = np.dot(W,H[:,ind])

        return datahat[:, ind], ind  # inds

    class ReconstructOpt:
        iset = None
        irep = None
        isect = None

        def __init__(self):
            self.iset = 0
            self.irep = 0
            self.isect = np.array([0])

        def getDefReconstructOpt(self):
            self.iset = 0
            self.irep = 0
            self.isect = np.array([0])

    def plot(self, opt=None, posaxes=None):
        # set options or use defaults
        if opt is None:
            defopt = self.PlotOpt().getDefPlotOpt(self.obj)
        else:
            defopt = self.PlotOpt()
            defopt.type = opt.type
            defopt.subtype = opt.subtype
            defopt.iset = opt.iset
            defopt.irep = opt.irep
            defopt.isort = opt.isort
            defopt.h_fig = opt.h_fig
            if not posaxes is None:
                defopt.posaxes = posaxes
            defopt.syncolor = opt.syncolor
        opt = defopt

        opt.h_fig.patch.set_facecolor((0.9, 0.9, 0.9))

        # plot
        if self.type == "spatiotemporal":
            if opt.type == "W":
                WW = self.W[opt.iset][opt.irep]
                M, Ntime = WW.shape
                N = self.opt.N[opt.iset]
                # N = self.C[opt.iset][opt.irep].shape[0]
                ntime = int(Ntime / N)
                W = np.zeros((M, N, ntime))
                for i in range(M):
                    W[i, :, :] = WW[i, :].reshape((N, ntime))
                # sort synergies
                if not opt.isort.size:  # is empty
                    isort = np.arange(N)
                else:
                    isort = opt.isort
                    W = W[:, isort, :]

                np.save("Synergies.npy", W)

                # ranges and number of plots
                maxW = np.max(W)  # by default, flattened array is used
                minW = np.min(W)

                if opt.subtype == "color":
                    print("Colored spatiotemporal synergies not implemented")
                else:
                    for i in range(N):
                        pos_i = ArrayBox(
                            opt.posaxes, 1, N, 0, i, np.array([0.02, 0.08])
                        )
                        hsect = opt.h_fig.add_subplot(111, label="%d" % (i + 1))
                        hsect.set_position(pos_i)

                        if M > 1:
                            Wi = np.transpose(np.squeeze(W[:, i, :]))
                        else:
                            Wi = np.squeeze(W[:, i, :])

                        tt_data = np.transpose(Wi)
                        tt_time = np.arange(Wi.shape[0])
                        t = Traces(tt_data, tt_time)
                        if i == 0:
                            t.chlabels = self.chlabels
                        t.timeunits = "samples"
                        t.label = "W%i" % (isort[i] + 1)
                        # t.opt.plottype = 'stack'

                        t.opt.autoscale = 0
                        t.opt.ylim = np.array([minW, maxW])
                        t.opt.fill = 1
                        if opt.syncolor and len(opt.syncolor) == N:
                            t.prop.color = opt.syncolor[i]

                        t.plot(i, hsect)

    class PlotOpt:
        type = None
        subtype = None
        iset = None
        irep = None
        isort = None
        h_fig = None
        h_axes = None
        posaxes = None
        syncolor = None
        chcolor = None

        def __init__(self):
            self.type = "W"
            self.subtype = "barh"
            self.iset = 0
            self.irep = 0
            self.isort = np.array([])
            self.h_fig = None
            self.h_axes = []
            self.posaxes = np.array([0.02, 0.02, 0.96, 0.96])
            self.syncolor = []
            self.chcolor = []

        def getDefPlotOpt(self, obj):
            self.type = "W"
            if obj.type == "spatial":
                self.subtype = "barh"
            self.iset = 0
            self.irep = 0
            self.isort = np.array([])
            self.h_fig = None
            self.h_axes = []
            self.posaxes = np.array([0.02, 0.02, 0.96, 0.96])
            self.syncolor = []
            self.chcolor = []

    """
    @Felipe
    Sort synergies based on the sum of all samples
    """

    def sort(self, opt=None):
        if opt is None:
            iset = 0
            irep = 0
        else:
            iset = opt.iset
            irep = opt.irep

        W = self.W[iset][irep]
        Nsyn = self.opt.N[iset]
        Nch, Ntime = W.shape
        Qsample = int(Ntime / Nsyn)

        s = np.zeros(Nsyn)
        for syn in range(Nsyn):  # Call np.arange Nsyn times instead of Ninst times
            k = syn * Qsample + np.arange(Qsample)
            s[syn] = np.sum(W[:, k])

        isort = np.argsort(s).astype(
            int
        )  # Order the synergies based on the sum of all samples and get their indices

        return isort

    """
    find set and repetition with max R^2 for N synergies
    """

    def indmaxR(self, N=None):
        Ns = self.num()

        if not Ns.size:
            return np.nan, np.nan

        if N is None:
            N = Ns[0]

        iN = np.where(Ns == N)
        if iN[0].size and self.R.size:
            iset = iN[0]
        else:  # find set with closest N
            iset = np.argmin(np.abs(Ns - N))
        irep = np.argmax(self.R[iset, :])
        return int(iset), int(irep)

    """
    return number of synergies
    """

    def num(self, iset=None):
        nset = len(self.C)  # number of sets

        if iset is None:
            iset = np.arange(nset)

        N = []
        for i in range(iset.size):
            if (
                self.type == "spatial"
                or self.type == "temporal"
                or self.type == "spatiotemporal"
            ):
                N.append(self.opt.N[iset[i]])

        return np.array(N)

    """
    @Felipe
    Similarity of sinergies from different experiments
    or
    Similarity of data and reconstruction 
    """

    def similarity(self, S1, S2):
        # normalize with 2-norm
        S1n = np.sqrt(np.sum(np.power(S1, 2), 1))[:, np.newaxis]
        S1 = np.divide(S1, S1n * np.ones((1, S1.shape[1])))

        S2n = np.sqrt(np.sum(np.power(S2, 2), 1))[:, np.newaxis]
        S2 = np.divide(S2, S2n * np.ones((1, S2.shape[1])))

        # sval = np.diagonal(distance.cdist(S1, S2, 'correlation'))
        # return sval
        Nch = S1.shape[0]
        s = np.zeros(Nch)
        for j in range(Nch):
            # a = signal.convolve(S1[j,:],S2[j,:])
            x = S1[j, :]
            y = S2[j, :]
            a = signal.correlate(x, y)
            a /= np.sqrt(np.dot(x, x) * np.dot(y, y))  # Normalized cross-correlation
            s[j] = a.max()
        return s

    """
    @Felipe
    Similarity of coefficients from different experiments
    or
    Similarity of data and reconstruction 
    """

    def similarity_coefficients(self, C1, C2):
        # normalize with 2-norm
        C1n = np.sqrt(np.sum(C1))
        C1 = C1 / C1n

        C2n = np.sqrt(np.sum(C2))
        C2 = C2 / C2n

        cval = distance.correlation(C1, C2)
        return cval

    """
    @Felipe
    Contribution of coefficients, time-shift and time-scale to reconstruct
    trials of opt.isect
    """

    def contribution(self, opt):
        W = self.W[opt.iset][opt.irep]
        C = self.C[opt.iset][opt.irep]
        I = self.I[opt.iset][opt.irep]
        T = self.T[opt.iset][opt.irep]
        S = self.S[opt.iset][opt.irep]
        ####Edition For Continuous Data##########
        Nsyn = self.opt.N[opt.iset]
        ########################################
        Ninst = I.size
        Qsample = int(W.shape[1] / Nsyn)
        Ksample = int(self.data.shape[1])

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

        Nsect = len(self.inds)
        isort = np.argsort(T)
        T = T[isort]
        S = S[isort]
        I = I[isort]
        Tcontribution = np.zeros((Nsect, Nsyn))
        Scontribution = np.zeros((Nsect, Nsyn))
        Ccontribution = np.zeros((Nsect, Nsyn))
        left_threshold = 0
        right_threshold = 0
        for ii in range(Nsect):
            ind = self.inds[ii]
            Ccontribution[ii, :] = c_data[
                :, ind[int(ind.size / 2)]
            ]  # Middle of the trial

            right_threshold += ind.size
            isect = np.where(
                ((T + ind.size / 2) >= left_threshold)
                & ((T + ind.size / 2) < right_threshold)
            )
            for i in isect:
                Tcontribution[ii, I[i]] = (T[i] - left_threshold) / ind.size
                Scontribution[ii, I[i]] = S[i] / ind.size

            left_threshold = right_threshold

        # print("Synergy Contribution:")
        # print(contribution)
        np.save("Coefficients.npy", Ccontribution)
        np.save("Time-Shifts.npy", Tcontribution)
        np.save("Time-Scales.npy", Scontribution)


"""
------------------------------------------------------------------
 subfunctions
------------------------------------------------------------------
"""
"""
extract non-negative synergies with Lee & Seung algorithm
"""


def find_nmf(data, W, Nsyn, opt):
    # data matrix V
    V = data

    # SST
    Vm = np.mean(V, axis=1, keepdims=True)
    resid = V - np.dot(Vm, np.ones((1, V.shape[1])))
    # SST = np.trace(np.dot(resid,np.transpose(resid)))
    SST = np.einsum("ij,ji->", resid, np.transpose(resid))  # trace(resid'*resid)

    V_default = V[:, 0 : int(V.shape[1] / 2)]
    Vm = np.mean(V_default, axis=1, keepdims=True)
    resid = V_default - np.dot(Vm, np.ones((1, V_default.shape[1])))
    SST_default = np.einsum(
        "ij,ji->", resid, np.transpose(resid)
    )  # trace(resid'*resid)

    V_scaled = V[:, int(V.shape[1] / 2) : -1]
    Vm = np.mean(V, axis=1, keepdims=True)
    resid = V_scaled - np.dot(Vm, np.ones((1, V_scaled.shape[1])))
    SST_scaled = np.einsum("ij,ji->", resid, np.transpose(resid))  # trace(resid'*resid)

    # monitor figures
    if opt.plot > 1:
        print("find_nmf plot not implemented")
    if opt.plot > 0:
        print("find_nmf plot not implemented")

    # timing info
    tic = time.time()
    # print(tic)

    #
    # general loop
    #
    niter = opt.niter[0]
    if opt.niter.size == 3:  # [min iter, monitor window, min R^2 diff]
        iterwin = opt.niter[1]
        errtol = opt.niter[2]
    else:
        iterwin = 1
        errtol = np.inf

    # loop while the rsq (abs) difference in the last iterwin iterations is less then errtol
    errtot = np.array([])
    rsq = np.array([])
    inderr = np.array([], dtype=int)
    it = 0
    while (
        it < niter or (np.abs(np.diff(rsq[inderr])) > errtol).any()
    ) and it < opt.nmaxiter:
        it = it + 1

        print(it)
        if opt.iprint:
            toc = time.time() - tic
            print("Iteration %i - time elspsed: %6.4f" % (it, toc))
            tic = time.time()
            tlast = toc

        sampleStep = 1
        scaleStep = 8
        #
        # find T
        #
        Qsample = int(W.shape[1] / Nsyn)
        Ksample = V.shape[1]
        # I, T = find_mp(V,W, Nsyn, Qsample, Ksample, sampleStep)

        #
        # find S
        #
        if opt.updateS:
            # S = find_tmp(V,W,I,T, Nsyn, Qsample, Ksample, sampleStep)
            I, T, S, C = find_parameters_Ntrial_Nsyn(
                V, W, Nsyn, Qsample, Ksample, sampleStep, scaleStep, opt.N_trial
            )
        else:
            I, T = find_mp(V, W, Nsyn, Qsample, Ksample, sampleStep, opt.N_trial)
            S = np.ones(I.size, dtype=int) * Qsample
            C = updateC(V, W, I, T, S, Nsyn, Qsample, Ksample)

        #
        # update C
        #
        # C = updateC(V,W,I,T,S, Nsyn, Qsample, Ksample)
        # H = getShiftedMatrix(I, T, C, Nsyn, Qsample, Ksample)

        #
        # update W
        #
        Vhat = getLinearCombination(W, I, T, S, C, Nsyn, Qsample, Ksample)
        """
        @todo optimize shiftAndScale
              optimize getLinearCombination
              optimize getSumOfInstances
        
        """
        if opt.updateW:
            # num = np.dot(V,np.transpose(H))# V*C'
            # den = np.dot(np.dot(W,H),np.transpose(H))# W*C*C'
            # den = np.multiply(den, den>0) + (den<=0)

            num = getSumOfInstances(V, I, T, S, C, Nsyn, Qsample, Ksample)
            den = getSumOfInstances(Vhat, I, T, S, C, Nsyn, Qsample, Ksample)
            den[den == 0] = 1  # don't update if zero
            # den = den + ((den>-0.001) & (den<0.001))
            W = np.multiply(W, np.divide(num, den))  # W.* num./den
            # W = W + 0.1*(num-den)

            # Normalize W
            for i in range(Nsyn):
                ii = i * Qsample + np.arange(Qsample)

                Wnorm = W[:, ii]
                Wn = np.sqrt(np.sum(np.power(Wnorm, 2)))  # Euclidean Norm
                # print(Wn)
                W[:, ii] = Wnorm / Wn  # W normalization
        #
        # display iteraction
        #
        if opt.plot > 1:
            print("find_nmf plot not implemented")
            # find_nmf_disppar(W,H,h_1)

        # Vhat = np.dot(W,H)

        # SSE
        resid = V - Vhat
        # ee = np.trace(np.dot(resid,np.transpose(resid)))
        ee = np.einsum("ij,ji->", resid, np.transpose(resid))  # trace(resid'*resid)
        errtot = np.append(errtot, ee)

        if opt.plot:
            print("find_nmf plot not implemented")
            # find_nmf_disperr(h_2,errtot,SST,errtol)
        inderr = np.fmax(0, np.arange(errtot.size - iterwin, errtot.size)).astype(int)

        # r-square
        rsq = 1 - errtot / SST
        print("R", rsq)

    resid_default = resid[:, 0 : int(resid.shape[1] / 2)]
    ee = np.einsum(
        "ij,ji->", resid_default, np.transpose(resid_default)
    )  # trace(resid'*resid)
    SSE_default = ee

    resid_scaled = resid[:, int(resid.shape[1] / 2) : -1]
    ee = np.einsum(
        "ij,ji->", resid_scaled, np.transpose(resid_scaled)
    )  # trace(resid'*resid)
    SSE_scaled = ee

    R_default = 1 - SSE_default / SST_default
    R_scaled = 1 - SSE_scaled / SST_scaled

    R = rsq

    if opt.plot:
        print("find_nmf plot not implemented")
    if opt.plot > 1:
        print("find_nmf plot not implemented")

    return W, C, I, T, S, R, R_default, R_scaled


"""
def find_nmf_disppar(W,C,h,labels):
def find_nmf_disperr(h,errtot,sst,errtol):

"""


def find_mp(data, W, Nsyn, Qsample, Ksample, sampleStep, N_trial):
    phiSize = (
        np.arange(0, Ksample, sampleStep).size
        + np.arange(0, Qsample, sampleStep).size
        - 1
    )
    threshold = 0.3 / sampleStep
    Nch = W.shape[0]
    most_small_sqrt = 10000

    # Reshape
    Wnorm = np.zeros((Nsyn, Nch, Qsample))
    filt = signal.hann(Qsample)
    for i in range(Nsyn):
        ii = i * Qsample + np.arange(Qsample)
        Wnorm[i] = np.multiply(W[:, ii], filt)
        # Wnorm[i] = W[:,ii]

    Hc = np.nan
    I = np.array([], dtype=int)  # intances of the synergies
    T = np.array([], dtype=int)  # delay of the instances of the synergies

    Ssample = Wnorm[0, 0, ::sampleStep].shape[0]
    # Use all the data
    ss = np.arange(Ksample)
    # print("ssは",ss)
    # time.sleep(3)
    residual = data[:, ss]
    residual2 = data[:, ss]
    # print("residualは",residual)
    #  time.sleep(3)

    refract = np.ones((Nsyn, phiSize))  # Refractory Period

    sumOfData = np.sum(residual)
    sumOfResidual = sumOfData
    windowsResidual = np.array([sumOfResidual / sumOfData])
    inderr = np.array([0], dtype=int)
    iterwin = 5
    errtol = 1e-3
    niter = 5
    count_FOR_while_number = 0

    while count_FOR_while_number < N_trial * Nsyn:
        # while (windowsResidual.size < niter or (np.abs(np.diff(windowsResidual[inderr]))>errtol).any()):
        count_FOR_while_number += 1

        t = np.zeros(Nsyn, dtype=int)  # index
        corr = np.zeros(Nsyn)
        corr2 = np.zeros(Nsyn)

        for i in range(Nsyn):
            # Compute the sum of the scalar products between the s-th data episode and the i-th synergy
            # use convolution to time-shift the synergy
            phi = np.zeros(phiSize)
            phi2 = np.zeros(phiSize)
            # print("phiのかたち",phi.shape)
            for j in range(Nch):
                # a = signal.convolve(residual[j,::sampleStep], Wnorm[i,j,::-sampleStep])
                x = residual[j, ::sampleStep]
                X_for_under = residual2[j, ::sampleStep]
                # print("Xは",x)
                y = Wnorm[i, j, ::sampleStep]
                a = signal.correlate(x, y)
                b = signal.correlate(x, y)

                # print("分母は",np.sqrt(np.dot(x, x) * np.dot(y, y)))
                if 0 < np.sqrt(np.dot(x, x) * np.dot(y, y)) < most_small_sqrt:
                    most_small_sqrt = np.sqrt(np.dot(x, x) * np.dot(y, y))
                if np.sqrt(np.dot(x, x) * np.dot(y, y)) != 0:
                    a /= np.sqrt(
                        np.dot(x, x) * np.dot(y, y)
                    )  # Normalized cross-correlation
                else:
                    # a /= most_small_sqrt
                    print("分母０きましたよー")
                    a = 0

                phi = phi + a
                phi2 = phi2 + b

                # print("phiは",phi)
                # time.sleep(2)
            # phi = np.sqrt(phi/Nch)
            t[i] = np.multiply(
                phi2 / Ssample, refract[i]
            ).argmax()  # extract refractory period from phi end get the time of the maximum correlation
            corr[i] = phi[t[i]]
            corr2[i] = phi2[t[i]]

        for each_I in range(Nsyn):
            # np.append(count_each_I,np.count_nonzero(I==each_I))
            if np.count_nonzero(I == each_I) == N_trial:
                corr2[each_I] = 0

        # select synergy and delay with highest correlation for the s-th episode
        Hsyn = corr2.argmax()  # get the synergy with highest correlation
        Ht = (
            int(t[Hsyn] - Ssample) * sampleStep + 1
        )  # Get the delay for the highest correlation
        # Ht = int(t[Hsyn])*sampleStep + 1#お試し
        Hc = corr[Hsyn] / Nch  # get the highest correlation
        # print("Hcは",Hc)
        if np.isnan(Hc):
            print("Hcはnanになりました", Hc)
            time.sleep(0.5)
            break
        # ( t , min(t+Q, K) )
        tInit = int(t[Hsyn] - Ssample + 1)
        tEnd = int(t[Hsyn] + Ssample - 1)
        refractPeriod = np.arange(
            np.maximum(tInit, 0), np.minimum(tEnd, phiSize)
        ).astype(int)
        refract[Hsyn, refractPeriod] *= 0

        I = np.append(I, Hsyn)
        T = np.append(T, Ht)

        # mutiply the selected element by its correlation and subtract from the data
        Wshifted = shiftChannels(Wnorm[Hsyn], Ht, Ksample)
        residual = residual - (Wshifted * (1 + Hc))

        # residual = np.multiply(residual,residual>0)
        residual[residual < 0.0] = 0.0  # set to zero any negative value
        sumOfResidual = np.sum(residual)
        windowsResidual = np.append(windowsResidual, sumOfResidual / sumOfData)

        inderr = np.fmax(
            0, np.arange(windowsResidual.size - iterwin, windowsResidual.size)
        ).astype(int)

        # print('Residual: ',windowsResidual[-1])

    # print('Residual: ',sumOfResidual)
    print("Synergies/Trial: ", I.size / 32)
    print("計算される推定Trial回数", I.size / Nsyn)
    print("Final Correlation: ", Hc)

    return I, T


"""
Find time scale of synergies using matching pursuit
Tries all possible scales between minSamples and maxSamples
and calculate the correlation with the data
"""


def find_tmp(data, W, I, T, Nsyn, Qsample, Ksample, sampleStep):
    ####Edition For Continuous Data##########
    Ninst = I.size
    ########################################

    minSamples = Qsample
    maxSamples = 10  # int(1.5*Qsample) - 1 #not always is Ksample

    S = np.ones(Ninst, dtype=int) * minSamples

    numSamples = np.arange(minSamples, maxSamples + 1, sampleStep)
    synInterval = [np.linspace(0, Qsample - 1, N) for N in numSamples]

    kk = []
    for syn in range(Nsyn):  # Call np.arange Nsyn times instead of Ninst times
        kk.append(syn * Qsample + np.arange(Qsample))
        # kk.append(slice(syn*Qsample,(syn+1)*Qsample))

    for i in range(Ninst):
        syn = I[i]
        k = kk[syn]

        center = T[i] + Qsample / 2
        init = center - numSamples / 2
        end = init + numSamples

        C = np.zeros(numSamples.size)
        interpFunction = interpolate.interp1d(synInterval[0], W[:, k])

        for j in range(numSamples.size):
            if not (int(end[j]) - int(init[j])) == numSamples[j]:
                end[j] += 1
            A = interpFunction(synInterval[j])[
                :,
                np.maximum(-int(init[j]), 0) : np.minimum(
                    numSamples[j], Ksample - int(init[j])
                ),
            ]
            B = data[:, np.maximum(int(init[j]), 0) : np.minimum(int(end[j]), Ksample)]
            """
            @Felipe
            cosine is a correlation with mean 0. 
            It is useful because most part of the synergy is 0.
            If you compare only the non-zero part (truncate data to the size of synergy),
            correlation has the same result of cosine.
            """
            # C = distance.cdist(A,B,'cosine')
            C[j] = sum(
                [distance.cosine(A[ch, :], B[ch, :]) for ch in range(W.shape[0])]
            )

        S[i] = numSamples[C.argmin()]

    return S


def find_parameters_Ntrial_Nsyn(
    data, W, Nsyn, Qsample, Ksample, sampleStep, scaleStep, N_trial
):
    # Parameters for time-scale
    how_many_trial = 31
    minSamples = Qsample - 10
    maxSamples = Qsample + 10  # int(1.5*Qsample) - 1 #not always is Ksample
    numSamples = np.arange(minSamples, maxSamples + 1, sampleStep)
    synInterval = [np.linspace(0, Qsample - 1, N) for N in numSamples]

    # Utils
    Nch = W.shape[0]
    phiSize = (
        np.arange(0, Ksample, sampleStep).size
        + np.arange(0, Qsample, sampleStep).size
        - 1
    )
    threshold = 0.15
    most_small_sqrt = 10000

    kk = []
    for syn in range(Nsyn):  # Call np.arange Nsyn times instead of Ninst times
        kk.append(syn * Qsample + np.arange(Qsample))
        # kk.append(slice(syn*Qsample,(syn+1)*Qsample))

    # Reshape
    Wnorm = np.zeros((Nsyn, Nch, Qsample))
    interpFunction = []
    filt = signal.hann(Qsample)
    for syn in range(Nsyn):
        k = kk[syn]
        Wnorm[syn] = np.multiply(W[:, k], filt)
        # Wnorm[syn] = W[:,k]
        interpFunction.append(interpolate.interp1d(np.arange(Qsample), W[:, k]))

    Ssample = Wnorm[0, 0, ::sampleStep].shape[0]

    # Outputs
    I = np.array([], dtype=int)  # intances of the synergies
    T = np.array([], dtype=int)  # delay of the instances of the synergies
    S = np.array([], dtype=int)  # scale of the instances of the synergies

    # Use all the data
    ss = np.arange(Ksample)
    residual = data[:, ss]
    refract = np.ones((Nsyn, phiSize))  # Refractory Period
    real_refract = np.ones((Nsyn, Ksample))

    Hc = 1
    countI = 0
    while countI < N_trial * Nsyn:

        # while Hc > threshold and countI<how_many_trial*Nsyn:
        # while np.mean(residual)>0.499 :
        countI += 1
        t = np.zeros(Nsyn, dtype=int)  # index
        corr = np.zeros(Nsyn)
        corr2 = np.zeros(Nsyn)

        for syn in range(Nsyn):
            # Compute the sum of the scalar products between the s-th data episode and the i-th synergy
            # use convolution to time-shift the synergy
            phi = np.zeros(phiSize)
            phi2 = np.zeros(phiSize)
            for ch in range(Nch):
                # a = signal.convolve(residual[ch,::sampleStep], Wnorm[syn,ch,::-sampleStep])
                #              x = residual[ch,::sampleStep]*real_refract[Nsyn-1,::sampleStep]
                x = residual[ch, ::sampleStep]
                y = Wnorm[syn, ch, ::sampleStep]
                a = signal.correlate(x, y)
                b = signal.correlate(x, y)

                if 0 < np.sqrt(np.dot(x, x) * np.dot(y, y)) < most_small_sqrt:
                    most_small_sqrt = np.sqrt(np.dot(x, x) * np.dot(y, y))
                if np.sqrt(np.dot(x, x) * np.dot(y, y)) != 0:
                    a /= np.sqrt(
                        np.dot(x, x) * np.dot(y, y)
                    )  # Normalized cross-correlation
                else:
                    # a /= most_small_sqrt
                    print("Warning Waring this is not a test")
                    # time.sleep(0.0001)
                    a = 0
                # else:
                #    a=0
                # a /= np.sqrt(np.dot(x, x) * np.dot(y, y))#Normalized cross-correlation
                phi = phi + a
                phi2 = phi2 + b

            # phi = np.sqrt(phi/Nch)
            t[syn] = np.multiply(
                phi2, refract[syn]
            ).argmax()  # extract refractory period from phi end get the time of the maximum correlation
            corr[syn] = phi[t[syn]]
            corr2[syn] = phi2[t[syn]]

        # select synergy and delay with highest correlation for the s-th episode
        # count_each_I=np.array([],dtype=int)
        for each_I in range(Nsyn):
            # np.append(count_each_I,np.count_nonzero(I==each_I))
            if np.count_nonzero(I == each_I) == N_trial:
                corr2[each_I] = 0

        Hsyn = corr2.argmax()  # get the synergy with highest correlation
        Ht = (
            int(t[Hsyn] - Ssample) * sampleStep + 1
        )  # Get the delay for the highest correlation
        Hc = corr[Hsyn] / Nch  # get the highest correlation

        # Update Refract Period
        if np.isnan(Hc):
            print("Hcはnanになりました", Hc)
            time.sleep(2)
            break

        center = Ht + Qsample / 2
        init = center - numSamples / 2
        end = init + numSamples

        C = np.zeros(numSamples.size)
        ReHt = np.zeros(numSamples.size)
        for j in range(numSamples.size):
            # print(numSamples[j])
            # if not (int(end[j])-int(init[j]))==numSamples[j]:
            #        end[j] +=1
            while end[j] - init[j] > numSamples[j]:
                end[j] += -1
            while end[j] - init[j] < numSamples[j]:
                end[j] += 1
            if end[j] > Ksample or init[j] < 0:
                print("break")
                break

            A = interpFunction[Hsyn](synInterval[j])[
                :,
                np.maximum(-int(init[j]), 0) : np.minimum(
                    numSamples[j], Ksample - int(init[j])
                ),
            ]
            B = data[:, np.maximum(int(init[j]), 0) : np.minimum(int(end[j]), Ksample)]

            Hc_retry = []
            ReT_each = []
            for Re in range(0, int(Qsample / 2), sampleStep):
                ReT = Re - Qsample / 2
                New_init = int(init[j] + ReT)
                New_end = int(end[j] + ReT)
                while New_end - New_init > numSamples[j]:
                    New_end += -1
                while New_end - New_init < numSamples[j]:
                    New_end += 1
                if New_end > Ksample or New_init < 0:
                    print("break")
                    break
                B = data[:, np.maximum(New_init, 0) : np.minimum(New_end, Ksample)]
                # Hc_retry.append(sum([np.dot(A[ch,:],B[ch,:]) for ch in range(Nch)])/numSamples[j])
                Hc_retry.append(
                    sum([distance.correlation(A[ch, :], B[ch, :]) for ch in range(Nch)])
                )
                ReT_each.append(ReT)
            if len(Hc_retry) == 0:
                C[j] = 0
                ReHt[j] = 0
            else:
                C[j] = np.min(Hc_retry)
                ReHt[j] = ReT_each[np.argmin(Hc_retry)]
        Hs = numSamples[C.argmin()]
        Ht = Ht + ReHt[C.argmin()]

        # refractWindow = Ssample
        refractWindow = round(Hs)
        # print("Ssample=",refractWindow)
        # time.sleep(0.5)
        tInit = int(t[Hsyn] - refractWindow / 2 + 1)
        tEnd = tInit + refractWindow - 2
        refractPeriod = np.arange(
            np.maximum(tInit, 0), np.minimum(tEnd, phiSize)
        ).astype(int)
        refract[Hsyn, refractPeriod] *= 0

        # Append new parameters
        I = np.append(I, Hsyn)
        T = np.append(T, Ht)
        S = np.append(S, Hs)

        # mutiply the selected element by its correlation and subtract from the data
        Wshifted = shiftAndScale(W[:, kk[Hsyn]], Ht, Hs, Ksample)
        residual = residual - (Wshifted * (Hc + 1))

        residual[residual < 0.0] = 0.0  # set to zero any negative value
    print(Ssample)
    print(how_many_trial * Nsyn)
    print("今何回回ったか", countI)
    print("residual平均", np.mean(residual))
    print("Synergies/Trial: ", I.size / 32)
    print("計算される推定Trial回数", I.size / Nsyn)
    print("Final Correlation: ", Hc)

    C = updateC(data, W, I, T, S, Nsyn, Qsample, Ksample)

    return I, T, S, C


"""
Back-Projection
"""


def updateC(data, W, I, T, S, Nsyn, Qsample, Ksample):
    ####Edition For Continuous Data##########
    Ninst = I.size
    ########################################
    C = np.zeros((Ninst, 1))
    nch = W.shape[0]

    i = np.arange(Ksample)
    Rf = data[:, i].flatten()[:, np.newaxis]

    g = np.zeros((Ninst, nch * Ksample))
    kk = []
    for syn in range(Nsyn):  # Call np.arange Nsyn times instead of Ninst times
        kk.append(syn * Qsample + np.arange(Qsample))
        # kk.append(slice(syn*Qsample,(syn+1)*Qsample))

    for j in range(Ninst):
        syn = I[j]
        k = kk[syn]
        g[j, :] = shiftAndScale(W[:, k], T[j], S[j], Ksample).flatten()[np.newaxis, :]

    G = np.dot(g, np.transpose(g))  # <g,g>
    Y = np.dot(g, Rf)  # <g,Rf> = <g,xg> = x<g,g>

    # conjugate gradient
    X, info = cg(G, Y, x0=np.zeros((Ninst, 1)))  # Y=GX
    C[:, 0] = X

    C[C < 0.0] = 0.0  # we enforce non-negativity by setting to zero any negative value
    return C


"""
Time-shift channels (ch) and reshape to the size of the data (Ksample)
n can be negative (>-Qsample), 0, and positive (<Ksample+Qsample)
"""


def shiftChannels(ch, n, Ksample):
    Nch, Qsample = ch.shape
    d = np.zeros((Nch, Ksample))

    if n < -Qsample or n > (Ksample + Qsample):
        warnings.warn(
            "Time delay out of range! t = %i, range = (%i,%i)"
            % (n, -Qsample, Ksample + Qsample),
            stacklevel=2,
        )
        return d

    d[:, np.maximum(n, 0) : np.minimum(n + Qsample, Ksample)] = ch[
        :, np.maximum(-n, 0) : np.minimum(Qsample, Ksample - n)
    ]

    return d


"""
Time-shift, scale channels (ch) and reshape to the size of the data (Ksample)
t can be negative (>-Qsample), 0, and positive (<Ksample+Qsample)
"""


def shiftAndScale(syn, t, s, Ksample):
    Nch, Qsample = syn.shape
    d = np.zeros((Nch, Ksample))

    if t < -Qsample or t > (Ksample + Qsample):
        warnings.warn(
            "Time delay out of range! t = %i, range = (%i,%i)"
            % (t, -Qsample, Ksample + Qsample),
            stacklevel=2,
        )
        return d

    synInterval = np.arange(Qsample)
    synScaledInterval = np.linspace(0, Qsample - 1, s)
    interpFunction = interpolate.interp1d(synInterval, syn)

    center = t + Qsample / 2
    init = center - s / 2
    end = init + s

    if not (int(end) - int(init)) == s:
        end += 1

    synScaled = interpFunction(synScaledInterval)[
        :, np.maximum(-int(init), 0) : np.minimum(int(s), Ksample - int(init))
    ]

    d[:, np.maximum(int(init), 0) : np.minimum(int(end), Ksample)] = synScaled
    # d[:,np.maximum(t,0):np.minimum(t+Qsample,Ksample)] = syn[:,np.maximum(-t,0):np.minimum(Qsample,Ksample-t)]

    return d


"""
    This function time-shift each synergy (Q samples) for each trial (K samples).
    Each synergy needs a shift matrix of size [Qsample,Ksample]. Since there are 
    Nsyn synergies and Nsect trials, the final matrix has size [Nsyn*Qsample,Nsect*Ksample].
"""


def getShiftedMatrix(I, T, C, Nsyn, Qsample, Ksample):
    ####Edition For Continuous Data##########
    Ninst = I.size
    ########################################
    H = np.zeros((Nsyn * Qsample, Ksample))
    # i,j = np.indices((Qsample,Ksample))

    ii = []
    for syn in range(Nsyn):  # Call np.arange Nsyn times instead of Ninst times
        # ii.append(slice(syn*Qsample,(syn+1)*Qsample))
        ii.append(np.arange(syn * Qsample, (syn + 1) * Qsample))
    # jj = [slice(0,Ksample)] #Call np.arange 1 time

    for inst in range(Ninst):
        syn = I[inst]

        # create the shift matrix S[i=j-t] = c for each synergy of each trial
        # where c is the non-negative scalar coeficient of each synergy of each trial
        """
        @Felipe
        i,i = np.indices((Qsample,Ksample)) creates 2*Qsample*Ksample indices,
        while the following operation creates less than 2*Qsample indices
        """
        jShift = np.array(
            [
                i + T[inst]
                for i in range(Qsample)
                if (i + T[inst] < Ksample and i + T[inst] >= 0)
            ],
            dtype=int,
        )
        iShift = jShift - T[inst]
        H[ii[syn][iShift], jShift] += C[inst, 0]
        # S[ii[syn],jj[0]] += (i==j-T[inst])*C[inst,0]
    return H


def getLinearCombination(W, I, T, S, C, Nsyn, Qsample, Ksample):
    ####Edition For Continuous Data##########
    Ninst = I.size
    ########################################
    Nch = W.shape[0]

    kk = []
    for syn in range(Nsyn):  # Call np.arange Nsyn times instead of Ninst times
        kk.append(syn * Qsample + np.arange(Qsample))
        # kk.append(slice(syn*Qsample,(syn+1)*Qsample))

    Vhat = np.zeros((Nch, Ksample))
    synInterval = np.arange(Qsample)
    for j in range(Ninst):
        syn = I[j]
        k = kk[syn]

        # Vhat += C[j] * shiftAndScale(W[:,k],T[j],S[j],Ksample)

        synScaledInterval = np.linspace(0, Qsample - 1, S[j])
        interpFunction = interpolate.interp1d(synInterval, W[:, k])

        center = T[j] + Qsample / 2
        init = center - S[j] / 2
        end = init + S[j]
        if not (int(end) - int(init)) == S[j]:
            end += 1

        # Make sure the synergy is inside the interval [0,Ksample]
        synScaled = interpFunction(synScaledInterval)[
            :, np.maximum(-int(init), 0) : np.minimum(int(S[j]), Ksample - int(init))
        ]

        # Linear Combination
        Vhat[:, np.maximum(int(init), 0) : np.minimum(int(end), Ksample)] += (
            C[j] * synScaled
        )

    return Vhat


"""
@Felipe
Alternative version of data*H' that does not use matrices
It allows use time-scalable synergies
"""


def getSumOfInstances(data, I, T, S, C, Nsyn, Qsample, Ksample):
    Ninst = I.size
    Nch = data.shape[0]
    SoI = np.zeros((Nch, Nsyn * Qsample))  # Shape of W

    kk = []
    for syn in range(Nsyn):  # Call np.arange Nsyn times instead of Ninst times
        kk.append(syn * Qsample + np.arange(Qsample))
        # kk.append(slice(syn*Qsample,(syn+1)*Qsample))

    synInterval = np.arange(Qsample)
    for j in range(Ninst):
        syn = I[j]
        k = kk[syn]

        center = T[j] + Qsample / 2
        init = center - S[j] / 2
        end = init + S[j]

        # Extract the instance from the data
        instance = np.zeros((Nch, S[j]))
        instance[
            :, np.maximum(-int(init), 0) : S[j] + np.minimum(Ksample - int(end), 0)
        ] = data[:, np.maximum(int(init), 0) : np.minimum(int(end), Ksample)]
        # instance= data[:,np.maximum(int(init),0):np.minimum(int(end),Ksample)]

        instanceInterval = np.linspace(0, Qsample - 1, S[j])
        interpFunction = interpolate.interp1d(instanceInterval, instance)

        # Change the number of samples to Qsample
        notScaledInstance = interpFunction(synInterval)
        # Multiply by the coefficient of the instance
        SoI[:, k] += C[j] * notScaledInstance

    return SoI
