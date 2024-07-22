"""
 Bars: class to plot multiple bar plots

 (c) Felipe Moreira Ramos - Tohoku University, Sendai, Japan -
 

 Date: 20180910
"""

import numpy as np
import matplotlib.pyplot as plt


class Bars:
    class Prop:
        color = None

    class Opt:
        type = None
        color = None
        chcolor = None
        posaxes = None

    data = np.array([])  # [nch,nsamp]
    chlabels = []  # [nch]
    label = []  # [n]
    dataunits = ""
    prop = None
    opt = None

    """
    t = Bars(data)
    t = Bars(data,chlabels)
    t = Bars(data,chlabels,labels)
    """

    def __init__(self, data, chlabels=None, labels=None):
        self.prop = self.Prop()
        self.opt = self.Opt()

        # assume data matrix
        self.data = data
        if not chlabels is None:
            self.chlabels = chlabels
        else:
            self.chlabels = ["ch %02i" % (i + 1) for i in range(data.shape[0])]
        if not labels is None:
            self.labels = labels
        else:
            self.labels = ["%02i" % (i + 1) for i in range(data.shape[1])]

        self.dataunits = ""
        self.setdefaults()

    """
    set default prop and opt
    """

    def setdefaults(self):
        self.prop.color = "black"

        self.opt.type = "barh"  #'bar'
        self.opt.color = []  # column colors
        self.opt.chcolor = []  # channel colors
        self.opt.posaxes = np.array([0.04, 0.02, 0.94, 0.96])  # outer axes position

    """
    plot bars object (only one set)
    """

    def plot(self, h_axes=None):
        W = self.data
        M, N = W.shape

        if h_axes is None:
            # create figure
            print("Create bars not implemented")

        # limits
        maxW = np.max(W)  # all dimensions
        minW = np.min(W)  # all dimensions

        if self.opt.type == "barh":
            for i in range(N):
                ha = h_axes[i]
                hb = ha.barh(np.arange(M), W[:, i], align="center")

                if len(self.opt.color) == N:
                    ha.set_facecolor(self.opt.color[i])
                elif len(self.opt.chcolor) == M:
                    for k in range(M):
                        hb[k].set_facecolor(self.opt.color[k])

                ha.set_xlim((np.fmin(0, minW * 1.1), maxW * 1.1))
                ha.set_ylim((-1, M))
                ha.set_yticks(range(0, M))
                if i == 0:
                    ha.set_yticklabels(self.chlabels)
                    ha.tick_params(
                        axis="y",  # changes apply to the y-axis
                        which="both",  # both major and minor ticks are affected
                        direction="in",
                    )

                else:
                    ha.tick_params(
                        axis="y",  # changes apply to the y-axis
                        which="both",  # both major and minor ticks are affected
                        direction="in",
                        labelleft=False,
                    )  # labels along the right edge are off

                ha.set_xlabel(self.labels[i])
                ha.tick_params(axis="x", which="both", direction="in")

                ha.invert_yaxis()  # labels read top-to-bottom
                # plt.box(False)
                ha.spines["top"].set_visible(False)
                ha.spines["right"].set_visible(False)
                ha.spines["bottom"].set_visible(False)
                ha.spines["left"].set_visible(False)
