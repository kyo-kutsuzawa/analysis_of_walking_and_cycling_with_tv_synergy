"""
 Traces: class to plot multi-channel data

 (c) Felipe Moreira Ramos - Tohoku University, Sendai, Japan -
 

 Date: 20180910
"""

import numpy as np
import matplotlib.pyplot as plt


class Traces:
    class Prop:
        color = None
        linewidth = None

    class Opt:
        autoscale = None
        autotrange = None
        fill = None
        fillpar = None
        chcolor = None
        baseline = None
        baselinestyle = None
        box = None
        xlim = None
        yscale = None
        yscalelabel = None
        ytick = None
        ylim = None

    data = np.array([])  # [nch,ssamp]
    time = np.array([])  # [1,nsamp]
    chlabels = np.array([])
    label = ""  # traces label
    dataunits = ""
    timeunits = "s"
    prop = None
    opt = None

    """
    t = Traces(data,time)
    t = Traces(data,time,chlabels)
    """

    def __init__(self, data, time, chlabels=None):
        self.prop = self.Prop()
        self.opt = self.Opt()

        self.data = data
        if time.size == data.shape[1]:
            self.time = time
        else:
            self.time = np.array([i for i in range(data.shape[1])])

        if not chlabels is None:
            if chlabels.size == data.shape[0]:
                labels = []
                for i in range(data.shape[0]):
                    if isinstance(chlabels[i], str):
                        labels.append(chlabels[i])
                    else:
                        labels.append("ch %02i" % (i + 1))
            else:
                labels = ["ch %02i" % (i + 1) for i in range(data.shape[0])]
        else:
            labels = ["ch %02i" % (i + 1) for i in range(data.shape[0])]

        self.chlabels = np.array(labels)
        self.label = ""
        self.dataunits = ""
        self.timeunits = "s"
        self.setdefaults()

    """
    set default prop and opt
    """

    def setdefaults(self):
        self.prop.color = "black"
        self.prop.linewidth = 0.5

        self.opt.autoscale = 1
        self.opt.autotrange = 1
        self.opt.fill = 0
        self.opt.baseline = 0
        self.opt.baselinestyle = ":"
        self.opt.box = 0
        self.opt.yscale = 0
        self.opt.yscalelabel = ""
        self.opt.ytick = 0

    """
    plot traces object
    """

    def plot(self, ind, h=None):
        if h is None:
            h = plt.gca()

        tlines = h.get_lines()
        # plot parameters
        # fillpar = .02 #hsv format -> luminance change

        nch = self.data.shape[0]

        if self.opt.autoscale:
            tmin, tmax = self.range()
        else:
            tmin = self.opt.ylim[0]
            tmax = self.opt.ylim[1]
        dt = tmax - tmin
        if self.opt.autotrange:
            trmin, trmax = self.trange()
        else:
            trmin = self.opt.xlim[0]
            trmax = self.opt.xlim[1]

        for j in range(nch):  # python starts from 0
            h.plot(
                self.time,
                self.data[j] + dt * (nch - (j + 1)),
                linewidth=self.prop.linewidth,
                color=self.prop.color,
            )
            if self.opt.fill:
                time = np.array([self.time[0]])
                time = np.append(time, self.time)
                time = np.append(time, [self.time[-1]])
                data = np.array([0])
                data = np.append(data, self.data[j])
                data = np.append(data, [0])
                h.fill(
                    time, data + dt * (nch - (j + 1)), color=self.prop.color, alpha=0.25
                )  # color=(.75,.75,.75))
            if self.opt.baseline:
                h.plot(
                    [trmin, trmax],
                    [dt * (nch - (j + 1)), dt * (nch - (j + 1))],
                    linewidth=self.prop.linewidth,
                    linestyle=self.opt.baselinestyle,
                    color=self.prop.color,
                )

        if self.opt.chcolor:
            print("Channel Color not implemented")
            if self.opt.fill:
                print("Fill not implemented")
        else:
            if self.opt.fillpar:
                print("Fillpar not implemented")
            # if self.opt.fill:
            #    print('Fill not implemented')

        if not tlines:  # first traces being plotted in axes
            h.set_xlim((trmin, trmax))
            h.set_ylim((tmin, dt * (nch - 1) + tmax))
            if ind == 0:
                h.set_yticks(dt * range(nch))
                h.set_yticklabels(np.flipud(self.chlabels))
                h.tick_params(
                    axis="y",  # changes apply to the y-axis
                    which="both",  # both major and minor ticks are affected
                    left=False,  # ticks along the left edge are off
                    right=False,
                )  # ticks along the right edge are off
            else:
                h.tick_params(
                    axis="y",  # changes apply to the y-axis
                    which="both",  # both major and minor ticks are affected
                    left=False,  # ticks along the left edge are off
                    right=False,  # ticks along the right edge are off
                    labelleft=False,
                )  # labels along the right edge are off

            h.set_xlabel("Time (%s)" % (self.timeunits))
            h.set_title(self.label)
            if not self.opt.box:
                # plt.box(False)
                # h.set_frame_on(False)
                h.spines["top"].set_visible(False)
                h.spines["right"].set_visible(False)
                h.spines["bottom"].set_visible(False)
                h.spines["left"].set_visible(False)
            if self.opt.yscale:
                print("Y scale not implemented")
            if self.opt.ytick:
                print("Y tick not implemented")

    """
    returns min and max over all channels
    """

    def range(self):
        valmin = self.data.min()
        valmax = self.data.max()

        return valmin, valmax

    """
    returns time range of trace
    """

    def trange(self):
        tmin = self.time.min()
        tmax = self.time.max()

        return tmin, tmax
