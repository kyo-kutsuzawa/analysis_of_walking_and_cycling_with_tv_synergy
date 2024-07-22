"""
 find position box depending on size
 
   sbox = SizBox(outbox,sizHe,sizWi,irow,icol,Sp)
"""

import warnings
from PositionBox import *


def SizBox(outbox, sizeH, sizeW, irow, icol, Sp=None):
    if Sp is None:
        Sp = np.array([0.05])
    nrow = sizeH.shape[0]
    ncol = sizeW.shape[1]

    if Sp.size > 1:
        SpH = Sp[0]
        SpV = Sp[1]
    else:
        SpH = Sp[0]
        SpV = Sp[0]

    maxH = sizeH.max(1)
    maxW = sizeW.max(0)
    maxH = maxH / np.sum(maxH)
    maxW = maxW / np.sum(maxW)
    BH = (1 - (nrow + 1) * SpV) * maxH[irow, 0]
    BW = (1 - (ncol + 1) * SpH) * maxW[0, icol]
    if BH < 0:
        warnings.warn("Too many rows!", stacklevel=2)
        BH = 1 / nrow
    if BW < 0:
        warnings.warn("Too many columns!", stacklevel=2)
        BW = 1 / ncol

    cumH = (1 - (nrow + 1) * SpV) * np.cumsum(maxH)
    cumW = (1 - (ncol + 1) * SpH) * np.cumsum(maxW)

    pos = np.array([0.0, 0.0, 0.0, 0.0])
    if icol > 0:
        pos[0] = (icol + 1) * SpH + cumW[0, icol - 1]
        # python starts from 0
    else:
        pos[0] = SpH
    pos[1] = 1 - (irow + 1) * SpV - cumH[irow, 0]
    # python starts from 0
    pos[2] = BW
    pos[3] = BH
    sbox = PositionBox(outbox, pos)
    return sbox
