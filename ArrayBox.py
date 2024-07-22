"""
 find position box depending on size
 
   sbox = SizBox(outbox,sizHe,sizWi,irow,icol,Sp)
"""

import warnings
from PositionBox import *


def ArrayBox(outbox, nrow, ncol, row, col, BSp=None):
    if BSp is None:
        BSp = np.array([0.01])

    if BSp.size > 1:
        SpH = BSp[0]
        SpV = BSp[1]
    else:
        SpH = BSp[0]
        SpV = BSp[0]

    BH = (1 - (nrow + 1) * SpV) / nrow
    BW = (1 - (ncol + 1) * SpH) / ncol

    pos = np.array([0.0, 0.0, 0.0, 0.0])

    pos[0] = SpH + col * (BW + SpH)  # python starts from 0
    pos[1] = 1 - (row + 1) * (BH + SpV)  # python starts from 0
    pos[2] = BW
    pos[3] = BH
    box = PositionBox(outbox, pos)
    return box
