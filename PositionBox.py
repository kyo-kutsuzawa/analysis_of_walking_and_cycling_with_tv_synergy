import numpy as np


def PositionBox(outbox, inbox):
    box = np.array([0.0, 0.0, 0.0, 0.0])
    box[0] = outbox[0] + outbox[2] * inbox[0]
    box[1] = outbox[1] + outbox[3] * inbox[1]
    box[2] = outbox[2] * inbox[2]
    box[3] = outbox[3] * inbox[3]
    return box
