import numpy as np
import re


# Handles the input buffer and writes sensor values to a file when recording
class DataProcessor:
    n = 20
    m = 6

    def __init__(self):
        self.buffer = np.ndarray(shape=(self.n, self.m), dtype=float)
        self.buffer.fill(0)
        self.arrayPointer = 0
        self.recState = RecordState()

    # insert raw data string into the buffer. Data will be parsed based on x,y,z,alpha,beta,gamma as float values
    def put_raw(self, raw_data):
        self.buffer[self.arrayPointer] = re.findall(r'[-+]?\d*\.\d+|\d+', raw_data)
        # todo check for recording state and consider writing to file; clean up everything afterwards
        self._inc_array_pointer()

    def _inc_array_pointer(self):
        self.arrayPointer = (self.arrayPointer + 1) % 20

    def record(self, gid):
        self.recState.record(gid, self.arrayPointer)

    def get_flat_buffer(self):
        out_array = np.array([])
        out_array = np.append(out_array, np.roll(self.buffer.flatten(), self.n * self.m - self.arrayPointer))
        out_array = np.append(out_array, self.recState.gesture_id)
        return out_array


class RecordState:

    def __init__(self):
        self.is_recording = False
        self.gesture_id = -1
        self.start_idx = -1

    def reset(self):
        self.is_recording = False
        self.gesture_id = -1
        self.start_idx = -1

    def record(self, gid, start_idx):
        self.gesture_id = gid
        self.is_recording = True
        self.start_idx = start_idx
