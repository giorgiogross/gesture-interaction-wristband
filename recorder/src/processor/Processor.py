import numpy as np
import re


class DataProcessor:
    n = 20
    m = 6

    def __init__(self):
        self.buffer = np.ndarray(shape=(self.n, self.m), dtype=float)
        self.buffer.fill(0)
        self.arrayPointer = 0

    # insert raw data string into the buffer. Data will be parsed based on x,y,z,alpha,beta,gamma as float values
    def put_raw(self, raw_data):
        self.buffer[self.arrayPointer] = re.findall(r'[-+]?\d*\.\d+|\d+', raw_data)
        self._inc_array_pointer()
        print self.buffer
        print self.arrayPointer

    def _inc_array_pointer(self):
        self.arrayPointer = (self.arrayPointer + 1) % 20
