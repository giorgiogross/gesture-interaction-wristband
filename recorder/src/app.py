from reader import AsyncReader
from processor import Processor


def rec(input):
    dp.put_raw(input)

dp = Processor.DataProcessor()

ar = AsyncReader.StdinReader(0, "reader", rec)
ar.start()
