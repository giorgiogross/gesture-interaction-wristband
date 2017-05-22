from reader import AsyncReader


def rec(input):
    print "Received new input: " + input

ar = AsyncReader.StdinReader(0, "reader", rec)
ar.start()
ar.end_reading()
