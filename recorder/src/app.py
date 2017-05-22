from reader import AsyncReader
from processor import Processor
import Tkinter


def process_input(in_stream):
    dataProcessor.put_raw(in_stream)


def start_recording():
    dataProcessor.record(int(gestureIdEntry.get()))


# handle data input of the sensor
dataProcessor = Processor.DataProcessor()
ar = AsyncReader.StdinReader(0, "reader", process_input)
ar.start()


dataProcessor.get_flat_buffer()

# show UI
top = Tkinter.Tk()
startButton = Tkinter.Button(top, text="GO!", command=start_recording)
startButton.pack(side=Tkinter.LEFT)
gestureIdEntry = Tkinter.Entry(top)
gestureIdEntry.pack(side=Tkinter.RIGHT)
top.mainloop()
