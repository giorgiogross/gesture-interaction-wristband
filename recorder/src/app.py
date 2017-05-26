from reader import AsyncReader
from processor import Processor
import Tkinter


def process_input(in_stream):
    dataProcessor.put_raw(in_stream)


def start_recording():
    try:
        dataProcessor.prepare_recording(int(gestureIdEntry.get()))
    except ValueError:
        print "Not an integer.."


def on_closing():
    top.destroy()
    ar.end_reading()
    ar.join()


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
top.protocol("WM_DELETE_WINDOW", on_closing)
try:
    top.mainloop()
except KeyboardInterrupt:
    on_closing()
    print "Closing app through terminal."
