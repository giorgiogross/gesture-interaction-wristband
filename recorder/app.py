import sys
import os.path
sys.path.append(os.path.abspath(__file__ + "/../.."))
from input.processor import Processor
from input.reader import AsyncReader
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
    sys.exit(0)


# handle data input of the sensor
dataProcessor = Processor.DataProcessor()
ar = AsyncReader.StdinReader(0, "reader", process_input)
ar.start()

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
    print "Closing app through terminal."
    on_closing()
