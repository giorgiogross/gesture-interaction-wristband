import threading
import sys


# Start listening asynchronously on stdin. Instantiation spawns a new thread which notifies the callback whenever a new
# line is available. The line is passed as an argument to the callback
class StdinReader(threading.Thread):
# todo synchronize keepReading variable
    def __init__(self, threadID, name, callback):
        threading.Thread.__init__(self)

        self.threadID = threadID
        self.name = name

        self.callback = callback
        self.keepReading = True

    def run(self):
        while self.keepReading:
            self.callback(sys.stdin.readline())
        sys.exit(0)

    def end_reading(self):
        self.keepReading = False
