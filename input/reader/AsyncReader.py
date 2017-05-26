import threading
import sys
import time
import fileinput
import os


# Start listening asynchronously on stdin. Instantiation spawns a new thread which notifies the callback whenever a new
# line is available. The line is passed as an argument to the callback
class StdinReader(threading.Thread):
    def __init__(self, threadID, name, callback):
        threading.Thread.__init__(self)

        self.threadID = threadID
        self.name = name
        # exit when parent process exits:
        self.daemon = True

        self.callback = callback

    def run(self):
        for line in fileinput.input():
            self.callback(line)
        sys.exit(0)

    def end_reading(self):
        # this is not working properly
        sys.stdin.close()
        os.close(0)
