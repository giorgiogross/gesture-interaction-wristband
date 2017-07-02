import sys
import os.path
sys.path.append(os.path.abspath(__file__ + "/../.."))
from input.processor import Processor
from input.reader import AsyncReader
from scanner.Gestures import Gestures
from scanner.GestureScanner import GestureScanner
from dashboard.plot import Dashboard
import time
import argparse

# Instantiate the parser
argument_parser = argparse.ArgumentParser(description='')
argument_parser.add_argument('--dashboard', action='store_true',
                    help='Start the dashboard')
args = argument_parser.parse_args()


if args.dashboard:
    dashboard = Dashboard()

def process_input(in_stream):
    global args
    global dashboard

    if not dataProcessor.put_raw(in_stream):
        return

    gesture_id = gestureScanner.check_for_gesture(dataProcessor.get_flat_buffer())

    # todo pass gesture to dashboard
    # todo track when last gesture was done and only pass this one on if there is enough time in between (~1sec)
    if gesture_id == Gestures.SWIPE_LEFT:
        print "SWIPE LEFT recognized"
        # clear the buffer to avoid future gesture triggers
        dataProcessor.clean()
    elif gesture_id == Gestures.SWIPE_RIGHT:
        print "SWIPE RIGHT recognized"
        # clear the buffer to avoid future gesture triggers
        dataProcessor.clean()
    elif gesture_id == Gestures.SWIPE_UP:
        print "SWIPE UP recognized"
        # clear the buffer to avoid future gesture triggers
        dataProcessor.clean()
    elif gesture_id == Gestures.SWIPE_DOWN:
        print "SWIPE DOWN recognized"
        # clear the buffer to avoid future gesture triggers
        dataProcessor.clean()
    elif gesture_id == Gestures.PUNCH:
        print "PUNCH recognized"
        # clear the buffer to avoid future gesture triggers
        dataProcessor.clean()
        
    if args.dashboard:
        dashboard.action(gesture_id)

# init gesture scanning
gestureScanner = GestureScanner("../input/raw/sensor_data.csv", True)

# handle data input of the sensor
dataProcessor = Processor.DataProcessor()
ar = AsyncReader.StdinReader(0, "reader", process_input)
ar.start()

if args.dashboard:
    dashboard.init()
else:
    while True:
        time.sleep(0.1)
