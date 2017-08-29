# gesture-interaction-wristband

This project focuses on recognizing hand gestures and interacting through them with a dashboard. We use the [Tunderboard React](http://www.silabs.com/products/development-tools/wireless/bluetooth/thunderboard-react-kit-sensor-cloud-connectivity) by Silicon Labs.

We created a uitility to record gestures in /recorder. These recoded data sets are used to train the machine learning algorithm. We chose a Voting classifier with RandomForest and 5-Neighbors as provided by the scikit-learn packages. Dashboard and gesture recognition are implemented in /recognition. Interfacing the Thunderboard React happens through a NodeJS script. For more details on system architecture please refer to /paper where we described the project and its outcome.

## Installation

Installation on Mac OS:
1. Make sure Xcode, NodeJS and Anaconda (Python 2.7) are installed
2. Go to `/connector`, run `npm install`
3. Run `pip install numpy scikit-learn pydot pandas matplotlib`

## Usage

To launch the recording cd into /connector and execute
```
node gesture.js ../recorder/app.py
```
To launch the recognition along with the dashboard cd into /connector and execute 
```
node gesture.js ../recognition/app.py
```

Note that this project was built upon Python 2.7 and relies on [Anaconda](https://www.continuum.io/what-is-anaconda).
