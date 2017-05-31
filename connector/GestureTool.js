//Bluetooth Library
var noble = require('noble');
//Logging Library
var log = require('winston');
//Event emitting
var events = require('events');
var eventEmitter = new events.EventEmitter();
//Utilities
var Util = require('./util');

var DataPackage = function(x, y, z, alpha, beta, gamma) {
    this.getX = function() { return x; }
    this.getY = function() { return y; }
    this.getZ = function() { return z; }
    this.getAlpha = function() { return alpha; }
    this.getBeta = function() { return beta; }
    this.getGamma = function() { return gamma; }
    var localVector = [[x], [y], [z]];
    var sinus_alpha = Util.sind(alpha),
        sinus_beta = Util.sind(beta),
        sinus_gamma = Util.sind(gamma),
        cosinus_alpha = Util.cosd(alpha),
        cosinus_beta = Util.cosd(beta),
        cosinus_gamma = Util.cosd(gamma);
    var localToReference1 = [
                            [1, 0, 0],
                            [0, cosinus_alpha, -sinus_alpha],
                            [0, sinus_alpha, cosinus_alpha]
                            ];
    var localToReference2 = [
                            [cosinus_beta, 0, sinus_beta],
                            [0, 1, 0],
                            [-sinus_beta, 0, cosinus_beta]
                            ];
    var localToReference3 = [
                            [cosinus_gamma, sinus_gamma, 0],
                            [sinus_gamma, cosinus_gamma, 0],
                            [0, 0, 1]
                            ];
                            
  /*var localToReference = Matrix([
        [cosinus_alpha*cosinus_gamma, -sinus_alpha*cosinus_beta + cosinus_alpha*sinus_gamma*sinus_beta, sinus_alpha*sinus_beta + cosinus_alpha*sinus_gamma*cosinus_beta],
        [sinus_alpha*cosinus_gamma, cosinus_alpha*cosinus_beta + sinus_alpha*sinus_gamma*sinus_beta, -cosinus_alpha*sinus_beta - sinus_alpha*sinus_gamma*cosinus_beta],
        [-sinus_gamma, cosinus_gamma*sinus_beta, cosinus_gamma*cosinus_beta]
    ]);*/
    var localToReference = Util.multiply(localToReference3, localToReference1, localToReference2);
    var worldVector = Util.multiply(localToReference, localVector);
    worldVector[2][0] -= 1.0;
    this.getWorldX = function() {
        return worldVector[0][0];
    }
    this.getWorldY = function() {
        return worldVector[1][0];
    }
    this.getWorldZ = function() {
        return worldVector[2][0];
    }
}
module.exports = GestureTool = (function() {
    var Device, accelerometerInitialized = false, gyroscopeInitialized = false;
    var lastAccelerometerData = null, lastGyroscopeData = null;
    var NobleAPI = {
        stateChange: function(state) {
            if (state === 'poweredOn') {
                log.info("Starting scan for devices...");
                noble.startScanning();
            } else {
                noble.stopScanning();
            }
        },
        discover: function(peripheral) {
            var name = peripheral.advertisement.localName;
            if(!name) return;
            log.info('Found device with local name: ' + name);
            if(name.match(/(thunder|react)/i)) {
                log.info("Stopping scan for devices...");
                noble.stopScanning();
                //Delay the connection, that the scan is successfully ended
                setTimeout(PrivateAPI.connectToDevice.bind(PrivateAPI, peripheral), 2000);
            } else {
                log.info("No Thunderboard detected. Scan remains active.");
            }
        }
    };
    var PrivateAPI = {
        connectToDevice: function(peripheral) {
            log.info("Connecting to " + peripheral.advertisement.localName);
            Device = peripheral;
            Device.connect(PrivateAPI.deviceConnected);
        },
        deviceConnected: function(error) {
            if (error) {
                log.error("Error connecting to the device. Retrying...");
                return setTimeout(PrivateAPI.connectToDevice.bind(PrivateAPI, Device), 2000);
            }
            log.info("Successfully connected to the device.");
            setTimeout(PrivateAPI.discoverServices.bind(PrivateAPI), 1000);
        },
        discoverServices: function() {
            var noServiceFound = setTimeout(function() {
                log.error("Device didn't return it's services. Retrying...");
                PrivateAPI.discoverServices();
            }, 1000);
            Device.discoverServices(null, function(error, services) {
                if(!error) clearTimeout(noServiceFound);
                else {
                    log.error("Device didn't return it's services. Retrying...");
                    return PrivateAPI.discoverServices();
                }
                var service = services.filter(function(service){ return service.uuid == "a4e649f44be511e5885dfeff819cdc9f"; })[0];
                service.discoverCharacteristics(null, function(error, characteristics) {
                    if(error) {
                        log.error("Device didn't return it's characteristics. Retrying...");
                        return PrivateAPI.discoverServices();
                    }
                    var accelerometer = characteristics[0];
                    var gyroscope = characteristics[1];
                    var calibration = characteristics[2];
                    if(calibration) calibration.write(new Buffer([0x01]), true, function(error) {
                        log.info('Calibration done.');
                    });
                    PrivateAPI.initNotify(accelerometer, gyroscope);
                });
                var battery = services.filter(function(service){ return service.uuid == "180f"; })[0];;
                battery.discoverCharacteristics(null, function(error, characteristics) {
                    characteristics[0].read(function(error, data) {
                        console.info(data.readInt8() + "% Battery left");
                    });
                });
            });
        },
        initNotify: function(accelerometer, gyroscope) {
            accelerometer.on('data', function(data, isNotification) {
                PrivateAPI.receiveAccelerometerData(data.readInt16LE(0)/1000, data.readInt16LE(2)/1000, data.readInt16LE(4)/1000);
            });
            var subscribeAccelerometer = function() {
                accelerometer.subscribe(function(error) {
                    if(error) {
                        log.error("Error subscribing to the accelerometer data. Retrying...");
                        return setTimeout(subscribeAccelerometer, 1000);
                    }
                    log.info("Successfully turned on accelerometer notifications.");
                    accelerometerInitialized = true;
                    PrivateAPI.ready();
                });
            }
            subscribeAccelerometer();
            gyroscope.on('data', function(data, isNotification) {
                PrivateAPI.receiveGyroscopeData(data.readInt16LE(0)/100, data.readInt16LE(2)/100, data.readInt16LE(4)/100);
            });
            var subscribeGyroscope = function() {
                gyroscope.subscribe(function(error) {
                    if(error) {
                        log.error("Error subscribing to the gyroscope data. Retrying...");
                        return setTimeout(subscribeGyroscope, 1000);
                    }
                    log.info("Successfully turned on gyroscope notifications.");
                    gyroscopeInitialized = true;
                    PrivateAPI.ready();
                });
            }
            subscribeGyroscope();
        },
        /**
         * Called when the initialization process is fully finished and we recieve all the data.
         */
        ready: function() {
            if(!accelerometerInitialized || !gyroscopeInitialized) return;
            log.info("Initialization process complete.");
            eventEmitter.emit("ready");
        },
        receiveAccelerometerData: function(x, y, z) {
            lastAccelerometerData = {x: x, y: y, z: z};
            if(lastGyroscopeData) PrivateAPI.sendBuffer();
        },
        receiveGyroscopeData: function(alpha, beta, gamma) {
            lastGyroscopeData = {alpha: alpha, beta: beta, gamma: gamma};
            if(lastAccelerometerData) PrivateAPI.sendBuffer();
        },
        sendBuffer: function() {
            if(!lastAccelerometerData || !lastGyroscopeData) return;
            eventEmitter.emit("data", new DataPackage(lastAccelerometerData.x, lastAccelerometerData.y, lastAccelerometerData.z,
                                                        lastGyroscopeData.alpha, lastGyroscopeData.beta, lastGyroscopeData.gamma));
            lastAccelerometerData = lastGyroscopeData = null;
        }
    };
    //Public API
    return {
        init: function() {
            noble.stopScanning();
            noble.on('stateChange', NobleAPI.stateChange);
            noble.on('discover', NobleAPI.discover);
        }, on: eventEmitter.on.bind(eventEmitter)
    };
})();