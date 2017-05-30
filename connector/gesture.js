//Logging Library
var log = require('winston');
var GestureTool = require("./GestureTool");
var PythonShell = require("python-shell");

var args = process.argv.slice(2);
var pyshell;

GestureTool.init();
GestureTool.on("ready", function() {
    if(!args[0]) return;
    var path = args[0].endsWith(".py")?args[0]:(args[0]+".py");
    log.info("Starting python script " + path);
    pyshell = PythonShell.run(path, function (err) {
        if (err) throw err;
        console.log('Script started');
    });
});
GestureTool.on("data", function(data) {
    log.info("angle alpha: " + data.getAlpha(), "beta: " + data.getBeta(), "gamma: "+ data.getGamma());
    log.info("local x: " + data.getX(), "y: " + data.getY(), "z: "+ data.getZ());
    log.info("world x: " + data.getWorldX(), "y: " + data.getWorldY(), "z: "+ data.getWorldZ());
    if(pyshell) pyshell.send(data.getWorldX() + " " + data.getWorldY() + " " + data.getWorldZ() + " " + data.getAlpha() + " " + data.getBeta() + " " + data.getGamma());
});
