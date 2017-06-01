//Logging Library
var log = require('winston');
var GestureTool = require("./GestureTool");
var spawn = require("child_process").spawn;
var os = require("os");

var args = process.argv.slice(2);
var proc, pyready = false;

GestureTool.init();
GestureTool.on("ready", function() {
    if(!args[0]) return;
    var path = args[0].endsWith(".py")?args[0]:(args[0]+".py");
    log.info("Starting python script " + path);

    proc = spawn('python' , ['-u', path]);
    proc.stdout.on('data', function (data){
        if (data.toString('utf-8').indexOf("ready") === 0) {
            pyready = true;
            log.info("Python scripts is ready!")
        }
    });
    proc.on('error', (err) => {
        console.log('Failed to start python.');
        process.exit(1);
    });
    proc.on('exit', function(code){
        if (code !== 0) {
            console.info("Python returned with an error: " + code);
        }
        console.info("Python finished");
        process.exit(code);
    });
});
GestureTool.on("data", function(data) {
    if(proc && pyready) {
        proc.stdin.write(data.getWorldX() + " " + data.getWorldY() + " " + data.getWorldZ() + " " + data.getAlpha() + " " + data.getBeta() + " " + data.getGamma() + os.EOL);
    }
});
