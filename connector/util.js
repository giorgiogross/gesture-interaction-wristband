module.exports = {
    deg2rad: function(degrees) {
        return degrees/180*Math.PI;
    },
    rad2deg: function(radians) {
        return radians/Math.PI*180;
    },
    sind: function(degrees) {
        return Math.sin(this.deg2rad(degrees));
    },
    cosd: function(degrees) {
        return Math.cos(this.deg2rad(degrees));
    },
    multiply: function(left, ...right) {
        if(right.length > 1) return this.multiply(left, this.multiply(...right));
        right = right[0];
        var rows = left.length;
        var cols = right[0].length;
        var res = [];
        for (var y = 0; y < rows; ++y) {
            res[y] = [];
            for (var x = 0; x < cols; ++x) {
                var sum = 0;
                for (var i = 0; i < right.length; ++i) {
                    sum += left[y][i] * right[i][x];
                }
                res[y][x] = sum;
            }
        }
        return res;
    }
};