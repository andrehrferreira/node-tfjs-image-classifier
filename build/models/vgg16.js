"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _Optimizer = _interopRequireDefault(require("../Optimizer"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }

function _createClass(Constructor, protoProps, staticProps) { if (protoProps) _defineProperties(Constructor.prototype, protoProps); if (staticProps) _defineProperties(Constructor, staticProps); return Constructor; }

var tf = process.env.gpu === true ? require("@tensorflow/tfjs-node-gpu") : require("@tensorflow/tfjs-node");
var DefaultOptions = {
  inputWidth: 224,
  inputHeight: 224,
  inputChannels: 3,
  learningRate: 0.009,
  optimizer: "sgd"
};

var VGG16Model = /*#__PURE__*/function () {
  function VGG16Model(options) {
    _classCallCheck(this, VGG16Model);

    this.options = options || DefaultOptions;
    this.inputWidth = options.inputWidth || DefaultOptions.inputWidth;
    this.inputHeight = options.inputHeight || DefaultOptions.inputHeight;
    this.inputChannels = options.inputChannels || DefaultOptions.inputChannels;
    this.learningRate = options.learningRate || DefaultOptions.learningRate;
  }

  _createClass(VGG16Model, [{
    key: "generate",
    value: function generate(outputSize) {
      //Input
      this.model = tf.sequential();
      this.model.add(tf.layers.zeroPadding2d({
        inputShape: [this.inputHeight, this.inputWidth, this.inputChannels],
        padding: 1
      }));
      this.model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        activation: "relu"
      }));
      this.model.add(tf.layers.zeroPadding2d({
        padding: 1
      }));
      this.model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        activation: "relu"
      }));
      this.model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2]
      }));
      this.model.add(tf.layers.zeroPadding2d({
        padding: 1
      }));
      this.model.add(tf.layers.conv2d({
        filters: 128,
        kernelSize: 3,
        activation: "relu"
      }));
      this.model.add(tf.layers.zeroPadding2d({
        padding: 1
      }));
      this.model.add(tf.layers.conv2d({
        filters: 128,
        kernelSize: 3,
        activation: "relu"
      }));
      this.model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2]
      }));
      this.model.add(tf.layers.zeroPadding2d({
        padding: 1
      }));
      this.model.add(tf.layers.conv2d({
        filters: 256,
        kernelSize: 3,
        activation: "relu"
      }));
      this.model.add(tf.layers.zeroPadding2d({
        padding: 1
      }));
      this.model.add(tf.layers.conv2d({
        filters: 256,
        kernelSize: 3,
        activation: "relu"
      }));
      this.model.add(tf.layers.zeroPadding2d({
        padding: 1
      }));
      this.model.add(tf.layers.conv2d({
        filters: 256,
        kernelSize: 3,
        activation: "relu"
      }));
      this.model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2]
      }));
      this.model.add(tf.layers.zeroPadding2d({
        padding: 1
      }));
      this.model.add(tf.layers.conv2d({
        filters: 512,
        kernelSize: 3,
        activation: "relu"
      }));
      this.model.add(tf.layers.zeroPadding2d({
        padding: 1
      }));
      this.model.add(tf.layers.conv2d({
        filters: 512,
        kernelSize: 3,
        activation: "relu"
      }));
      this.model.add(tf.layers.zeroPadding2d({
        padding: 1
      }));
      this.model.add(tf.layers.conv2d({
        filters: 512,
        kernelSize: 3,
        activation: "relu"
      }));
      this.model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2]
      }));
      this.model.add(tf.layers.zeroPadding2d({
        padding: 1
      }));
      this.model.add(tf.layers.conv2d({
        filters: 512,
        kernelSize: 3,
        activation: "relu"
      }));
      this.model.add(tf.layers.zeroPadding2d({
        padding: 1
      }));
      this.model.add(tf.layers.conv2d({
        filters: 512,
        kernelSize: 3,
        activation: "relu"
      }));
      this.model.add(tf.layers.zeroPadding2d({
        padding: 1
      }));
      this.model.add(tf.layers.conv2d({
        filters: 512,
        kernelSize: 3,
        activation: "relu"
      }));
      this.model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2]
      })); //Output

      this.model.add(tf.layers.flatten());
      this.model.add(tf.layers.dense({
        units: 4096,
        activation: "relu"
      }));
      this.model.add(tf.layers.dropout({
        rate: 0.5
      }));
      this.model.add(tf.layers.dense({
        units: 4096,
        activation: "relu"
      }));
      this.model.add(tf.layers.dropout({
        rate: 0.5
      }));
      this.model.add(tf.layers.dense({
        units: 1000,
        activation: "relu"
      }));
      this.model.add(tf.layers.dropout({
        rate: 0.5
      }));
      this.model.add(tf.layers.dense({
        units: outputSize,
        activation: "softmax"
      }));
      var optimzer = new _Optimizer["default"](this.options);
      this.model.compile({
        optimizer: optimzer.getOptimzer(),
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
      });
      return tf.model({
        inputs: this.model.layers[0].input,
        outputs: this.model.layers[1].output
      });
    }
  }]);

  return VGG16Model;
}();

var _default = VGG16Model;
exports["default"] = _default;
//# sourceMappingURL=vgg16.js.map