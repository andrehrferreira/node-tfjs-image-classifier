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
  inputWidth: 32,
  inputHeight: 32,
  inputChannels: 3,
  learningRate: 0.001,
  optimizer: "adam"
};

var CIFAR10Model = /*#__PURE__*/function () {
  function CIFAR10Model(options) {
    _classCallCheck(this, CIFAR10Model);

    this.options = options || DefaultOptions;
    this.inputWidth = options.inputWidth || DefaultOptions.inputWidth;
    this.inputHeight = options.inputHeight || DefaultOptions.inputHeight;
    this.inputChannels = options.inputChannels || DefaultOptions.inputChannels;
    this.learningRate = options.learningRate || DefaultOptions.learningRate;
  }

  _createClass(CIFAR10Model, [{
    key: "generate",
    value: function generate(outputSize) {
      //Input
      this.model = tf.sequential();
      this.model.add(tf.layers.conv2d({
        inputShape: [this.inputHeight, this.inputWidth, this.inputChannels],
        kernelSize: 3,
        filters: 32,
        activation: "relu",
        padding: "same"
      }));
      this.model.add(tf.layers.conv2d({
        kernelSize: 3,
        filters: 32,
        activation: "relu"
      }));
      this.model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2]
      }));
      this.model.add(tf.layers.dropout({
        rate: 0.25
      }));
      this.model.add(tf.layers.conv2d({
        kernelSize: 3,
        filters: 64,
        activation: "relu",
        padding: "same"
      }));
      this.model.add(tf.layers.conv2d({
        kernelSize: 3,
        filters: 64,
        activation: "relu"
      }));
      this.model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2]
      }));
      this.model.add(tf.layers.dropout({
        rate: 0.25
      })); //Output

      this.model.add(tf.layers.flatten());
      this.model.add(tf.layers.dense({
        units: 512,
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

  return CIFAR10Model;
}();

var _default = CIFAR10Model;
exports["default"] = _default;
//# sourceMappingURL=cifar10.js.map