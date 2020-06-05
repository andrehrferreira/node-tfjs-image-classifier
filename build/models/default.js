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
  learningRate: 0.0001,
  optimizer: "adam"
};

var DefaultModel = /*#__PURE__*/function () {
  function DefaultModel(options) {
    _classCallCheck(this, DefaultModel);

    this.options = options || DefaultOptions;
    this.inputWidth = options.inputWidth || DefaultOptions.inputWidth;
    this.inputHeight = options.inputHeight || DefaultOptions.inputHeight;
    this.inputChannels = options.inputChannels || DefaultOptions.inputChannels;
    this.learningRate = options.learningRate || DefaultOptions.learningRate;
  }

  _createClass(DefaultModel, [{
    key: "generate",
    value: function generate(outputSize) {
      //Input
      this.model = tf.sequential({
        layers: [tf.layers.flatten({
          inputShape: [this.inputHeight, this.inputWidth, this.inputChannels]
        }), tf.layers.dense({
          units: 64,
          activation: "relu",
          kernelInitializer: "varianceScaling",
          useBias: true
        }), tf.layers.dense({
          units: outputSize,
          kernelInitializer: "varianceScaling",
          activation: "softmax",
          useBias: false
        })]
      }); //Output

      var optimzer = new _Optimizer["default"](this.options);
      this.model.compile({
        optimizer: optimzer.getOptimzer(),
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
      });
      return this.model;
    }
  }]);

  return DefaultModel;
}();

var _default = DefaultModel;
exports["default"] = _default;
//# sourceMappingURL=default.js.map