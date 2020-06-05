"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }

function _createClass(Constructor, protoProps, staticProps) { if (protoProps) _defineProperties(Constructor.prototype, protoProps); if (staticProps) _defineProperties(Constructor, staticProps); return Constructor; }

var tf = process.env.gpu === true ? require("@tensorflow/tfjs-node-gpu") : require("@tensorflow/tfjs-node");

var Optimizer = /*#__PURE__*/function () {
  function Optimizer(options) {
    _classCallCheck(this, Optimizer);

    var optimizerName = options.optimizer || "adam";
    var learningRate = options.learningRate || 0.001;

    switch (optimizerName) {
      case "adam":
        this.optimize = tf.train.adam(learningRate);
        break;

      case "sgd":
        this.optimize = tf.train.sgd(learningRate);
        break;

      case "rmsprop":
        this.optimize = tf.train.rmsprop(learningRate);
        break;

      default:
        this.optimize = optimizerName;
        break;
    }
  }

  _createClass(Optimizer, [{
    key: "getOptimzer",
    value: function getOptimzer() {
      return this.optimize;
    }
  }]);

  return Optimizer;
}();

var _default = Optimizer;
exports["default"] = _default;
//# sourceMappingURL=Optimizer.js.map