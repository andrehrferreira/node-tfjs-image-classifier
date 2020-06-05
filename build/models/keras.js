"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _kerasJs = _interopRequireDefault(require("keras-js"));

var _Optimizer = _interopRequireDefault(require("../Optimizer"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }

function _createClass(Constructor, protoProps, staticProps) { if (protoProps) _defineProperties(Constructor.prototype, protoProps); if (staticProps) _defineProperties(Constructor, staticProps); return Constructor; }

var tf = process.env.gpu === true ? require("@tensorflow/tfjs-node-gpu") : require("@tensorflow/tfjs-node");
var DefaultOptions = {
  inputWidth: 7,
  inputHeight: 7,
  inputChannels: 1024,
  learningRate: 0.0001,
  optimizer: "adam"
};

var KerasModel = /*#__PURE__*/function () {
  function KerasModel(options) {
    _classCallCheck(this, KerasModel);

    this.options = options || DefaultOptions;
    this.inputWidth = options.inputWidth || DefaultOptions.inputWidth;
    this.inputHeight = options.inputHeight || DefaultOptions.inputHeight;
    this.inputChannels = options.inputChannels || DefaultOptions.inputChannels;
    this.learningRate = options.learningRate || DefaultOptions.learningRate;
  }

  _createClass(KerasModel, [{
    key: "load",
    value: function load(filepath) {
      this.model = new _kerasJs["default"].Model({
        filepath: filepath,
        filesystem: true
      });
    }
  }]);

  return KerasModel;
}();

var _default = KerasModel;
exports["default"] = _default;
//# sourceMappingURL=keras.js.map