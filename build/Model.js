"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _path = _interopRequireDefault(require("path"));

var _fsExtra = _interopRequireDefault(require("fs-extra"));

var _fs = _interopRequireDefault(require("fs"));

var _default2 = _interopRequireDefault(require("./models/default"));

var _mnist = _interopRequireDefault(require("./models/mnist"));

var _cifar = _interopRequireDefault(require("./models/cifar10"));

var _vgg = _interopRequireDefault(require("./models/vgg16"));

var _keras = _interopRequireDefault(require("./models/keras"));

var _Data = _interopRequireDefault(require("./Data"));

var _crypto = require("crypto");

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

function asyncGeneratorStep(gen, resolve, reject, _next, _throw, key, arg) { try { var info = gen[key](arg); var value = info.value; } catch (error) { reject(error); return; } if (info.done) { resolve(value); } else { Promise.resolve(value).then(_next, _throw); } }

function _asyncToGenerator(fn) { return function () { var self = this, args = arguments; return new Promise(function (resolve, reject) { var gen = fn.apply(self, args); function _next(value) { asyncGeneratorStep(gen, resolve, reject, _next, _throw, "next", value); } function _throw(err) { asyncGeneratorStep(gen, resolve, reject, _next, _throw, "throw", err); } _next(undefined); }); }; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }

function _createClass(Constructor, protoProps, staticProps) { if (protoProps) _defineProperties(Constructor.prototype, protoProps); if (staticProps) _defineProperties(Constructor, staticProps); return Constructor; }

var tf = process.env.gpu === true ? require("@tensorflow/tfjs-node-gpu") : require("@tensorflow/tfjs-node");
var DefaultOptions = {
  inputWidth: 7,
  inputHeight: 7,
  inputChannels: 1024,
  learningRate: 0.0001,
  optimizer: "adam",
  denseUnits: 100,
  epochs: 200,
  batchSizeFraction: 0.2
};

var TFJSModel = /*#__PURE__*/function () {
  function TFJSModel(options) {
    _classCallCheck(this, TFJSModel);

    this.currentModelPath = null;
    this.options = options || DefaultOptions;
    this.type = options.type || "default"; // default, cifar10, mnist, vgg16, keras

    this.denseUnits = options.denseUnits || DefaultOptions.denseUnits;
    this.epochs = options.epochs || DefaultOptions.epochs;
    this.batchSizeFraction = options.batchSizeFraction || DefaultOptions.batchSizeFraction;
    this.outputSize = null;
    this.data = new _Data["default"](options);
    /* eslint-disable */

    switch (this.type) {
      case "default":
        this.modelClass = new _default2["default"](options);
        break;

      case "mnist":
        this.modelClass = new _mnist["default"](options);
        break;

      case "cifar10":
        this.modelClass = new _cifar["default"](options);
        break;

      case "vgg16":
        this.modelClass = new _vgg["default"](options);
        break;

      case "keras":
        this.modelClass = new _keras["default"](options);
        break;

      default:
        throw "The informed model is not compatible";
    }
    /* eslint-enable */

  }
  /**
   * Function to input data for training
   * 
   * @param object data 
   * @return void
   */


  _createClass(TFJSModel, [{
    key: "input",
    value: function input(data) {
      if (this.type == "keras") throw "Keras model cannot be generated";
      if (!data.labels || !data.images) throw "There are no valid labels or images in the entry";
      this.data = data;
      this.outputSize = this.data.labels.length;
    }
    /**
     * Function to generate model based on configurations
     * 
     * @return void
     */

  }, {
    key: "generate",
    value: function generate() {
      if (this.type == "keras") throw "Keras model cannot be generated";
      if (!this.outputSize) throw "Error when trying to generate the model for not identifying outputSize";
      this.model = this.modelClass.generate(this.outputSize);
    }
    /**
     * Function to perform data training
     * 
     * @return Promise
     */

  }, {
    key: "train",
    value: function () {
      var _train = _asyncToGenerator( /*#__PURE__*/regeneratorRuntime.mark(function _callee() {
        var loadTrainData, batchSize, shuffledIndices;
        return regeneratorRuntime.wrap(function _callee$(_context) {
          while (1) {
            switch (_context.prev = _context.next) {
              case 0:
                if (!(this.type == "keras")) {
                  _context.next = 2;
                  break;
                }

                throw "Keras type model cannot be trained";

              case 2:
                _context.next = 4;
                return this.data.toTrain(this.model);

              case 4:
                loadTrainData = _context.sent;
                batchSize = Math.floor(loadTrainData.images.shape[0] * this.batchSizeFraction);

                if (batchSize > 0) {
                  _context.next = 8;
                  break;
                }

                throw "Batch size is 0 or NaN. Please choose a non-zero fraction.";

              case 8:
                shuffledIndices = new Int32Array(tf.util.createShuffledIndices(loadTrainData.labels.shape[0]));
                console.time("Training Time");
                return _context.abrupt("return", this.model.fit(loadTrainData.images.gather(shuffledIndices), loadTrainData.labels.gather(shuffledIndices), {
                  batchSize: batchSize,
                  epochs: this.epochs,
                  validationSplit: 0.15,
                  callbacks: tf.node.tensorBoard("/tmp/tf_fit_logs")
                  /*callbacks: {
                      onBatchEnd: async (batch, logs) => {
                          //console.log("Loss: " + logs.loss.toFixed(5));
                      },
                      onTrainEnd: async logs => {
                          console.timeEnd("Training Time");
                      }
                  }*/

                }));

              case 11:
              case "end":
                return _context.stop();
            }
          }
        }, _callee, this);
      }));

      function train() {
        return _train.apply(this, arguments);
      }

      return train;
    }()
    /**
     * Function to load pre-trained model
     * 
     * @param string fileOrDirname
     * @return Promise
     */

  }, {
    key: "load",
    value: function () {
      var _load = _asyncToGenerator( /*#__PURE__*/regeneratorRuntime.mark(function _callee2(fileOrDirname) {
        return regeneratorRuntime.wrap(function _callee2$(_context2) {
          while (1) {
            switch (_context2.prev = _context2.next) {
              case 0:
                if (!(this.type == "keras")) {
                  _context2.next = 4;
                  break;
                }

                this.model = this.modelClass.load(fileOrDirname);
                _context2.next = 11;
                break;

              case 4:
                _context2.next = 6;
                return tf.loadLayersModel("file://" + fileOrDirname + "/model.json");

              case 6:
                this.model = _context2.sent;
                _context2.next = 9;
                return _fsExtra["default"].readJson(_path["default"].join(fileOrDirname, "labels.json")).then(function (obj) {
                  return obj.Labels;
                });

              case 9:
                this.data.labels = _context2.sent;
                this.currentModelPath = fileOrDirname;

              case 11:
              case "end":
                return _context2.stop();
            }
          }
        }, _callee2, this);
      }));

      function load(_x) {
        return _load.apply(this, arguments);
      }

      return load;
    }()
    /**
     * Function to save pre-trained model
     * 
     * @param string dirname
     * @return Promise
     */

  }, {
    key: "save",
    value: function () {
      var _save = _asyncToGenerator( /*#__PURE__*/regeneratorRuntime.mark(function _callee3(dirname) {
        return regeneratorRuntime.wrap(function _callee3$(_context3) {
          while (1) {
            switch (_context3.prev = _context3.next) {
              case 0:
                _fsExtra["default"].ensureDirSync(dirname);

                _context3.next = 3;
                return this.model.save("file://" + dirname);

              case 3:
                _context3.next = 5;
                return _fsExtra["default"].writeJson(_path["default"].join(dirname, "labels.json"), {
                  Labels: this.data.labels
                });

              case 5:
                this.currentModelPath = dirname;

              case 6:
              case "end":
                return _context3.stop();
            }
          }
        }, _callee3, this);
      }));

      function save(_x2) {
        return _save.apply(this, arguments);
      }

      return save;
    }()
    /**
     * Function to perform prediction by file
     * 
     * @param string filename
     * @return Promise
     */

  }, {
    key: "predictFromFile",
    value: function () {
      var _predictFromFile = _asyncToGenerator( /*#__PURE__*/regeneratorRuntime.mark(function _callee4(filename) {
        var bufferImage, _this$model$predict$t, values, indices;

        return regeneratorRuntime.wrap(function _callee4$(_context4) {
          while (1) {
            switch (_context4.prev = _context4.next) {
              case 0:
                if (!(this.type == "keras")) {
                  _context4.next = 15;
                  break;
                }

                _context4.next = 3;
                return this.model.ready();

              case 3:
                _context4.t0 = this.model;
                _context4.t1 = Float32Array;
                _context4.next = 7;
                return _fs["default"].readFileSync(filename);

              case 7:
                _context4.t2 = _context4.sent;
                _context4.t3 = new _context4.t1(_context4.t2);
                _context4.t4 = {
                  input_1: _context4.t3
                };
                _context4.next = 12;
                return _context4.t0.predict.call(_context4.t0, _context4.t4);

              case 12:
                return _context4.abrupt("return", _context4.sent);

              case 15:
                _context4.next = 17;
                return this.data.__fileToTensor(filename);

              case 17:
                bufferImage = _context4.sent;
                _this$model$predict$t = this.model.predict(bufferImage).topk(), values = _this$model$predict$t.values, indices = _this$model$predict$t.indices;
                return _context4.abrupt("return", {
                  label: this.data.labels[indices.dataSync()[0]],
                  confidence: values.dataSync()[0]
                });

              case 20:
              case "end":
                return _context4.stop();
            }
          }
        }, _callee4, this);
      }));

      function predictFromFile(_x3) {
        return _predictFromFile.apply(this, arguments);
      }

      return predictFromFile;
    }()
    /**
     * Buffer prediction function
     * 
     * @param string filename
     * @return Promise
     */

  }, {
    key: "predict",
    value: function () {
      var _predict = _asyncToGenerator( /*#__PURE__*/regeneratorRuntime.mark(function _callee5(buffer) {
        var bufferImage, _this$model$predict$t2, values, indices;

        return regeneratorRuntime.wrap(function _callee5$(_context5) {
          while (1) {
            switch (_context5.prev = _context5.next) {
              case 0:
                if (!(this.type == "keras")) {
                  _context5.next = 8;
                  break;
                }

                _context5.next = 3;
                return this.model.ready();

              case 3:
                _context5.next = 5;
                return this.model.predict({
                  input_1: new Float32Array(buffer)
                });

              case 5:
                return _context5.abrupt("return", _context5.sent);

              case 8:
                _context5.next = 10;
                return this.data.__bufferToTensor(buffer);

              case 10:
                bufferImage = _context5.sent;
                _this$model$predict$t2 = this.model.predict(bufferImage).topk(), values = _this$model$predict$t2.values, indices = _this$model$predict$t2.indices;
                return _context5.abrupt("return", {
                  label: this.data.labels[indices.dataSync()[0]],
                  confidence: values.dataSync()[0]
                });

              case 13:
              case "end":
                return _context5.stop();
            }
          }
        }, _callee5, this);
      }));

      function predict(_x4) {
        return _predict.apply(this, arguments);
      }

      return predict;
    }()
  }]);

  return TFJSModel;
}();

var _default = TFJSModel;
exports["default"] = _default;
//# sourceMappingURL=Model.js.map