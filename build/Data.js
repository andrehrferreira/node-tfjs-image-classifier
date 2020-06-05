"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _path = _interopRequireDefault(require("path"));

var _fastGlob = _interopRequireDefault(require("fast-glob"));

var _fsExtra = _interopRequireDefault(require("fs-extra"));

var _fs = _interopRequireDefault(require("fs"));

var _sharp = _interopRequireDefault(require("sharp"));

var _typedarrayToBuffer = _interopRequireDefault(require("typedarray-to-buffer"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

function _typeof(obj) { "@babel/helpers - typeof"; if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") { _typeof = function _typeof(obj) { return typeof obj; }; } else { _typeof = function _typeof(obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }; } return _typeof(obj); }

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
  optimizer: "adam"
};

var TFJSData = /*#__PURE__*/function () {
  function TFJSData(options) {
    _classCallCheck(this, TFJSData);

    this.options = options || DefaultOptions;
    this.inputWidth = options.inputWidth || DefaultOptions.inputWidth;
    this.inputHeight = options.inputHeight || DefaultOptions.inputHeight;
    this.inputChannels = options.inputChannels || DefaultOptions.inputChannels;
    this.labels = null;
  }

  _createClass(TFJSData, [{
    key: "__stripAlphaChannel",
    value: function __stripAlphaChannel(tensor, info) {
      return tf.tidy(function () {
        return tensor.slice([0, 0, 0, 0], [1, info.height, info.width, 3]);
      });
    }
  }, {
    key: "__imageToTensor",
    value: function __imageToTensor(pixelData, filename) {
      var _this = this;

      var outShape = [1, this.inputHeight, this.inputWidth, this.inputChannels];

      try {
        return tf.tidy(function () {
          // rgba -> rgb / the rest of the pipeline throws if 
          // alpha channel is present so pull it out and 
          // keep a shape [1, 224, 224, 3] tensor
          var outShape = [1, _this.inputHeight, _this.inputWidth, _this.inputChannels];
          var tensor = tf.tensor4d(pixelData, outShape, "int32");

          var noAlpha = _this.__stripAlphaChannel(tensor, {
            height: _this.inputHeight,
            width: _this.inputWidth
          }); // Normalize the rgb data from [0, 255] to [-1, 1].


          var normalized = noAlpha.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
          return normalized;
        });
        /*return tf.tidy(() =>
            tf.tensor4d(pixelData, outShape, "int32")
                //.resizeBilinear([this.inputWidth, this.inputHeight])
                .toFloat()
                .div(tf.scalar(127))
                .sub(tf.scalar(1))
        );*/
      } catch (err) {
        console.log(filename, pixelData);
        throw err;
      }
    }
  }, {
    key: "__fileToTensor",
    value: function () {
      var _fileToTensor = _asyncToGenerator( /*#__PURE__*/regeneratorRuntime.mark(function _callee(filename) {
        var _yield$sharp$resize$r, data;

        return regeneratorRuntime.wrap(function _callee$(_context) {
          while (1) {
            switch (_context.prev = _context.next) {
              case 0:
                _context.next = 2;
                return (0, _sharp["default"])(filename).resize({
                  width: this.inputWidth,
                  height: this.inputHeight
                }).raw().toBuffer({
                  resolveWithObject: true
                });

              case 2:
                _yield$sharp$resize$r = _context.sent;
                data = _yield$sharp$resize$r.data;
                return _context.abrupt("return", this.__imageToTensor(data, filename));

              case 5:
              case "end":
                return _context.stop();
            }
          }
        }, _callee, this);
      }));

      function __fileToTensor(_x) {
        return _fileToTensor.apply(this, arguments);
      }

      return __fileToTensor;
    }()
  }, {
    key: "__bufferToTensor",
    value: function () {
      var _bufferToTensor = _asyncToGenerator( /*#__PURE__*/regeneratorRuntime.mark(function _callee2(buffer) {
        return regeneratorRuntime.wrap(function _callee2$(_context2) {
          while (1) {
            switch (_context2.prev = _context2.next) {
              case 0:
                return _context2.abrupt("return", this.__imageToTensor(Buffer.from(buffer)));

              case 1:
              case "end":
                return _context2.stop();
            }
          }
        }, _callee2, this);
      }));

      function __bufferToTensor(_x2) {
        return _bufferToTensor.apply(this, arguments);
      }

      return __bufferToTensor;
    }()
  }, {
    key: "__fileToBuffer",
    value: function () {
      var _fileToBuffer = _asyncToGenerator( /*#__PURE__*/regeneratorRuntime.mark(function _callee3(filename) {
        var _yield$sharp$resize$r2, data;

        return regeneratorRuntime.wrap(function _callee3$(_context3) {
          while (1) {
            switch (_context3.prev = _context3.next) {
              case 0:
                _context3.next = 2;
                return (0, _sharp["default"])(filename).resize({
                  width: this.inputWidth,
                  height: this.inputHeight
                }).raw().toBuffer({
                  resolveWithObject: true
                });

              case 2:
                _yield$sharp$resize$r2 = _context3.sent;
                data = _yield$sharp$resize$r2.data;
                return _context3.abrupt("return", data);

              case 5:
              case "end":
                return _context3.stop();
            }
          }
        }, _callee3, this);
      }));

      function __fileToBuffer(_x3) {
        return _fileToBuffer.apply(this, arguments);
      }

      return __fileToBuffer;
    }()
  }, {
    key: "loadFromDirectory",
    value: function () {
      var _loadFromDirectory = _asyncToGenerator( /*#__PURE__*/regeneratorRuntime.mark(function _callee5(dirname) {
        var diretories;
        return regeneratorRuntime.wrap(function _callee5$(_context5) {
          while (1) {
            switch (_context5.prev = _context5.next) {
              case 0:
                _context5.next = 2;
                return (0, _fastGlob["default"])("".concat(dirname, "/*"), {
                  onlyDirectories: true
                });

              case 2:
                diretories = _context5.sent;
                this.labels = diretories.map(function (dir) {
                  return _path["default"].basename(dir);
                });
                _context5.next = 6;
                return Promise.all(diretories.map( /*#__PURE__*/function () {
                  var _ref = _asyncToGenerator( /*#__PURE__*/regeneratorRuntime.mark(function _callee4(dir) {
                    return regeneratorRuntime.wrap(function _callee4$(_context4) {
                      while (1) {
                        switch (_context4.prev = _context4.next) {
                          case 0:
                            _context4.next = 2;
                            return (0, _fastGlob["default"])("".concat(dir, "/*"), {
                              onlyFiles: true
                            });

                          case 2:
                            return _context4.abrupt("return", _context4.sent);

                          case 3:
                          case "end":
                            return _context4.stop();
                        }
                      }
                    }, _callee4);
                  }));

                  return function (_x5) {
                    return _ref.apply(this, arguments);
                  };
                }()));

              case 6:
                this.images = _context5.sent;
                this.sizes = this.images.map(function (arr) {
                  return arr.length;
                });

              case 8:
              case "end":
                return _context5.stop();
            }
          }
        }, _callee5, this);
      }));

      function loadFromDirectory(_x4) {
        return _loadFromDirectory.apply(this, arguments);
      }

      return loadFromDirectory;
    }()
  }, {
    key: "loadFromData",
    value: function () {
      var _loadFromData = _asyncToGenerator( /*#__PURE__*/regeneratorRuntime.mark(function _callee6(filenameData, filenameLabels) {
        var filesBin, i, buffer, bufferArr, embeddingsShape, embeddingsFlatSize, index, pArr;
        return regeneratorRuntime.wrap(function _callee6$(_context6) {
          while (1) {
            switch (_context6.prev = _context6.next) {
              case 0:
                _context6.next = 2;
                return (0, _fastGlob["default"])("".concat(filenameData, ".*"), {
                  onlyFiles: true
                });

              case 2:
                filesBin = _context6.sent;
                this.images = [];

                for (i = 0; i < filesBin.length; i++) {
                  console.log("Loading ".concat(filenameData, ".").concat(i, "..."));
                  buffer = _fs["default"].readFileSync("".concat(filenameData, ".").concat(i));
                  bufferArr = new Float32Array(buffer);
                  if (!this.images[i]) this.images[i] = [];
                  embeddingsShape = [this.inputHeight, this.inputWidth, this.inputChannels];
                  embeddingsFlatSize = tf.util.sizeFromShape(embeddingsShape);
                  index = 0;

                  for (pArr = 0; pArr < bufferArr.length; pArr += embeddingsFlatSize) {
                    this.images[i][index] = bufferArr.slice(pArr, pArr + embeddingsFlatSize);
                    index++;
                  }
                }

                this.sizes = this.images.map(function (arr) {
                  return arr.length;
                });
                _context6.next = 8;
                return _fsExtra["default"].readJson(filenameLabels).then(function (obj) {
                  return obj.Labels;
                });

              case 8:
                this.labels = _context6.sent;

              case 9:
              case "end":
                return _context6.stop();
            }
          }
        }, _callee6, this);
      }));

      function loadFromData(_x6, _x7) {
        return _loadFromData.apply(this, arguments);
      }

      return loadFromData;
    }()
  }, {
    key: "saveTrainingData",
    value: function () {
      var _saveTrainingData = _asyncToGenerator( /*#__PURE__*/regeneratorRuntime.mark(function _callee7(model, filenameData, filenameLabels) {
        var embeddings, key, embeddingsOffset, embeddingsShape, embeddingsFlatSize, keyImages, dataBuffer;
        return regeneratorRuntime.wrap(function _callee7$(_context7) {
          while (1) {
            switch (_context7.prev = _context7.next) {
              case 0:
                embeddings = [];
                _context7.t0 = regeneratorRuntime.keys(this.labels);

              case 2:
                if ((_context7.t1 = _context7.t0()).done) {
                  _context7.next = 25;
                  break;
                }

                key = _context7.t1.value;
                embeddingsOffset = 0;
                embeddingsShape = [this.images[key].length, this.inputHeight, this.inputWidth, this.inputChannels];
                embeddingsFlatSize = tf.util.sizeFromShape([this.inputHeight, this.inputWidth, this.inputChannels]);
                embeddings[key] = new Float32Array(tf.util.sizeFromShape(embeddingsShape));
                _context7.t2 = regeneratorRuntime.keys(this.images[key]);

              case 9:
                if ((_context7.t3 = _context7.t2()).done) {
                  _context7.next = 20;
                  break;
                }

                keyImages = _context7.t3.value;
                _context7.t4 = embeddings[key];
                _context7.next = 14;
                return this.__fileToBuffer(this.images[key][keyImages]);

              case 14:
                _context7.t5 = _context7.sent;
                _context7.t6 = embeddingsOffset;

                _context7.t4.set.call(_context7.t4, _context7.t5, _context7.t6);

                embeddingsOffset += embeddingsFlatSize;
                _context7.next = 9;
                break;

              case 20:
                dataBuffer = Buffer.from(embeddings[key]);

                _fs["default"].writeFileSync("".concat(filenameData, ".").concat(key), dataBuffer);

                console.log("Saving ".concat(filenameData, ".").concat(key, "..."));
                _context7.next = 2;
                break;

              case 25:
                _context7.next = 27;
                return _fsExtra["default"].writeJson(filenameLabels, {
                  Labels: this.labels
                });

              case 27:
              case "end":
                return _context7.stop();
            }
          }
        }, _callee7, this);
      }));

      function saveTrainingData(_x8, _x9, _x10) {
        return _saveTrainingData.apply(this, arguments);
      }

      return saveTrainingData;
    }()
  }, {
    key: "toTrain",
    value: function () {
      var _toTrain = _asyncToGenerator( /*#__PURE__*/regeneratorRuntime.mark(function _callee8(model) {
        var totalImages, embeddingsShape, embeddingsShapeTotal, embeddingsFlatSize, totalDataSize, maxSize, batchCount, maxBatchSize, predictionsPerBatch, batchSize, batchIndex, embeddings, embeddingsOffset, labels, labelsOffset, key, keyImages, tmpBuffer, prediction, imagesTensor;
        return regeneratorRuntime.wrap(function _callee8$(_context8) {
          while (1) {
            switch (_context8.prev = _context8.next) {
              case 0:
                totalImages = this.images.map(function (item) {
                  return item.length;
                }).reduce(function (accumulator, currentValue) {
                  return accumulator + currentValue;
                });
                embeddingsShape = [this.inputHeight, this.inputWidth, this.inputChannels];
                embeddingsShapeTotal = [totalImages, this.inputHeight, this.inputWidth, this.inputChannels];
                embeddingsFlatSize = tf.util.sizeFromShape(embeddingsShape);
                totalDataSize = tf.util.sizeFromShape(embeddingsShapeTotal);
                maxSize = Math.pow(2, 30) - 1;
                batchCount = totalDataSize > maxSize ? Math.ceil(totalDataSize / maxSize) : 1;
                maxBatchSize = totalDataSize / batchCount;
                predictionsPerBatch = Math.floor(maxBatchSize / embeddingsFlatSize);
                batchSize = predictionsPerBatch * embeddingsFlatSize;
                batchIndex = 0; //let embeddings = new Float32Array(tf.util.sizeFromShape(embeddingsShape));
                //const embeddings = new Float32Array(batchSize);

                embeddings = [];
                embeddingsOffset = 0;
                labels = new Int32Array(totalImages);
                labelsOffset = 0;
                console.time("Loading Training Data");
                _context8.t0 = regeneratorRuntime.keys(this.labels);

              case 17:
                if ((_context8.t1 = _context8.t0()).done) {
                  _context8.next = 43;
                  break;
                }

                key = _context8.t1.value;
                _context8.t2 = regeneratorRuntime.keys(this.images[key]);

              case 20:
                if ((_context8.t3 = _context8.t2()).done) {
                  _context8.next = 40;
                  break;
                }

                keyImages = _context8.t3.value;

                if (!(_typeof(this.images[key][keyImages]) == "object")) {
                  _context8.next = 28;
                  break;
                }

                _context8.next = 25;
                return this.__bufferToTensor(this.images[key][keyImages]);

              case 25:
                _context8.t4 = _context8.sent;
                _context8.next = 31;
                break;

              case 28:
                _context8.next = 30;
                return this.__fileToTensor(this.images[key][keyImages]);

              case 30:
                _context8.t4 = _context8.sent;

              case 31:
                tmpBuffer = _context8.t4;
                prediction = model.predict(tmpBuffer);
                tmpBuffer.dispose(); //embeddings.push(prediction.squeeze());
                //labels.set([key], labelsOffset);

                console.log(prediction.squeeze());
                process.exit(1);
                embeddingsOffset += embeddingsFlatSize;
                labelsOffset++;
                _context8.next = 20;
                break;

              case 40:
                console.timeLog("Loading Training Data", {
                  label: this.labels[key],
                  count: this.images[key].length
                });
                _context8.next = 17;
                break;

              case 43:
                _context8.prev = 43;
                imagesTensor = tf.stack(embeddings);
                embeddings.forEach(function (tensor) {
                  tensor.dispose();
                });
                console.log(imagesTensor);
                console.log(tf.tensor4d(imagesTensor, embeddingsShapeTotal));
                process.exit(1);
                return _context8.abrupt("return", {
                  images: imagesTensor,
                  //tf.tensor4d(embeddings, embeddingsShape),
                  labels: tf.oneHot(tf.tensor1d(labels, "int32"), this.labels.length)
                });

              case 52:
                _context8.prev = 52;
                _context8.t5 = _context8["catch"](43);
                console.log("aki", _context8.t5);

              case 55:
              case "end":
                return _context8.stop();
            }
          }
        }, _callee8, this, [[43, 52]]);
      }));

      function toTrain(_x11) {
        return _toTrain.apply(this, arguments);
      }

      return toTrain;
    }()
  }]);

  return TFJSData;
}();

var _default = TFJSData;
exports["default"] = _default;
//# sourceMappingURL=Data.js.map