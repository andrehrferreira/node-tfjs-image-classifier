"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = exports.Data = exports.Model = void 0;

var _Model = _interopRequireDefault(require("./Model"));

var _Data = _interopRequireDefault(require("./Data"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

var Model = _Model["default"];
exports.Model = Model;
var Data = _Data["default"];
exports.Data = Data;
var _default = {
  Model: _Model["default"],
  Data: _Data["default"]
};
exports["default"] = _default;
//# sourceMappingURL=index.js.map