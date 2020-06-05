import KerasJS from "keras-js";
import Optimizer from "../Optimizer";
const tf = (process.env.gpu === true) ? require("@tensorflow/tfjs-node-gpu") : require("@tensorflow/tfjs-node");

const DefaultOptions = {
    inputWidth: 7,
    inputHeight: 7,
    inputChannels: 1024,
    learningRate: 0.0001,
    optimizer: "adam"
};

class KerasModel {
    constructor(options){
        this.options = options || DefaultOptions;
        this.inputWidth = options.inputWidth || DefaultOptions.inputWidth;
        this.inputHeight = options.inputHeight || DefaultOptions.inputHeight;
        this.inputChannels = options.inputChannels || DefaultOptions.inputChannels;
        this.learningRate = options.learningRate || DefaultOptions.learningRate;
    }

    load(filepath){
        this.model = new KerasJS.Model({
            filepath,
            filesystem: true
        });
    }
}

export default KerasModel;