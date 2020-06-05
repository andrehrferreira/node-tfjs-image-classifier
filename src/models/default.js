import Optimizer from "../Optimizer";
const tf = (process.env.gpu === true) ? require("@tensorflow/tfjs-node-gpu") : require("@tensorflow/tfjs-node");

const DefaultOptions = {
    inputWidth: 224,
    inputHeight: 224,
    inputChannels: 3,
    learningRate: 0.0001,
    optimizer: "adam"
};

class DefaultModel {
    constructor(options){
        this.options = options || DefaultOptions;
        this.inputWidth = options.inputWidth || DefaultOptions.inputWidth;
        this.inputHeight = options.inputHeight || DefaultOptions.inputHeight;
        this.inputChannels = options.inputChannels || DefaultOptions.inputChannels;
        this.learningRate = options.learningRate || DefaultOptions.learningRate;
    }

    generate(outputSize){       

        //Input
        this.model = tf.sequential({
            layers: [
                tf.layers.flatten({ 
                    inputShape: [this.inputHeight, this.inputWidth, this.inputChannels] 
                }),
                tf.layers.dense({ 
                    units: 64, 
                    activation: "relu", 
                    kernelInitializer: "varianceScaling", 
                    useBias: true 
                }),
                tf.layers.dense({ 
                    units: outputSize, 
                    kernelInitializer: "varianceScaling",
                    activation: "softmax",
                    useBias: false,
                })
            ]
        });
 
        //Output
        let optimzer = new Optimizer(this.options);
        
        this.model.compile({
            optimizer: optimzer.getOptimzer(),
            loss: "categoricalCrossentropy",
            metrics: ["accuracy"]
        });
    
        return this.model;
    }
}

export default DefaultModel;