import Optimizer from "../Optimizer";
const tf = (process.env.gpu === true) ? require("@tensorflow/tfjs-node-gpu") : require("@tensorflow/tfjs-node");

const DefaultOptions = {
    inputWidth: 32,
    inputHeight: 32,
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

        this.model = tf.sequential();

        this.model.add(tf.layers.conv2d({
            inputShape: [this.inputWidth, this.inputHeight, this.inputChannels],
            kernelSize: 3,
            padding: "same",
            filters: 32,
            strides: 1,
            activation: "relu",
            kernelInitializer: "varianceScaling"
        }));

        this.model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
        this.model.add(tf.layers.batchNormalization());
        this.model.add(tf.layers.dropout({ rate: 0.25 }));

        this.model.add(tf.layers.conv2d({
            kernelSize: 3,
            filters: 64,
            padding: "same",
            strides: 1,
            activation: "relu",
            kernelInitializer: "varianceScaling"
        }));

        this.model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
        this.model.add(tf.layers.batchNormalization());
        this.model.add(tf.layers.dropout({ rate: 0.25 }));
        this.model.add(tf.layers.flatten());

        this.model.add(tf.layers.dense({
            units: outputSize,
            kernelInitializer: "varianceScaling",
            activation: "softmax"
        }));
 
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