/**
 * CIFAR10 Model
 * 
 * @see https://www.cs.toronto.edu/~kriz/cifar.html
 * @see https://github.com/MazenAly/Cifar100/blob/master/CIFAR10.py
 * @see https://github.com/zqingr/tfjs-examples/blob/master/src/examples/cifar10_cnn/index.ts
 * @see https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html
 * @see https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/
 * @see https://keras.io/examples/cifar10_cnn/
 */

import Optimizer from "../Optimizer";
const tf = (process.env.gpu === true) ? require("@tensorflow/tfjs-node-gpu") : require("@tensorflow/tfjs-node");

const DefaultOptions = {
    inputWidth: 32,
    inputHeight: 32,
    inputChannels: 3,
    learningRate: 0.001,
    optimizer: "adam"
};

class CIFAR10Model {
    constructor(options){
        this.options = options || DefaultOptions;
        this.inputWidth = options.inputWidth || DefaultOptions.inputWidth;
        this.inputHeight = options.inputHeight || DefaultOptions.inputHeight;
        this.inputChannels = options.inputChannels || DefaultOptions.inputChannels;
        this.learningRate = options.learningRate || DefaultOptions.learningRate;
    }

    generate(outputSize){       

        //Input
        this.model = tf.sequential();
        this.model.add(tf.layers.conv2d({ 
            inputShape: [this.inputHeight, this.inputWidth, this.inputChannels],
            kernelSize: 3,
            filters: 32,
            activation: "relu",
            padding: "same"
        }));

        this.model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: "relu" }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
        this.model.add(tf.layers.dropout({ rate: 0.25 }));
        this.model.add(tf.layers.conv2d({ kernelSize: 3, filters: 64, activation: "relu", padding: "same" }));
        this.model.add(tf.layers.conv2d({ kernelSize: 3, filters: 64, activation: "relu" }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
        this.model.add(tf.layers.dropout({ rate: 0.25 }));
        
        //Output
        this.model.add(tf.layers.flatten());
        this.model.add(tf.layers.dense({ units: 512, activation: "relu" }));
        this.model.add(tf.layers.dropout({ rate: 0.5 }));
        this.model.add(tf.layers.dense({ units: outputSize, activation: "softmax" }));

        let optimzer = new Optimizer(this.options);
        
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
}

export default CIFAR10Model;