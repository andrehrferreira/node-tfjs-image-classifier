/**
 * MNIST Model
 * 
 * @see http://yann.lecun.com/exdb/mnist/
 * @see https://github.com/tensorflow/tfjs-examples/blob/master/mnist/index.js
 * @see https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/
 * @see https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
 * @see https://github.com/tensorflow/tfjs-examples/blob/master/mnist-node/data.js
 */

import Optimizer from "../Optimizer";
const tf = (process.env.gpu === true) ? require("@tensorflow/tfjs-node-gpu") : require("@tensorflow/tfjs-node");

const DefaultOptions = {
    inputWidth: 28,
    inputHeight: 28,
    inputChannels: 1,
    optimizer: "rmsprop"
};

class MNISTModel {
    constructor(options){
        this.options = options || DefaultOptions;
        this.inputWidth = options.inputWidth || DefaultOptions.inputWidth;
        this.inputHeight = options.inputHeight || DefaultOptions.inputHeight;
        this.inputChannels = options.inputChannels || DefaultOptions.inputChannels;
    }

    generate(outputSize){

        //Input
        this.model = tf.sequential();
        this.model.add(tf.layers.conv2d({
            inputShape: [this.inputHeight, this.inputWidth, this.inputChannels],
            filters: 28,
            kernelSize: 3,
            activation: "relu",
        }));
        
        this.model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
        this.model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" }));
        this.model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

        //Output
        this.model.add(tf.layers.flatten());
        this.model.add(tf.layers.dropout({ rate: 0.25 }));
        this.model.add(tf.layers.dense({ units: 128, activation: "relu" }));
        this.model.add(tf.layers.dropout({ rate: 0.5 }));
        this.model.add(tf.layers.dense({ units: outputSize, activation: "softmax" }));

        let optimzer = new Optimizer(this.options);
        
        this.model.compile({
            optimizer: optimzer.getOptimzer(),
            loss: "categoricalCrossentropy",
            metrics: ["accuracy"],
        });

        return tf.model({
            inputs: this.model.layers[0].input, 
            outputs: this.model.layers[1].output
        });
    }
}

export default MNISTModel;