/**
 * VGG16 Model
 * 
 * @see https://keras.io/api/applications/vgg/
 * @see https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg16.py
 * @see https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
 * @see https://github.com/tensorflow/tfjs/issues/898
 * @see https://neurohive.io/en/popular-networks/vgg16/
 */

import Optimizer from "../Optimizer";
const tf = (process.env.gpu === true) ? require("@tensorflow/tfjs-node-gpu") : require("@tensorflow/tfjs-node");

const DefaultOptions = {
    inputWidth: 224,
    inputHeight: 224,
    inputChannels: 3,
    learningRate: 0.009,
    optimizer: "sgd"
};

class VGG16Model {
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
        this.model.add(tf.layers.zeroPadding2d({ 
            inputShape: [this.inputHeight, this.inputWidth, this.inputChannels],
            padding: 1 
        }));

        this.model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" }));
        this.model.add(tf.layers.zeroPadding2d({ padding: 1 }));
        this.model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

        this.model.add(tf.layers.zeroPadding2d({ padding: 1 }));
        this.model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: "relu" }));
        this.model.add(tf.layers.zeroPadding2d({ padding: 1 }));
        this.model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: "relu" }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

        this.model.add(tf.layers.zeroPadding2d({ padding: 1 }));
        this.model.add(tf.layers.conv2d({ filters: 256, kernelSize: 3, activation: "relu" }));
        this.model.add(tf.layers.zeroPadding2d({ padding: 1 }));
        this.model.add(tf.layers.conv2d({ filters: 256, kernelSize: 3, activation: "relu" }));
        this.model.add(tf.layers.zeroPadding2d({ padding: 1 }));
        this.model.add(tf.layers.conv2d({ filters: 256, kernelSize: 3, activation: "relu" }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

        this.model.add(tf.layers.zeroPadding2d({ padding: 1 }));
        this.model.add(tf.layers.conv2d({ filters: 512, kernelSize: 3, activation: "relu" }));
        this.model.add(tf.layers.zeroPadding2d({ padding: 1 }));
        this.model.add(tf.layers.conv2d({ filters: 512, kernelSize: 3, activation: "relu" }));
        this.model.add(tf.layers.zeroPadding2d({ padding: 1 }));
        this.model.add(tf.layers.conv2d({ filters: 512, kernelSize: 3, activation: "relu" }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

        this.model.add(tf.layers.zeroPadding2d({ padding: 1 }));
        this.model.add(tf.layers.conv2d({ filters: 512, kernelSize: 3, activation: "relu" }));
        this.model.add(tf.layers.zeroPadding2d({ padding: 1 }));
        this.model.add(tf.layers.conv2d({ filters: 512, kernelSize: 3, activation: "relu" }));
        this.model.add(tf.layers.zeroPadding2d({ padding: 1 }));
        this.model.add(tf.layers.conv2d({ filters: 512, kernelSize: 3, activation: "relu" }));
        this.model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

        //Output
        this.model.add(tf.layers.flatten());
        this.model.add(tf.layers.dense({ units: 4096, activation: "relu" }));
        this.model.add(tf.layers.dropout({ rate: 0.5 }));
        this.model.add(tf.layers.dense({ units: 4096, activation: "relu" }));
        this.model.add(tf.layers.dropout({ rate: 0.5 }));
        this.model.add(tf.layers.dense({ units: 1000, activation: "relu" }));
        this.model.add(tf.layers.dropout({ rate: 0.5 }));
        this.model.add(tf.layers.dense({ units: outputSize, activation: "softmax"}));

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

export default VGG16Model;