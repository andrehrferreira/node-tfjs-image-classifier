const tf = (process.env.gpu === true) ? require("@tensorflow/tfjs-node-gpu") : require("@tensorflow/tfjs-node");

class Optimizer {
    constructor(options){
        let optimizerName = options.optimizer || "adam";
        let learningRate = options.learningRate || 0.001;

        switch(optimizerName){
        case "adam": this.optimize = tf.train.adam(learningRate); break;
        case "sgd": this.optimize = tf.train.sgd(learningRate); break;
        case "rmsprop": this.optimize = tf.train.rmsprop(learningRate); break;
        default: this.optimize = optimizerName; break;
        }
    }

    getOptimzer(){
        return this.optimize;
    }
}

export default Optimizer;