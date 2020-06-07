import path from "path";
import fse from "fs-extra";
import fs from "fs";

import DefaultModel from "./models/default";
import MNISTModel from "./models/mnist";
import CIFAR10Model from "./models/cifar10";
import VGG16Model from "./models/vgg16";
import KerasModel from "./models/keras";
import TFJSData from "./Data";

import { timingSafeEqual } from "crypto";

const tf = (process.env.gpu === true) ? require("@tensorflow/tfjs-node-gpu") : require("@tensorflow/tfjs-node");

const DefaultOptions = {
    inputWidth: 7,
    inputHeight: 7,
    inputChannels: 1024,
    learningRate: 0.0001,
    optimizer: "adam",
    denseUnits: 100,
    epochs: 200,
    batchSizeFraction: 0.2
};

class TFJSModel{
    constructor(options){
        this.currentModelPath = null;
        this.options = options || DefaultOptions;
        this.type = options.type || "default"; // default, cifar10, mnist, vgg16, keras
        this.denseUnits = options.denseUnits || DefaultOptions.denseUnits; 
        this.epochs = options.epochs || DefaultOptions.epochs; 
        this.batchSizeFraction = options.batchSizeFraction || DefaultOptions.batchSizeFraction;
        this.outputSize = null;

        this.data = new TFJSData(options);

        /* eslint-disable */
        switch(this.type){
            case "default": this.modelClass = new DefaultModel(options); break;
            case "mnist": this.modelClass = new MNISTModel(options); break;
            case "cifar10": this.modelClass = new CIFAR10Model(options); break;
            case "vgg16": this.modelClass = new VGG16Model(options); break;
            case "keras": this.modelClass = new KerasModel(options); break;
            default: throw "The informed model is not compatible";
        }
        /* eslint-enable */
    }

    /**
     * Function to input data for training
     * 
     * @param object data 
     * @return void
     */
    input(data){
        if(this.type == "keras")
            throw "Keras model cannot be generated";

        if(!data.labels || !data.images)
            throw "There are no valid labels or images in the entry";

        this.data = data;
        this.outputSize = this.data.labels.length;
    }

    /**
     * Function to generate model based on configurations
     * 
     * @return void
     */
    generate(){
        if(this.type == "keras")
            throw "Keras model cannot be generated";

        if(!this.outputSize)
            throw "Error when trying to generate the model for not identifying outputSize";

        this.model = this.modelClass.generate(this.outputSize);
    }

    /**
     * Function to perform data training
     * 
     * @return Promise
     */
    async train(){
        if(this.type == "keras")
            throw "Keras type model cannot be trained";

        let loadTrainData = await this.data.toTrain(this.model);
        
        let batchSize = Math.floor(
            loadTrainData.images.shape[0] * this.batchSizeFraction
        );

        if (!(batchSize > 0)) 
            throw "Batch size is 0 or NaN. Please choose a non-zero fraction.";
        
        let shuffledIndices = new Int32Array(
            tf.util.createShuffledIndices(loadTrainData.labels.shape[0])
        );
        
        console.time("Training Time");
        return this.model.fit(
            loadTrainData.images.gather(shuffledIndices),
            loadTrainData.labels.gather(shuffledIndices),
            {
                batchSize,
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
            }
        );
    }

    /**
     * Function to load pre-trained model
     * 
     * @param string fileOrDirname
     * @return Promise
     */
    async load(fileOrDirname){
        if(this.type == "keras"){
            this.model = this.modelClass.load(fileOrDirname);
        }
        else{
            if(!fs.existsSync(fileOrDirname))
                throw "Model not exists";
                
            this.model = await tf.loadLayersModel(
                "file://" + fileOrDirname + "/model.json"
            );
    
            this.data.labels = await fse
                .readJson(path.join(fileOrDirname, "labels.json"))
                .then(obj => obj.Labels);
    
            this.currentModelPath = fileOrDirname;
        }
    }

    /**
     * Function to save pre-trained model
     * 
     * @param string dirname
     * @return Promise
     */
    async save(dirname){
        fse.ensureDirSync(dirname);

        await this.model.save("file://" + dirname);
        await fse.writeJson(path.join(dirname, "labels.json"), {
            Labels: this.data.labels
        });

        this.currentModelPath = dirname;
    }

    /**
     * Function to perform prediction by file
     * 
     * @param string filename
     * @return Promise
     */
    async predictFromFile(filename){
        if(this.type == "keras"){
            await this.model.ready();

            return await this.model.predict({
                input_1: new Float32Array(await fs.readFileSync(filename))
            });
        }
        else{
            let bufferImage = await this.data.__fileToTensor(filename);
            let { values, indices } = this.model.predict(bufferImage).topk();
            
            return {
                label: this.data.labels[indices.dataSync()[0]],
                confidence: values.dataSync()[0]
            };
        }
    }

    /**
     * Buffer prediction function
     * 
     * @param string filename
     * @return Promise
     */
    async predict(buffer){
        if(this.type == "keras"){
            await this.model.ready();

            return await this.model.predict({
                input_1: new Float32Array(buffer)
            });
        }
        else{
            let bufferImage = await this.data.__bufferToTensor(buffer);
            let { values, indices } = this.model.predict(bufferImage).topk();
            
            return {
                label: this.data.labels[indices.dataSync()[0]],
                confidence: values.dataSync()[0]
            };
        }
    }
}

export default TFJSModel;