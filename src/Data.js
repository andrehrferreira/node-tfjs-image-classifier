/**
 * 
 * @see https://github.com/longlost/node-tfjs-retrain/blob/master/data.js
 * @see https://github.com/GantMan/rps_tfjs_demo/blob/master/src/tfjs/data.js
 */

import path from "path";
import fg from "fast-glob";
import fse from "fs-extra";
import fs from "fs";
import sharp from "sharp";
import tobuffer from "typedarray-to-buffer";

const tf = (process.env.gpu === true) ? require("@tensorflow/tfjs-node-gpu") : require("@tensorflow/tfjs-node");

const DefaultOptions = {
    inputWidth: 7,
    inputHeight: 7,
    inputChannels: 1024,
    learningRate: 0.0001,
    optimizer: "adam"
};

class TFJSData {
    constructor(options){
        this.options = options || DefaultOptions;
        this.inputWidth = options.inputWidth || DefaultOptions.inputWidth;
        this.inputHeight = options.inputHeight || DefaultOptions.inputHeight;
        this.inputChannels = options.inputChannels || DefaultOptions.inputChannels;
        this.labels = null;
    }

    __stripAlphaChannel(tensor, info){
        return tf.tidy(() => tensor.slice(
            [0, 0, 0 ,0], 
            [1, info.height, info.width, 3]
        ));
    }

    __imageToTensor(pixelData, filename){
        let outShape = [1, this.inputHeight, this.inputWidth, this.inputChannels];        

        try{
            return tf.tidy(() => {
                const outShape = [1, this.inputHeight, this.inputWidth, this.inputChannels];
                const tensor   = tf.tensor4d(pixelData, outShape, "int32"); 
                /*const noAlpha  = this.__stripAlphaChannel(tensor, {
                    height: this.inputHeight,
                    width: this.inputWidth
                });*/

                const normalized = tensor.  
                    resizeBilinear([this.inputWidth, this.inputHeight]).
                    toFloat().
                    div(tf.scalar(127)).
                    sub(tf.scalar(1));
            
                return normalized;
            });

            /*return tf.tidy(() =>
                tf.tensor4d(pixelData, outShape, "int32")
                    //.resizeBilinear([this.inputWidth, this.inputHeight])
                    .toFloat()
                    .div(tf.scalar(127))
                    .sub(tf.scalar(1))
            );*/
        }
        catch(err){
            console.log(filename, pixelData);
            throw err;
        }
    }

    async __fileToTensor(filename){
        let { data } = await sharp(filename)
            .resize({ width: this.inputWidth, height: this.inputHeight })
            .raw()
            .toBuffer({ resolveWithObject: true });

        return this.__imageToTensor(data, filename);
    }

    async __bufferToTensor(buffer){
        return this.__imageToTensor(Buffer.from(buffer));
    }

    async __fileToBuffer(filename){
        const { data } = await sharp(filename)
            .resize({ width: this.inputWidth, height: this.inputHeight })
            .raw()
            .toBuffer({ resolveWithObject: true });

        return data;
    }

    async loadFromDirectory(dirname){
        let diretories = await fg(`${dirname}/*`, { onlyDirectories: true });
        this.labels = diretories.map((dir) => { return path.basename(dir); });
        this.images = await Promise.all(diretories.map(async (dir) => { return await fg(`${dir}/*`, { onlyFiles: true }); }));
        this.sizes = this.images.map((arr) => arr.length);
    }

    async loadFromData(filenameData, filenameLabels){
        let filesBin = await fg(`${filenameData}.*`, { onlyFiles: true });
        this.images = [];
        
        for(let i = 0; i < filesBin.length; i++){
            console.log(`Loading ${filenameData}.${i}...`);

            let buffer = fs.readFileSync(`${filenameData}.${i}`);
            let bufferArr = new Float32Array(buffer);

            if(!this.images[i])
                this.images[i] = [];

            let embeddingsShape = [this.inputHeight, this.inputWidth, this.inputChannels];
            let embeddingsFlatSize = tf.util.sizeFromShape(embeddingsShape);
            let index = 0;

            for (let pArr = 0; pArr < bufferArr.length; pArr += embeddingsFlatSize){
                this.images[i][index] = bufferArr.slice(pArr, pArr + embeddingsFlatSize);
                index++;
            }
        }

        this.sizes = this.images.map((arr) => arr.length);
        
        this.labels = await fse
            .readJson(filenameLabels)
            .then(obj => obj.Labels);
    }

    async saveTrainingData(model, filenameData, filenameLabels){
        let embeddings = [];
    
        for (let key in this.labels) {
            let embeddingsOffset = 0;
            let embeddingsShape = [this.images[key].length, this.inputHeight, this.inputWidth, this.inputChannels];
            let embeddingsFlatSize = tf.util.sizeFromShape([this.inputHeight, this.inputWidth, this.inputChannels]);
            embeddings[key] = new Float32Array(tf.util.sizeFromShape(embeddingsShape));

            for (let keyImages in this.images[key]) {
                embeddings[key].set(await this.__fileToBuffer(this.images[key][keyImages]), embeddingsOffset);
                embeddingsOffset += embeddingsFlatSize;
            }

            let dataBuffer = Buffer.from(embeddings[key]);
            fs.writeFileSync(`${filenameData}.${key}`, dataBuffer);
            console.log(`Saving ${filenameData}.${key}...`);
        }

        await fse.writeJson(filenameLabels, {
            Labels: this.labels
        });
    }

    async toTrain(model, retrain = false) {
        let totalImages = this.images.map((item) => item.length).reduce((accumulator, currentValue) => accumulator + currentValue);
        let embeddingsShape = [this.inputHeight, this.inputWidth, this.inputChannels];
        let embeddingsShapeTotal = [totalImages, this.inputHeight, this.inputWidth, this.inputChannels];
        let embeddingsFlatSize = tf.util.sizeFromShape(embeddingsShape);

        const totalDataSize = tf.util.sizeFromShape(embeddingsShapeTotal);
        const maxSize = (2 ** 30) - 1;

        const batchCount = totalDataSize > maxSize ?  Math.ceil(totalDataSize / maxSize) : 1;
        const maxBatchSize = totalDataSize / batchCount;
        const predictionsPerBatch = Math.floor(maxBatchSize / embeddingsFlatSize);

        const batchSize = predictionsPerBatch * embeddingsFlatSize;
        let batchIndex = 0;
                
        //let embeddings = new Float32Array(tf.util.sizeFromShape(embeddingsShape));
        //const embeddings = new Float32Array(batchSize);
        const embeddings = [];
        let embeddingsOffset = 0;

        const labels = new Int32Array(totalImages);        
        let labelsOffset = 0;

        console.time("Loading Training Data");

        for (let key in this.labels) {            
            for (let keyImages in this.images[key]) {
                let tmpBuffer = (typeof this.images[key][keyImages] == "object") ? await this.__bufferToTensor(this.images[key][keyImages]) : await this.__fileToTensor(this.images[key][keyImages]);
                
                if(retrain)
                    embeddings.push(model.predict(tmpBuffer).squeeze());
                else
                    embeddings.push(tmpBuffer.squeeze());

                labels.set([key], labelsOffset);
                tmpBuffer.dispose();
                    
                embeddingsOffset += embeddingsFlatSize;
                labelsOffset++;  
            }

            console.timeLog("Loading Training Data", {
                label: this.labels[key],
                count: this.images[key].length
            });
        }

        try{
            const imagesTensor = tf.stack(embeddings);
            embeddings.forEach(tensor => { tensor.dispose(); });

            return {
                images: imagesTensor, //tf.tensor4d(embeddings, embeddingsShape),
                labels: tf.oneHot(tf.tensor1d(labels, "int32"), this.labels.length)
            };
        }
        catch(err){
            console.log("aki", err);
        }
    }
}

export default TFJSData;