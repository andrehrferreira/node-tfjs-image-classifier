# node-tfjs-image-classifier

Study project to create data set for Tensorflow using Node.js

What does this plugin do?

* Creating binary image blocks (yarn filestodata)
* Creation of a training model on the VGG16, MNIST, CIFAR10 models
* Importing Keras models
* Model training
* Prediction with pre-trained base

## Instalation

```bash
$ yarn add node-tfjs-image-classifier --save
```

## Downloading test files

To perform the tests of the module used in the Stanford Dogs Dataset available at http://vision.stanford.edu/aditya86/ImageNetDogs/, this base was collected and organized by Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei at Stanford University. The base has 20.580 dog files classified in 120 categories, ~ 150 images per category.

The categories are also organized in the ImageNet (http://www.image-net.org/) standard, making it possible to increase pre-trained models.

## Rebuild the native addon module

So that it is possible to make use of Tensorflow in Nodejs it will be necessary to rebuild the core to add the module, to do so just execute the command below:

```bash
$ npm rebuild @tensorflow/tfjs-node build-addon-from-source
```

## Files to binary block 

When there are many class files for training I recommend turning the base into binary blocks that will be treated and compressed, these blocks can be used in training models and their loading is much more efficient than mapping files.

```bash
$ yarn filestodata
```

If you want to customize the way these files are generated below, follow the explanation of how the conversion process works.

```javascript
import "@babel/polyfill/noConflict";

import path from "path";
import { Model, Data } from "../src/index";

const argv = require("minimist")(process.argv.slice(2));
const options = require(`./options/${argv.options || "default"}`); //default, cifar10, vgg16, mnist

(async () => {
    try{    
        let data = new Data(options); //Creating dataset
        console.log("Loading files...");
        await data.loadFromDirectory(path.resolve("./sample/train")); //Loading files for conversion, where the name of the directories will be the label
        let model = new Model(options); //Creating model class
    
        await model.input(data); // Inputting data for conversion
        await model.generate(); // Generating the model according to the settings

        console.log("Saving data training...");

        //Converting and saving binary data based on the model
        await data.saveTrainingData(model.model, path.resolve("./sample/data/data.bin"), path.resolve("./sample/data/labels.json"));
    }
    catch(err){
        console.log(err);
    }
})();
```

As previously mentioned, the data can be used for training any model, obviously if the data was converted into a model whose image size setting is smaller than the training model, there may be data loss when resizing, so I recommend using the model VGG16 for creating binary files, so all images will have the standard size of 224x224.

## Loading binary data for training

Inside the example folder there is a functional script that loads binary data and performs training with model options for training, to use the script just run the command below.

```bash
$ yarn trainfromdata --options=default
```

Remembering that the module has support for VGG16, MNIST and CIFAR10. Below is the commented code:

```javascript
import "@babel/polyfill/noConflict";

import path from "path";
import { Model, Data } from "../src/index";

const argv = require("minimist")(process.argv.slice(2));
const options = require(`./options/${argv.options || "default"}`);

(async () => {
    try{    
        let data = new Data(options); //Creating dataset
        await data.loadFromData(path.resolve("./sample/data/data.bin"), path.resolve("./sample/data/labels.json")); //Loading binary files and labels
        let model = new Model(options); //Creating model class
    
        await model.input(data); // Inputting data for conversion
        await model.generate(); // Generating the model according to the settings   
        let trainResult = await model.train(); // Model training

        console.log("Training Complete!");
        let losses = trainResult.history.loss;
        console.log(`Final Loss: ${Number(losses[losses.length - 1]).toFixed(5)}`);

        await model.save(path.resolve("./sample/model")); //Saving pre training model for later use
    }
    catch(err){
        console.log(err);
    }
})();
```

## Training model using files directly

It is possible to train the model without the need to create binary files just run the command below

```bash
$ yarn trainfromfiles --options=default
```

## Create Keras model with Python

The recommendation for creating more complex models using Keras and using Phyton and not Javascript, for installation follows a tutorial link for Ubuntu 18.04: https://www.pyimagesearch.com/2019/01/30/ubuntu-18-04-install-tensorflow-and-keras-for-deep-learning/. An example of a keras model is available in the /python directory.

## Prediction using pre-trained model

After creating the model to perform prediction, just use the script below:

```javascript
import "@babel/polyfill/noConflict";

import path from "path";
import fg from "fast-glob";

import { Model, Data } from "../src/index";

const argv = require("minimist")(process.argv.slice(2));
const options = require(`./options/${argv.options || "default"}`);

(async () => {
    try{
        let model = new Model(options);
        await model.load(path.resolve("./sample/model")); // Loading pre-trained model

        let files = await fg(`${path.resolve("./sample/validate")}/*.jpg`); // Loading validation file
        files.sort(() => { return 0.5 - Math.random(); }); // Shuffle

        for(let key in files){
            let result = await model.predictFromFile(files[key]); // Direct prediction by file
            console.log(path.basename(files[key]), result); // Displaying prediction result
        }
    }
    catch(err){
        console.log(err);
    }
})();
```

## Prediction using Keras models

```javascript
import "@babel/polyfill/noConflict";

import path from "path";
import fg from "fast-glob";

import { Model, Data } from "../src/index";

(async () => {
    try{
        let model = new Model({ type: "keras" });
        await model.load(path.resolve("./sample/model-keras.bin")); // Loading pre-trained model  

        let files = await fg(`${path.resolve("./sample/validate")}/*.jpg`); // Loading validation file
        files.sort(() => { return 0.5 - Math.random(); }); // Shuffle

        for(let key in files){
            let result = await model.predictFromFile(files[key]); // Direct prediction by file
            console.log(path.basename(files[key]), result); // Displaying prediction result
        }
    }
    catch(err){
        console.log(err);
    }
})();
```