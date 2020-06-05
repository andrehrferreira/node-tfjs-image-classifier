/**
 * Script to create treated binary files from the training image base
 * 
 * @use yarn filestodata --options=default
 */

import "@babel/polyfill/noConflict";

import path from "path";
import { Model, Data } from "../src/index";

const argv = require("minimist")(process.argv.slice(2));
const options = require(`./options/${argv.options || "default"}`);

(async () => {
    try{    
        let data = new Data(options);
        console.log("Loading files...");
        await data.loadFromDirectory(path.resolve("./sample/train"));
        let model = new Model(options);
    
        await model.input(data);
        await model.generate();    

        console.log("Saving data training...");
        await data.saveTrainingData(model.model, path.resolve("./sample/data/data.bin"), path.resolve("./sample/data/labels.json"));
    }
    catch(err){
        console.log(err);
    }
})();