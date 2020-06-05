/**
 * Script to perform model training based on binary files 
 * 
 * @use yarn trainfromdata --options=default
 */

import "@babel/polyfill/noConflict";

import path from "path";
import { Model, Data } from "../src/index";

const argv = require("minimist")(process.argv.slice(2));
const options = require(`./options/${argv.options || "default"}`);

(async () => {
    try{    
        let data = new Data(options);
        await data.loadFromData(path.resolve("./sample/data/data.bin"), path.resolve("./sample/data/labels.json"));
        let model = new Model(options);
    
        await model.input(data);
        await model.generate();    
        let trainResult = await model.train();

        console.log("Training Complete!");
        let losses = trainResult.history.loss;
        console.log(`Final Loss: ${Number(losses[losses.length - 1]).toFixed(5)}`);

        await model.save(path.join("./sample/models", argv.options || "default"));
    }
    catch(err){
        console.log(err);
    }
})();