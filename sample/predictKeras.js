/**
 * Script to test predict based on a Keras dataset 
 * 
 * @use yarn predictkeras --model=./mobilenet.bin
 */

import "@babel/polyfill/noConflict";

import path from "path";
import fg from "fast-glob";

import { Model, Data } from "../src/index";

const argv = require("minimist")(process.argv.slice(2));
const options = require(`./options/${argv.options || "default"}`);

(async () => {
    try{
        let model = new Model({
            type: "keras"
        });

        await model.load(path.resolve("./sample/model-keras.bin"));   

        let files = await fg(`${path.resolve("./sample/validate")}/*.jpg`);
        files.sort(() => { return 0.5 - Math.random(); }); //shuffle

        for(let key in files){
            let result = await model.predictFromFile(files[key]);
            console.log(path.basename(files[key]), result);
        }
    }
    catch(err){
        console.log(err);
    }
})();