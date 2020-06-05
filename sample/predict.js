/**
 * Script to test predict
 * 
 * @use yarn predict --options=default
 */

import "@babel/polyfill/noConflict";

import path from "path";
import fg from "fast-glob";

import { Model, Data } from "../src/index";

const argv = require("minimist")(process.argv.slice(2));
const options = require(`./options/${argv.options || "default"}`);

(async () => {
    try{
        let model = new Model(options);
        await model.load(path.join("./sample/models", argv.options || "default"));   

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