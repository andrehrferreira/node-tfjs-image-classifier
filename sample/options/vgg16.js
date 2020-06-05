module.exports = {
    type: "vgg16",
    inputWidth: 224,
    inputHeight: 224,
    inputChannels: 3,
    learningRate: 0.009,
    batchSizeFraction: 0.20,
    epochs: 10,
    optimizer: "sgd"
};