module.exports = {
    type: "cifar10",
    inputWidth: 32,
    inputHeight: 32,
    inputChannels: 3,
    learningRate: 0.0001,
    batchSizeFraction: 0.32,
    epochs: 100,
    optimizer: "rmsprop"
};