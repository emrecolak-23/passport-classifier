const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");

const LOOKUP = ["Not Passport", "Passport"];

const imageSize = 150;
const inputsArray = [];
const outputsArray = [];

async function loadImageAsArray(imagePath) {
  const imageBuffer = fs.readFileSync(imagePath);
  const imageTensor = tf.node
    .decodeImage(imageBuffer, 3)
    .resizeNearestNeighbor([imageSize, imageSize])
    .toFloat()
    .div(tf.scalar(255.0));
  const imageData = imageTensor.flatten().arraySync();
  return imageData;
}

async function loadImagesAsInputsArray(imageDir) {
  const files = fs.readdirSync(imageDir);

  for (const file of files) {
    const ext = path.extname(file).toLowerCase();
    if (ext === ".jpg") {
      const filePath = path.join(imageDir, file);
      const imageData = await loadImageAsArray(filePath);
      inputsArray.push(imageData);
      outputsArray.push(file.includes("trash_data") ? 0 : 1);
    }
  }

  return { inputsArray, outputsArray };
}

async function main() {
  const { inputsArray, outputsArray } = await loadImagesAsInputsArray(imageDir);
  console.log("Inputs Array: ", inputsArray);
  console.log("Outputs Array: ", outputsArray);

  tf.util.shuffleCombo(inputsArray, outputsArray);

  const INPUT_TENSOR = normalize(tf.tensor2d(inputsArray), 0, 1);
  const OUTPUT_TENSOR = tf.oneHot(tf.tensor1d(outputsArray, "int32"), 2);

  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [imageSize * imageSize * 3],
      units: 100,
      activation: "relu",
    })
  );

  model.add(
    tf.layers.dense({
      units: 2,
      activation: "softmax",
    })
  );
  await train(INPUT_TENSOR, OUTPUT_TENSOR, model);
}

function normalize(tensor, min, max) {
  const result = tf.tidy(() => {
    const MIN_VALUES = tf.scalar(min);
    const MAX_VALUES = tf.scalar(max);
    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);
    return NORMALIZED_VALUES;
  });
  return result;
}

async function train(INPUTS_TENSOR, OUTPUTS_TENSOR, model) {
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
    shuffle: true,
    validationSplit: 0.1,
    batchSize: 512,
    epochs: 50,
    callbacks: { onEpochEnd: logProgress },
  });

  OUTPUTS_TENSOR.dispose();
  INPUTS_TENSOR.dispose();

  evaluate(model);
}

function logProgress(epoch, logs) {
  console.log("Data for epoch " + epoch, logs);
}

function evaluate(model) {
  const OFFSET = Math.floor(Math.random() * inputsArray.length);

  let answer = tf.tidy(function () {
    let newInput = normalize(tf.tensor1d(inputsArray[OFFSET]), 0, 1);
    let output = model.predict(newInput.expandDims());
    output.print();
    return output.squeeze().argMax();
  });

  answer.array().then(function (index) {
    console.log("Predicted: ", LOOKUP[index]);
  });
}

main();
