/* globals tf */

const MODEL_PATH =
  "https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4";
const EXAMPLE_IMG = document.getElementById("exampleImg");
const CANVAS = document.getElementById("testCanvas");
const CTX = CANVAS.getContext("2d");

let movenet = undefined;
let measurePose = true;

// loadAndRunModel();

const STATUS = document.getElementById("status");
const VIDEO = document.getElementById("webcam");
const ENABLE_CAM_BUTTON = document.getElementById("enableCam");
const RESET_BUTTON = document.getElementById("reset");
const TRAIN_BUTTON = document.getElementById("train");
const MOBILE_NET_INPUT_WIDTH = 192;
const MOBILE_NET_INPUT_HEIGHT = 192;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = [];

// ENABLE_CAM_BUTTON.addEventListener("click", enableCam);
TRAIN_BUTTON.addEventListener("click", trainAndPredict);
RESET_BUTTON.addEventListener("click", reset);

// start camera when page is loaded
if (window.addEventListener) {
  window.addEventListener("load", enableCam, false); //W3C
  window.addEventListener("load", drawPoseOnFrame, false); //not working.
} else {
  window.attachEvent("onload", enableCam); //IE
}



// Just add more buttons in HTML to allow classification of more classes of data!
let dataCollectorButtons = document.querySelectorAll("button.dataCollector");

for (let i = 0; i < dataCollectorButtons.length; i++) {
  dataCollectorButtons[i].addEventListener("mousedown", drawPoseOnFrame);
  dataCollectorButtons[i].addEventListener("mouseup", drawPoseOnFrame);
  // For mobile.
  dataCollectorButtons[i].addEventListener("touchend", gatherDataForClass);

  // Populate the human readable names for classes.
  CLASS_NAMES.push(dataCollectorButtons[i].getAttribute("data-name"));
}

let mobilenet = undefined;
let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;

/**
 * Loads the MobileNet model and warms it up so ready for use.
 **/
async function loadMobileNetFeatureModel() {
  const URL =
    "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1";
  // mobilenet = await tf.loadGraphModel(URL, {fromTFHub: true});

  movenet = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true });

  STATUS.innerText = "MobileNet v3 loaded successfully!";

  // Warm up the model by passing zeros through it once.
  tf.tidy(function () {
    let answer = movenet.predict(
      tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]).toInt()
    );
    console.log(answer.toString());
  });
}

loadMobileNetFeatureModel();

let model = tf.sequential();
model.add(
  tf.layers.dense({ inputShape: [1024], units: 128, activation: "relu" })
);
model.add(
  tf.layers.dense({ units: CLASS_NAMES.length, activation: "softmax" })
);

model.summary();

// Compile the model with the defined optimizer and specify a loss function to use.
model.compile({
  // Adam changes the learning rate over time which is useful.
  optimizer: "adam",
  // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
  // Else categoricalCrossentropy is used if more than 2 classes.
  loss:
    CLASS_NAMES.length === 2 ? "binaryCrossentropy" : "categoricalCrossentropy",
  // As this is a classification problem you can record accuracy in the logs too!
  metrics: ["accuracy"],
});

/**
 * Check if getUserMedia is supported for webcam access.
 **/
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

/**
 * Enable the webcam with video constraints applied.
 **/
function enableCam() {
  if (hasGetUserMedia()) {
    // getUsermedia parameters.
    const constraints = {
      video: true,
      width: 640,
      height: 480,
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
      VIDEO.srcObject = stream;
      VIDEO.addEventListener("loadeddata", function () {
        videoPlaying = true;
        ENABLE_CAM_BUTTON.classList.add("removed");
        STATUS.innerText = "camera loaded, button disappear";
      });
    });
  } else {
    console.warn("getUserMedia() is not supported by your browser");
  }
}

/**
 * Handle Data Gather for button mouseup/mousedown.
 **/
function gatherDataForClass() {
  let classNumber = parseInt(this.getAttribute("data-1hot"));
  gatherDataState =
    gatherDataState === STOP_DATA_GATHER ? classNumber : STOP_DATA_GATHER;
  dataGatherLoop();
}

async function calculateFeaturesOnCurrentFrame() {
  // Grab pixels from current VIDEO frame.

  let videoFrameAsTensor = tf.browser.fromPixels(VIDEO);
  // Resize video frame tensor to be 224 x 224 pixels which is needed by MobileNet for input.
  tf.browser.toPixels(videoFrameAsTensor, CANVAS);

  let resizedTensorFrame = tf.image.resizeBilinear(
    videoFrameAsTensor,
    [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
    true
  );

  resizedTensorFrame = tf.expandDims(resizedTensorFrame.toInt());
  console.log("resizedTensorFrame", resizedTensorFrame.shape.toString());
  console.log("resizedTensorFrame", tf.max(resizedTensorFrame).toString());

  let tensorOutput = movenet.predict(resizedTensorFrame);

  console.log("tensorOutput", tensorOutput.shape.toString());
  let arrayOutput = await tensorOutput.array();

  console.log("tensorOutput.array", tensorOutput.toString());

  return arrayOutput;
}

async function drawPoseOnFrame() {
  if (measurePose) {
    let pose = await calculateFeaturesOnCurrentFrame();

    console.log("typeof(pose)", typeof pose);

    for (let i = 0; i < pose[0][0].length; i++) {
      let y = pose[0][0][i][0];
      let x = pose[0][0][i][1];
      let score = pose[0][0][i][2];
      console.log("x", x.toString());
      console.log("y", y.toString());

      CTX.fillStyle = "#00ff00";
      CTX.beginPath();
      CTX.arc(x * 640, y * 480, 5, 0, 2 * Math.PI);
      CTX.fill();
    }
    window.requestAnimationFrame(drawPoseOnFrame);
  }
}

/**
 * When a button used to gather data is pressed, record feature vectors along with class type to arrays.
 **/
function dataGatherLoop() {
  // Only gather data if webcam is on and a relevent button is pressed.
  if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
    // Ensure tensors are cleaned up.
    let imageFeatures = calculateFeaturesOnCurrentFrame();

    trainingDataInputs.push(imageFeatures);
    trainingDataOutputs.push(gatherDataState);

    // Intialize array index element if currently undefined.
    if (examplesCount[gatherDataState] === undefined) {
      examplesCount[gatherDataState] = 0;
    }
    // Increment counts of examples for user interface to show.
    examplesCount[gatherDataState]++;

    STATUS.innerText = "";
    for (let n = 0; n < CLASS_NAMES.length; n++) {
      STATUS.innerText +=
        CLASS_NAMES[n] + " data count: " + examplesCount[n] + ". ";
    }

    window.requestAnimationFrame(dataGatherLoop);
  }
}

/**
 * Once data collected actually perform the transfer learning.
 **/
async function trainAndPredict() {
  predict = false;
  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);

  let outputsAsTensor = tf.tensor1d(trainingDataOutputs, "int32");
  let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
  let inputsAsTensor = tf.stack(trainingDataInputs);

  let results = await model.fit(inputsAsTensor, oneHotOutputs, {
    shuffle: true,
    batchSize: 5,
    epochs: 10,
    callbacks: { onEpochEnd: logProgress },
  });

  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();

  predict = true;
  predictLoop();
}

/**
 * Log training progress.
 **/
function logProgress(epoch, logs) {
  console.log("Data for epoch " + epoch, logs);
}

/**
 *  Make live predictions from webcam once trained.
 **/
function predictLoop() {
  if (predict) {
    tf.tidy(function () {
      let imageFeatures = calculateFeaturesOnCurrentFrame();
      let prediction = model.predict(imageFeatures.expandDims()).squeeze();
      let highestIndex = prediction.argMax().arraySync();
      let predictionArray = prediction.arraySync();
      STATUS.innerText =
        "Prediction: " +
        CLASS_NAMES[highestIndex] +
        " with " +
        Math.floor(predictionArray[highestIndex] * 100) +
        "% confidence";
    });

    window.requestAnimationFrame(predictLoop);
  }
}

/**
 * Purge data and start over. Note this does not dispose of the loaded
 * MobileNet model and MLP head tensors as you will need to reuse
 * them to train a new model.
 **/
function reset() {
  predict = false;
  examplesCount.splice(0);
  for (let i = 0; i < trainingDataInputs.length; i++) {
    trainingDataInputs[i].dispose();
  }
  trainingDataInputs.splice(0);
  trainingDataOutputs.splice(0);
  STATUS.innerText = "No data collected";

  console.log("Tensors in memory: " + tf.memory().numTensors);
}
