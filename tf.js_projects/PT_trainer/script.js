// import * as poseDetection from '@tensorflow-models/pose-detection';

import { Camera } from "./camera.js";
import { STATE, BLAZEPOSE_CONFIG, MOVENET_CONFIG, VIDEO_SIZE } from "./params.js";

/* globals tf */

const MODEL_PATH =
  "https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4";
const EXAMPLE_IMG = document.getElementById("exampleImg");
const CANVAS = document.getElementById("testCanvas");
const CTX = CANVAS.getContext("2d");

const CANVAS_THUMB = document.getElementById("thumbnails");
const CTX_THUMB = CANVAS_THUMB.getContext("2d");

let detector, camera, currentThumbnail;
let currentPose = null;
let initializationTime = 0;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let measurePose = true;
let predict = false;
let collectingPose = false;
let rafId, nIntervId;
let local_model;

let featureLength = 34;
let poseClasses = 2;
let sampleCounter = 0;

// loadAndRunModel();

const INIT_TXT = document.getElementById("init_time");
const FPS_TXT = document.getElementById("framerate");
const DATA_TXT = document.getElementById("data_status");

const VIDEO = document.getElementById("video");
const ENABLE_CAM_BUTTON = document.getElementById("enableCam");
const RESET_BUTTON = document.getElementById("reset");
const START_COLLECTING = document.getElementById("startCollect");
const STOP_COLLECTING = document.getElementById("stopCollect");
const TRAIN_BUTTON = document.getElementById("train");
const MOBILE_NET_INPUT_WIDTH = 192;
const MOBILE_NET_INPUT_HEIGHT = 192;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = [];
const MIN_SAMPLES = 50;

// ENABLE_CAM_BUTTON.addEventListener("click", enableCam);
TRAIN_BUTTON.addEventListener("click", trainAndPredict);
RESET_BUTTON.addEventListener("click", reset);

START_COLLECTING.addEventListener("click", startGatherLoop);
STOP_COLLECTING.addEventListener("click", stopGatherLoop);

// start camera when page is loaded
// if (window.addEventListener) {
//   window.addEventListener("load", enableCam, false); //W3C
//   // window.addEventListener("load", drawPoseOnFrame, false); //not working.
// } else {
//   window.attachEvent("onload", enableCam); //IE
// }

// Just add more buttons in HTML to allow classification of more classes of data!
let dataCollectorButtons = document.querySelectorAll("button.dataCollector");

// for (let i = 0; i < dataCollectorButtons.length; i++) {
// dataCollectorButtons[i].addEventListener("mousedown", startGatherLoop);
// dataCollectorButtons[i].addEventListener("mouseup", drawPoseOnFrame);
// For mobile.
// dataCollectorButtons[i].addEventListener("touchend", gatherDataForClass);

// Populate the human readable names for classes.
// CLASS_NAMES.push(dataCollectorButtons[i].getAttribute("data-name"));
// }

let mobilenet = undefined;
let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];

/**
 * Loads the MobileNet model and warms it up so ready for use.
 **/
async function createDetector_old() {
  const URL =
    "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1";
  // mobilenet = await tf.loadGraphModel(URL, {fromTFHub: true});

  // model =

  FPS_TXT.innerText = "MoveNet loaded successfully!";

  return await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true });
}

async function warmUpModel(model) {
  // Warm up the model by passing zeros through it once.
  const zeros = tf
    .zeros([MOBILE_NET_INPUT_WIDTH, MOBILE_NET_INPUT_HEIGHT, 3])
    .toInt();
  // let answer = model.predict(zeros);
  console.log("zero shape", zeros.shape);

  let answer = await model.estimatePoses(zeros);
  console.log("Warm up model:", answer);
  console.log("Warm up model:", answer.toString());

  FPS_TXT.innerText = "MoveNet warmed up!";
}

function createLocalModel(channelIn, channelOut) {
  let model = tf.sequential();
  model.add(
    tf.layers.dense({ inputShape: [channelIn], units: 128, activation: "relu" })
  );
  model.add(
    tf.layers.dense({ units: channelOut, activation: "softmax" })
  );

  model.summary();

  // Compile the model with the defined optimizer and specify a loss function to use.
  model.compile({
    // Adam changes the learning rate over time which is useful.
    optimizer: "adam",
    // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
    // Else categoricalCrossentropy is used if more than 2 classes.
    loss:
    poseClasses === 2
        ? "binaryCrossentropy"
        : "categoricalCrossentropy",
    // As this is a classification problem you can record accuracy in the logs too!
    metrics: ["accuracy"],
  });
  return model;
}

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
        FPS_TXT.innerText = "camera loaded, button disappear";
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
  // console.log("resizedTensorFrame", resizedTensorFrame.shape.toString());
  // console.log("resizedTensorFrame", tf.max(resizedTensorFrame).toString());

  let tensorOutput = detector.predict(resizedTensorFrame);

  // console.log("tensorOutput", tensorOutput.shape.toString());
  let arrayOutput = await tensorOutput.array();

  // console.log("tensorOutput.array", tensorOutput.toString());

  return arrayOutput;
}

function extrctPoseValues() {
  if (sampleCounter >= MIN_SAMPLES) {
    stopGatherLoop();

    // draw this last pose to canvas for visualisation.
    CANVAS_THUMB.width = 0.5*camera.video.videoWidth;
    CANVAS_THUMB.height = 0.5*camera.video.videoHeight;

    currentThumbnail = camera.video;
      CTX_THUMB.drawImage(
        camera.video, 0, 0, 0.5*camera.video.videoWidth, 0.5*camera.video.videoHeight);
      return;
  }

  let poseValues = [];

  for (let i = 0; i < currentPose.keypoints.length; i++) {
    poseValues.push(currentPose.keypoints[i].x);
    poseValues.push(currentPose.keypoints[i].y);
  }

  trainingDataInputs.push(poseValues);
  featureLength = poseValues.length;

  if (collectingPose){
    // collect real pose
    trainingDataOutputs.push(1);
    sampleCounter++;
  } else {
    // collect background
    trainingDataOutputs.push(0);
  }
  
  console.log("pose_values", poseValues);
  DATA_TXT.innerText = "Collecting pose samples: " + trainingDataInputs.length;
}

function beginEstimatePosesStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimatePosesStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    // stats.customFpsPanel.update(
        // 1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
    const currFPS = Math.floor(1000.0 / averageInferenceTime);
    FPS_TXT.innerText = "FPS: " + currFPS;
    // console.log("currFPS", currFPS);
  }
}

async function renderResult() {
  if (camera.video.readyState < 2) {
    await new Promise((resolve) => {
      camera.video.onloadeddata = () => {
        resolve(video);
      };
    });
  }

  let poses = null;

  // Detector can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (detector != null) {
    // FPS only counts the time it takes to finish estimatePoses.
    beginEstimatePosesStats();

    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      poses = await detector.estimatePoses(camera.video, {
        maxPoses: STATE.modelConfig.maxPoses,
        flipHorizontal: false,
      });
    } catch (error) {
      alert(error);
      detector.dispose();
      detector = null;
    }

    endEstimatePosesStats();
  }

  camera.drawCtx();

  // log the app initialization time only once.
  if (currentPose == null){
    initializationTime = Math.floor(performance.now() - initializationTime);
    INIT_TXT.innerText =  "Initialization time: " + initializationTime + " ms";
    // console.log("initializationTime", initializationTime);
  }

  // if current learning the new pose, put them into a list
  // if (collectPose && poses.length > 0) {
  if (poses.length > 0) {
    // poses[0].keypoints
    currentPose = poses[0];
  }

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old model,
  // which shouldn't be rendered.
  if (poses && poses.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(poses);
  }

  if (predict){
    predictLoop(currentPose);
  }

  rafId = requestAnimationFrame(renderResult);
}

async function drawPoseOnFrame() {
  if (measurePose) {
    let pose = await calculateFeaturesOnCurrentFrame();

    // console.log("typeof(pose)", typeof pose);

    for (let i = 0; i < pose[0][0].length; i++) {
      let y = pose[0][0][i][0];
      let x = pose[0][0][i][1];
      let score = pose[0][0][i][2];
      // console.log("x", x.toString());
      // console.log("y", y.toString());

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

    // STATUS.innerText = "";
    for (let n = 0; n < CLASS_NAMES.length; n++) {
      INIT_TXT.innerText +=
        CLASS_NAMES[n] + " data count: " + examplesCount[n] + ". ";
    }

    window.requestAnimationFrame(dataGatherLoop);
  }
}

function startGatherLoop() {
  // Only gather data if webcam is on and a relevent button is pressed.
  // window.cancelAnimationFrame(rafId);
  collectingPose = true;
  nIntervId = setInterval(extrctPoseValues, 100);
}

function stopGatherLoop() {
  collectingPose = false;

  // cancel interval and stop collecting data
  clearInterval(nIntervId);
  nIntervId = null;
  DATA_TXT.innerText =
    "Data collection done -- samples: " + trainingDataInputs.length;
}

/**
 * Once data collected actually perform the transfer learning.
 **/
async function trainAndPredict() {

  predict = false;
  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);

  console.log("trainingDataOutputs", trainingDataOutputs);

  let outputsAsTensor = tf.tensor1d(trainingDataOutputs, "int32");
  let oneHotOutputs = tf.oneHot(outputsAsTensor, poseClasses);
  let inputsAsTensor = tf.stack(trainingDataInputs);

  let results = await local_model.fit(inputsAsTensor, oneHotOutputs, {
    shuffle: true,
    batchSize: 5,
    epochs: 10,
    callbacks: { onEpochEnd: logProgress },
  });

  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();

  predict = true;
  debugger;

  // predictLoop();
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
function predictLoop(pose) {
    tf.tidy(function () {
      let poseValues = [];

      for (let i = 0; i < pose.keypoints.length; i++) {
        poseValues.push(currentPose.keypoints[i].x);
        poseValues.push(currentPose.keypoints[i].y);
      }
      poseTensor = tf.tensor2d([poseValues]);
      console.log("poseTensor shape", poseTensor.shape);

      let prediction = model.predict(poseValues.expandDims()).squeeze();
      let highestIndex = prediction.argMax().arraySync();
      let predictionArray = prediction.arraySync();
      DATA_TXT.innerText =
        "Prediction: " +
        highestIndex +
        " with " +
        Math.floor(predictionArray[highestIndex] * 100) +
        "% confidence";
    });

    // window.requestAnimationFrame(predictLoop);
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
  DATA_TXT.innerText = "No data collected";

  console.log("Tensors in memory: " + tf.memory().numTensors);
}

async function createMoveNetDetector() {
  let modelType;
  STATE.modelConfig = MOVENET_CONFIG;
  STATE.model = poseDetection.SupportedModels.MoveNet;

  if (STATE.modelConfig.type == "lightning") {
    modelType = poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING;
  } else if (STATE.modelConfig.type == "thunder") {
    modelType = poseDetection.movenet.modelType.SINGLEPOSE_THUNDER;
  } else if (STATE.modelConfig.type == "multipose") {
    modelType = poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING;
  }
  const modelConfig = { modelType };

  if (STATE.modelConfig.customModel !== "") {
    modelConfig.modelUrl = STATE.modelConfig.customModel;
  }
  if (STATE.modelConfig.type === "multipose") {
    modelConfig.enableTracking = STATE.modelConfig.enableTracking;
  }
  return poseDetection.createDetector(STATE.model, modelConfig);
}

async function createBlazePoseDetector() {
  STATE.modelConfig = BLAZEPOSE_CONFIG;
  STATE.model = poseDetection.SupportedModels.BlazePose;
  const detectorConfig = {
    runtime: "tfjs",
    enableSmoothing: true,
    modelType: "full",
  };

  return poseDetection.createDetector(STATE.model, detectorConfig);
}

async function app() {
  initializationTime = performance.now();

  camera = await Camera.setupCamera(STATE.camera);
  local_model = createLocalModel(featureLength, poseClasses);
  // detector = await createBlazePoseDetector();
  detector = await createMoveNetDetector();

  tf.engine().startScope();
  // do your thing
  // warmUpModel(movenet);

  // drawPoseOnFrame();
  renderResult();

  tf.engine().endScope();
}

app();
