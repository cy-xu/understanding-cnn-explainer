// define key constants 

const STATUS = document.getElementById('status');
const VIDEO = document.getElementById('webcam');
const ENABLE_CAM_BUTTON = document.getElementById('enableCam');
const RESET_BUTTON = document.getElementById('reset');
const TRAIN_BUTTON = document.getElementById('train');
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = [];

// key event listeners

ENABLE_CAM_BUTTON.addEventListener('click', enableCam);
TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', reset);

function enableCam() {
  // TODO: Fill this out later in the codelab!
}

function trainAndPredict() {
  // TODO: Fill this out later in the codelab!
}

function reset() {
  // TODO: Fill this out later in the codelab!
}

// Add event listeners for each of the class buttons
let dataCollectorButtons = document.querySelectorAll('button.dataCollector');

for (let i = 0; i < dataCollectorButtons.length; i++) {
  dataCollectorButtons[i].addEventListener('mousedown', gatherDataForClass);
  dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);
  // Populate the human readable names for classes.
  CLASS_NAMES.push(dataCollectorButtons[i].getAttribute('data-name'));
}

function gatherDataForClass() {
  // TODO: Fill this out later in the codelab!
}

// other variables

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
      'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
    
    mobilenet = await tf.loadGraphModel(URL, {fromTFHub: true});
    // change html message to show model is loaded
    STATUS.innerText = 'MobileNet v3 loaded successfully!';
    
    // Warm up the model by passing zeros through it once.
    // this sends fake data of zeros so the model is loaded when user requests it
    tf.tidy(function () {
      let answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
      console.log(answer.shape);
    });
}
  
// Call the function immediately to start loading.
loadMobileNetFeatureModel();

// Following step 9, https://codelabs.developers.google.com/tensorflowjs-transfer-learning-teachable-machine#8
// This should load the model but if I run locally it doesn't load the model