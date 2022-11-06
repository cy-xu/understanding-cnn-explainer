#Notes on cnn.js and cnn-tf.js

The files cnn.js and cnn-tf.js contain the implementation of the CNN model using tensorflow.js which implements the CNN modules like convolution, pooling, flattening, etc. from scratch. 

The cnn.js file takes in the input images and corresponding labels and creates dictionaries that map together these values and dumps them into json files. Then individual methods for convolution operation, pooling operation, etc. were invoked with these json files to perform the  corresponding operation on the data. Now this implementation is different from that used in the tiny-vgg.py in that this entire re-implementation is focused towards the visualization component and this would interact with the javascript code for the webpage page to render the visualization in real time. Tensorflow.js provides this flexibility that enables this real time rendering faster and compatible with the browser. 

The basic difference between the cnn.js implementation and cnn-tf.js is that cnn.js executes in the background running the network through the data while cnn-tf.js renders the images, text and animation in the front end correlating the changes in the model with the visualization in real time.

This would help the developers to easily distinguish between the rendering component, the background network implementation and the actual network running. 

The CNN explainer is an excellent example of visualizing the neural networks in real time and this level of separation between various components could actually enable us to identify properly the areas of improvement to make it more flexible and real time.