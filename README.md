# Sensor Fusion Self-Driving Car Course - Detect Objects from Camera Images

#### Automatically identify the vehicles in the scene using object detection with [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

### Project Status:

![issue_badge](https://img.shields.io/badge/build-Passing-green) ![issue_badge](https://img.shields.io/badge/UdacityRubric-Passing-green)

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* yolo - pre-trained network’s weights
  * 'yolov3.cfg' : contains network configuration see more detail in [here](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
  * coco.names : contains the 80 different class names used in the [COCO dataset](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

## Project Overview

#### Here's Main workflow steps

### 1. Initialize the Parameters

* Every bounding box predicted by YOLOv3 is associated with a confidence score. The parameter 'confThreshold' is used to remove all bounding boxes with a lower score value.

* Then, a non-maximum suppression is applied to the remaining bounding boxes. The NMS procedure is controlled by the parameter ‚nmsThreshold‘.

* The size of the input image is controlled by the parameters ‚inpWidth‘ and ‚inpHeight‘, which is set to 416 as proposed by the YOLO authors. Other values could e.g. be 320 (faster) or 608 (more accurate).

### 2. Prepare the Model

* load the model weights as well as the associated model configuration
```c++
    // load image from file
    cv::Mat img = cv::imread("./images/img1.png");

    // load class names from file
    string yoloBasePath = "./dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights"; 

    vector<string> classes;
    ifstream ifs(yoloClassesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // load neural network
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(yoloModelConfiguration, yoloModelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
```

If OpenCV is built with Intel’s Inference Engine, `DNN_BACKEND_INFERENCE_ENGINE` should be used instead. 

The target is set to CPU in the code, as opposed to using `DNN_TARGET_OPENCL`, which would be the method of choice if a (Intel) GPU was available.

### 3. Generate 4D Blob from Input Image

Blob is the standard array and unified memory interface for many frameworks, including Caffe. A blob is a wrapper over the actual data being processed and passed along and also provides synchronization capability between the CPU and the GPU

More details on blobs can be found [here - Tutorial(net_layer_blob)](http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html)


More details about actual implementation can be found [here - Math Kernel Library for Deep Neural Networks](https://intel.github.io/mkl-dnn/understanding_memory_formats.html)

```c++
    // generate 4D blob from input image
    cv::Mat blob;
    double scalefactor = 1/255.0;
    cv::Size size = cv::Size(416, 416);
    cv::Scalar mean = cv::Scalar(0,0,0);
    bool swapRB = false;
    bool crop = false;
    cv::dnn::blobFromImage(img, blob, scalefactor, size, mean, swapRB, crop);
```

### 3. Generate 4D Blob from Input Image

The code below shows how an image loaded from the file is passed through the `blobFromImage` function to be converted into an input block for the neural network. 

```c++
    // generate 4D blob from input image
    cv::Mat blob;
    double scalefactor = 1/255.0;
    cv::Size size = cv::Size(416, 416);
    cv::Scalar mean = cv::Scalar(0,0,0);
    bool swapRB = false;
    bool crop = false;
    cv::dnn::blobFromImage(img, blob, scalefactor, size, mean, swapRB, crop);
```

The pixel values are scaled with a scaling factor of 1/255 to a target range of 0 to 1. 

It also adjusts the size of the image to the specified size of (416, 416, 416) without cropping.

The output blob will be passed as input to the network. 

Then, a forward pass will be executed to obtain a list of predicted bounding boxes as output from the network. 

These boxes go through a post-processing step to filter out those with low confidence values.

#### 4. Run Forward Pass Through the Network

Run the forward-function of OpenCV to perform a single forward-pass through the network. 

For that, It's needed to identify the last layer of the network and provide the associated internal names to the function. 

This can be done by using the OpenCV function `getUnconnectedOutLayers`, which gives the names of all unconnected output layers, which are in fact the last layers of the network.


The following code shows how this can be achieved:

```c++
    // Get names of output layers
    vector<cv::String> names;
    vector<int> outLayers = net.getUnconnectedOutLayers(); // get indices of output layers, i.e. layers with unconnected outputs
    vector<cv::String> layersNames = net.getLayerNames(); // get names of all layers in the network

    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i) // Get the names of the output layers in names
    {
        names[i] = layersNames[outLayers[i] - 1];
    }

    // invoke forward propagation through network
    vector<cv::Mat> netOutput;
    net.setInput(blob);
    net.forward(netOutput, names);
```

The output of the network is a vector of size C (the number of blob classes) with the first four elements in each class representing the center in x, the center in y as well as the width and height of the associated bounding box. 

The fifth element represents the trust or confidence that the respective bounding box actually encloses an object. 

The remaining elements of the matrix are the confidence associated with each of the classes contained in the `coco.cfg` file. 

Each box is assigned to the class corresponding to the highest confidence.

Here's an example of confidence format from [`yolov3.cfg`](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)

```
[net]
# Testing
# batch=1
# subdivisions=1
# Training
batch=64
subdivisions=16
width=608
height=608
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1
...
```

The following code shows how to scan through the network results and assemble the bounding boxes with a sufficiently high confidence score into a vector. 

The function `cv::minMaxLoc` finds the minimum and maximum element values and their positions with extremums searched across the whole array.

```c++
    // Scan through all bounding boxes and keep only the ones with high confidence
    float confThreshold = 0.20;
    vector<int> classIds;
    vector<float> confidences;
    vector<cv::Rect> boxes;
    for (size_t i = 0; i < netOutput.size(); ++i)
    {
        float* data = (float*)netOutput[i].data;
        for (int j = 0; j < netOutput[i].rows; ++j, data += netOutput[i].cols)
        {
            cv::Mat scores = netOutput[i].row(j).colRange(5, netOutput[i].cols);
            cv::Point classId;
            double confidence;

            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classId);
            if (confidence > confThreshold)
            {
                cv::Rect box; int cx, cy;
                cx = (int)(data[0] * img.cols);
                cy = (int)(data[1] * img.rows);
                box.width = (int)(data[2] * img.cols);
                box.height = (int)(data[3] * img.rows);
                box.x = cx - box.width/2; // left
                box.y = cy - box.height/2; // top

                boxes.push_back(box);
                classIds.push_back(classId.x);
                confidences.push_back((float)confidence);
            }
        }
    }
```

### 5. Post-Processing of Network Output

Apply non-maximum suppression, for remove redundant bounding boxes. 

The following figure shows the results, where green indicates preserved bounding boxes while red bounding boxes have been removed during NMS.

<img width="766" alt="nms_detect" src="https://user-images.githubusercontent.com/12381733/77242635-d75c9900-6c43-11ea-88eb-5301ba56aa8b.png">

---
 ### Reference
  * [Darknet](https://github.com/pjreddie/darknet)
  * [YOLO: Real Time Object Detection](https://github.com/pjreddie/darknet/wiki/YOLO:-Real-Time-Object-Detection)
  * [Udacity Sensor Fusion Nanodegree](https://www.udacity.com/course/sensor-fusion-engineer-nanodegree--nd313)

