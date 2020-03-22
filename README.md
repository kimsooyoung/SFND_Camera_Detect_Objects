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

> Those parameters and datasets provided from KITTI sensor setup

  3. Transform points back into Euclidean coordinates and store the result.

---
 ### Reference
  * [Darknet](https://github.com/pjreddie/darknet)
  * [Udacity Sensor Fusion Nanodegree](https://www.udacity.com/course/sensor-fusion-engineer-nanodegree--nd313)

