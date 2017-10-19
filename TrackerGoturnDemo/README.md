# TrackerGoturn Demo
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

This is the code for our TrackerGoturn demo, using GOTURN: Generic Object Tracking Using Regression Networks.

GOTURN appeared in this paper:

**[Learning to Track at 100 FPS with Deep Regression Networks](http://davheld.github.io/GOTURN/GOTURN.html)**,
<br>
[David Held](http://davheld.github.io/),
[Sebastian Thrun](http://robots.stanford.edu/),
[Silvio Savarese](http://cvgl.stanford.edu/silvio/),
<br>
European Conference on Computer Vision (ECCV), 2016 (In press)

GOTURN addresses the problem of **single target tracking**: given a bounding box label of an object in the first frame of the video, we track that object through the rest of the video.  

Note that our current method does not handle occlusions; however, it is fairly robust to viewpoint changes, lighting changes, and deformations.  Here is a brief overview of how our system works:

<img src="imgs/pull7f-web_e2.png" width=85%>

Using a collection of videos and images with bounding box labels (but no
class information), we train a neural network to track generic objects. At test time, the
network is able to track novel objects without any fine-tuning. By avoiding fine-tuning,
our network is able to track at 100 fps.

If you find this code useful in your research, please cite:

```
@inproceedings{held2016learning,
  title={Learning to Track at 100 FPS with Deep Regression Networks},
  author={Held, David and Thrun, Sebastian and Savarese, Silvio},
  booktitle = {European Conference Computer Vision (ECCV)},
  year      = {2016}
}
```

Contents:
* [Installation](#installation)
* [Compile GOTURN demo](#Compile)
* [Camera live demo](#Camera live demo)

## Installation

### Install dependencies:

* Install CMake:
```
sudo apt-get install cmake
```

* Install OpenCV3.3
```
wget https://github.com/opencv/opencv/archive/3.3.0.zip --no-check-certificate
unzip 3.30.zip
cd opencv-3.3.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local/opencv ..
make -j4
sudo make install
sudo ldconfig
```

After installed, you can see the "bin","lib","include" and "share" dirs in installed dir: /usr/local/opencv

* Install CaffeOnACL and compile using the CMake build instructions:
```
Referenced by https://github.com/OAID/CaffeOnACL
Modify the Makefile "LIBRARIES += glog gflags protobuf leveldb snappy lmdb boost_system hdf5_hl hdf5 m opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs",if errors ocurred.
```

* Install TinyXML (needed to load Imagenet annotations):
```
sudo apt-get install libtinyxml-dev 
```
* Install trax
```
cd ~
git clone https://github.com/votchallenge/trax.git
cd trax
mkdir build
cd build
cmake ..
make
```

* GOTURN_demo Environment check

From GOTURN_demo main directory, to view CMakeLists.txt file and check the following env vars according your installed paths

```
find_package(Caffe REQUIRED)
set(Caffe_DIR /home/firefly/caffeOnACL/distribute)
set(Caffe_INCLUDE_DIR /home/firefly/caffeOnACL/distribute/include)
set(Caffe_LIBRARIES /home/firefly/caffeOnACL/build/lib/libcaffe.so)
set(ACL_LIBRARIES /home/firefly/ComputeLibrary/build/libarm_compute.so)
set(TRAX_LIBRARIES /home/firefly/trax/build/libtrax.so)
set(Caffe_DEFINITIONS –DCPU_ONLY)

```
And add "set(OpenCV_DIR "/usr/local/opencv/share/OpenCV")" at the 3rd line in the CMakeList.txt file if OPENCV_DIR cannot be found.

## GOTURN_demo Compile
From GOTURN_demo main directory, type:

```
mkdir build
cd build
cmake ..
make
```

## Camera live demo 
```
1) Change to camera directory , type as "cd GOTURN_demo/src/capture" directory, 
2) Check the trax, caffeOnACL, opencv and ACL lib path in "build.sh" scripte file
3) Run "build.sh" to generate exec files
4) Plugin USB camera device
5) Check the caffe prototxt file, caffemodel file,trax, caffeOnACL and ACL lib path in "run.sh" script file
6) Run "run.sh" to startup the demo, then click mouse to select target object for tracking
```



