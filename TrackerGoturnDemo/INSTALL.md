# 1. User Quick Guide
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

This Installation will help you get started to setup TrackerGoturnDemo on RK3399 quickly.

# 2. Preparation
## 2.1 CMake installation
	sudo apt-get install cmake

## 2.2 OpenCV3.3 installation

	wget https://github.com/opencv/opencv/archive/3.3.0.zip --no-check-certificate
  	unzip 3.30.zip
  	cd opencv-3.3.0
  	mkdir build
  	cd build
  	cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local/opencv ..
  	make -j4
 	 sudo make install
  	sudo ldconfig
  
  After installed, you can see the "bin","lib","include" and "share" dirs in installed dir: /usr/local/opencv

## 2.3 Install CaffeOnACL and compile using the CMake build instructions:
  	Referenced by https://github.com/OAID/CaffeOnACL
  	Modify the Makefile "LIBRARIES += glog gflags protobuf leveldb snappy lmdb boost_system hdf5_hl hdf5 m opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs",if errors ocurred.

## 2.4 TinyXML installation(needed to load Imagenet annotations):
	sudo apt-get install libtinyxml-dev 

## 2.4 Trax installation:
	cd ~
	git clone https://github.com/votchallenge/trax.git
	cd trax
	mkdir build
	cd build
	cmake ..
	make

## 2.5 TrackerGoturnDemo Environment check:
  From TrackerGoturnDemo main directory, to view CMakeLists.txt file and check the following env vars according your installed paths

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

# 3. Build TrackerGoturnDemo
  From GOTURN_demo main directory, type:

  ```
  mkdir build
  cd build
  cmake ..
  make
  ```
  
# 4.Camera live demo 
```
1) Change to camera directory , type as "cd GOTURN_demo/src/capture" directory, 
2) Check the trax, caffeOnACL, opencv and ACL lib path in "build.sh" scripte file
3) Run "build.sh" to generate exec files
4) Plugin USB camera device
5) Check the caffe prototxt file, caffemodel file,trax, caffeOnACL and ACL lib path in "run.sh" script file
6) Run "run.sh" to startup the demo, then click mouse to select target object for tracking
```
