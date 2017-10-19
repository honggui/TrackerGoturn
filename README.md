# TrackerGoturn
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

TrackerGoturn is a project that is maintained by **OPEN** AI LAB, it used [Caffe](http://caffe.berkeleyvision.org/) platform to training a SqueezeNet and provide a camera live demo.

The release version is 0.1.0, is based on [Rockchip RK3399](http://www.rock-chips.com/plus/3399.html) Platform, target OS is Ubuntu 16.04. Can download the source code from [OAID/TrackerGoturn](https://github.com/OAID/TrackerGoturn)

* The ARM Computer Vision and Machine Learning library is a set of functions optimised for both ARM CPUs and GPUs using SIMD technologies. See also [Arm Compute Library](https://github.c
om/ARM-software/ComputeLibrary).

### Documents
* [Installation instructions](https://github.com/OAID/CaffeOnACL/blob/master/acl_openailab/installation.md)
* [Performance Report PDF](https://github.com/OAID/CaffeOnACL/blob/master/acl_openailab/performance_report.pdf)

### Arm Compute Library Compatibility Issues :
There are some compatibility issues between ACL and Caffe Layers, we bypass it to Caffe's original layer class as the workaround solution for the below issues

* Normalization in-channel issue
* Tanh issue
* Softmax supporting multi-dimension issue
* Group issue

Performance need be fine turned in the future
