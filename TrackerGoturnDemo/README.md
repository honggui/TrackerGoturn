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
# Installation Guide
[Installation instructions](https://github.com/OAID/TrackerGoturn/blob/master/TrackerGoturnDemo/INSTALL.md)
