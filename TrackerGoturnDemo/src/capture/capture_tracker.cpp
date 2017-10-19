#include <string>
#include <signal.h>


#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "helper/high_res_timer.h"
#include "network/regressor.h"
//#include "loader/loader_alov.h"
#include "loader/loader_vot.h"
#include "native/vot.h"
#include "tracker/tracker.h"
#include "tracker/tracker_manager.h"
  
using namespace cv;  
  
Mat org,img;
int state;  
VOTRegion region;

volatile bool sgFlag=true;

void on_mouse(int event,int x,int y,int flags,void *ustc)//event鼠标事件代号，x,y鼠标坐标，flags拖拽和键盘操作的代号  
{  
    static Point pre_pt = Point(-1,-1);//初始坐标  
    static Point cur_pt = Point(-1,-1);//实时坐标  
    char temp[16];  
    if (event == CV_EVENT_LBUTTONDOWN)//左键按下，读取初始坐标，并在图像上该点处划圆  
    {  
        state = 0;
        org.copyTo(img);//将原始图片复制到img中  
        sprintf(temp,"(%d,%d)",x,y);  
        pre_pt = Point(x,y);  
        putText(img,temp,pre_pt,FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,0,255),1,8);//在窗口上显示坐标  
        circle(img,pre_pt,2,Scalar(255,0,0,0),CV_FILLED,CV_AA,0);//划圆  
        imshow("img",img);  
    }  
    else if (event == CV_EVENT_MOUSEMOVE && !(flags & CV_EVENT_FLAG_LBUTTON))//左键没有按下的情况下鼠标移动的处理函数  
    {  
        //img.copyTo(tmp);//将img复制到临时图像tmp上，用于显示实时坐标  
        sprintf(temp,"(%d,%d)",x,y);  
        cur_pt = Point(x,y);  
        putText(img,temp,cur_pt,FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,0,255));//只是实时显示鼠标移动的坐标  
        imshow("img",img);  
    }  
    else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))//左键按下时，鼠标移动，则在图像上划矩形  
    {  
        //img.copyTo(tmp);  
        sprintf(temp,"(%d,%d)",x,y);  
        cur_pt = Point(x,y);  
        putText(img,temp,cur_pt,FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,0,255));  
        rectangle(img,pre_pt,cur_pt,Scalar(0,255,0,0),1,8,0);//在临时图像上实时显示鼠标拖动时形成的矩形  
        imshow("img",img);  
    }  
    else if (event == CV_EVENT_LBUTTONUP)//左键松开，将在图像上划矩形  
    {  
        //org.copyTo(img);  
        sprintf(temp,"(%d,%d)",x,y);  
        cur_pt = Point(x,y);  
        putText(img,temp,cur_pt,FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,0,255));  
        circle(img,pre_pt,2,Scalar(255,0,0,0),CV_FILLED,CV_AA,0);  
        rectangle(img,pre_pt,cur_pt,Scalar(0,255,0,0),1,8,0);//根据初始点和结束点，将矩形画到img上  
        imshow("img",img);   
        //截取矩形包围的图像，并保存到dst中  
        int width = abs(pre_pt.x - cur_pt.x);  
        int height = abs(pre_pt.y - cur_pt.y);  
        if (width == 0 || height == 0)  
        {  
            state = 0;
            printf("width == 0 || height == 0");  
            return;  
        }
        state = 1;
        //init region
        region.set_x(min(cur_pt.x,pre_pt.x));
        region.set_y(min(cur_pt.y,pre_pt.y));
        region.set_width(width);
        region.set_height(height); 
    }  
}  

void sig_user_interrupt(int sig, siginfo_t *info, void * arg)
{
     std::cout<<"User interrput the program ...\n"<<std::endl;
     sgFlag=false;
}

int main (int argc, char *argv[]) {
    if (argc < 3) {
      std::cerr << "Usage: " << argv[0]
                << " deploy.prototxt network.caffemodel"
                << " [gpu_id]" << std::endl;
      return 1;
    }
  
    ::google::InitGoogleLogging(argv[0]);
  
    const string& model_file   = argv[1];
    const string& trained_file = argv[2];
  
    int gpu_id = 0;
    if (argc >= 4) {
      gpu_id = atoi(argv[3]);
    }
  
    const bool do_train = false;
    Regressor regressor(model_file, trained_file, gpu_id, do_train);
  
    // Ensuring randomness for fairness.
    srandom(time(NULL));
  
  
    // Create a tracker object.
    const bool show_intermediate_output = false;
    Tracker tracker(show_intermediate_output);
    
    VideoCapture cap(0);
    if(!cap.isOpened())
        std::cout << "Error: Capture open failed!" << std::endl;
    namedWindow("img");//定义一个img窗口 
    setMouseCallback("img",on_mouse,0);//调用回调函数
    cap >> org;
    imshow("img", org);

    struct sigaction sa;    
    //bool sgFlag = true;

    sa.sa_sigaction=sig_user_interrupt;
    sa.sa_flags=SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);


    while(sgFlag)
    {
        cap >> org;
        org.copyTo(img);
        if(state == 1)
        {
            // Load the first frame and use the initialization region to initialize the tracker.
            tracker.Init(org, region, &regressor);
            imshow("img", org);
            state = 2;
            cv::waitKey(30);
        }
        else if(state == 2)
        {           
            // Track and estimate the bounding box location.
            BoundingBox bbox_estimate;
            tracker.Track(org, &regressor, &bbox_estimate);  
            
            rectangle(org, Point(bbox_estimate.x1_, bbox_estimate.y1_), \
                      Point(bbox_estimate.x2_, bbox_estimate.y2_), Scalar(0,0,255), 1, 8, 0);
            imshow("img", org);
            waitKey(5);
        }
        else
        { 
            std::cout << "waitkey: " << waitKey(30) << std::endl; 
            imshow("img", org);
        }
    }

    return 0;
    
}  
