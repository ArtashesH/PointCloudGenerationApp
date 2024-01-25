#include "opencv2/opencv.hpp"
#include <filesystem>
#include <iostream>

#include<direct.h>
using namespace cv;

int main(int argc, char* argv[])
{

    if (argc != 2) {
        return 0;
    }
    std::string videoFilePath(argv[1]);
    std::size_t pos = videoFilePath.find(".");
    std::string dirPath = videoFilePath.substr(0, pos);
    std::string commandCrDir = "mkdir " + dirPath;
    int status = std::system(commandCrDir.c_str());
    if ((status < 0) && (errno != EEXIST));
  
   

    VideoCapture cap(videoFilePath); // open the default camera
    if (!cap.isOpened())  // check if we succeeded
        return -1;

    //Ptr<BackgroundSubtractor> pMOG = new BackgroundSubtractorMOG2();

    Mat fg_mask;
    Mat frame;
    int count = -1;
    int writeFrameCount = 0;
    int FPS = cap.get(CAP_PROP_FPS);
    std::cout << "|FPS of cideo " << FPS << std::endl;
    return 0;
    for (;;)
    {
        // Get frame
        cap >> frame; // get a new frame from camera

        // Update counter
        ++count;
       // if (count % 5 != 0) {
        //    continue;
        //}
        // Background subtraction
       // pMOG->operator()(frame, fg_mask);
        std::string outputImagePath = dirPath + "/" + std::to_string(count) + ".png";
        std::cout << outputImagePath << std::endl;
       // if (frame.rows == 0 || frame.cols == 0) {
        //    continue;
       // }
        imwrite(outputImagePath, frame);
        ++writeFrameCount;

       // if (waitKey(1) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}