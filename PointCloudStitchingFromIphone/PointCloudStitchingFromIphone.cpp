// PointCloudStitchingFromIphone.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "PointCloudStitchingFromIphone.h"

#include "Knn2BFMatcherBasedFeatureStore.hpp"
#include "KniftFeatureDetectorCalculator.hpp"
#include "Constants.hpp"

pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloudStitcher::createPointCloudFromRGBDepth(const cv::Mat& rgbImage, const cv::Mat& depthImage, const cv::Mat& calibMatrix)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new  pcl::PointCloud<pcl::PointXYZRGB>());


    cv::Mat points3d;


    for (int i = 0; i < depthImage.rows; ++i) {
        for (int j = 0; j < depthImage.cols; ++j) {
            pcl::PointXYZRGB tmpPoint;
          

            tmpPoint.b = rgbImage.at<cv::Vec3b>(i, j)[0];
            tmpPoint.g = rgbImage.at<cv::Vec3b>(i, j)[1];
            tmpPoint.r = rgbImage.at<cv::Vec3b>(i, j)[2];
          


            //tmpPoint.x = (((float)j*6.3 - calibMatrix.at<float>(0,2)) * (float)((depthImage.at<unsigned short>(i, j)))) / calibMatrix.at<float>(0,0);
            //tmpPoint.y = (((float)i*6.3 - calibMatrix.at<float>(1, 2)) * (float)((depthImage.at<unsigned short>(i, j)))) / calibMatrix.at<float>(1, 1);
            //tmpPoint.z = depthImage.at<unsigned short>(i, j);


            cv::Mat tmpPointL = depthImage.at<unsigned short>(i, j) * calibMatrix.inv() * cv::Vec3f((4032.0 * j)/640, (3024.0* i) /480, 1);

            tmpPoint.x = tmpPointL.at<float>(0, 0);
            tmpPoint.y = tmpPointL.at<float>(0, 1);
            tmpPoint.z = tmpPointL.at<float>(0, 2);
            

            pointCloud->push_back(tmpPoint);

        }
    }
    pointCloud->height = 1;
    pointCloud->width = pointCloud->points.size();
    pointCloud->is_dense = true;
    return pointCloud;

}


cv::Vec3d pointCloudStitcher::CalculateMean(const cv::Mat_<cv::Vec3d>& points)
{
    cv::Mat_<cv::Vec3d> result;
    cv::reduce(points, result, 0, cv::REDUCE_AVG);
    return result(0, 0);
}

//Own implementation of rigit transform calculation
cv::Mat_<double> pointCloudStitcher::FindRigidTransform(const cv::Mat_<cv::Vec3d>& points1, const cv::Mat_<cv::Vec3d> points2)
{
    /* Calculate centroids. */
    cv::Vec3d t1 = -CalculateMean(points1);
    cv::Vec3d t2 = -CalculateMean(points2);

    cv::Mat_<double> T1 = cv::Mat_<double>::eye(4, 4);
    T1(0, 3) = t1[0];
    T1(1, 3) = t1[1];
    T1(2, 3) = t1[2];

    cv::Mat_<double> T2 = cv::Mat_<double>::eye(4, 4);
    T2(0, 3) = -t2[0];
    T2(1, 3) = -t2[1];
    T2(2, 3) = -t2[2];

    /* Calculate covariance matrix for input points. Also calculate RMS deviation from centroid
    * which is used for scale calculation.
    */
    cv::Mat_<double> C(3, 3, 0.0);
    double p1Rms = 0, p2Rms = 0;
    for (int ptIdx = 0; ptIdx < points1.rows; ptIdx++)
    {
        cv::Vec3d p1 = points1(ptIdx, 0) + t1;
        cv::Vec3d p2 = points2(ptIdx, 0) + t2;
        p1Rms += p1.dot(p1);
        p2Rms += p2.dot(p2);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                C(i, j) += p2[i] * p1[j];
            }
        }
    }

    cv::Mat_<double> u, s, vh;
    cv::SVD::compute(C, s, u, vh);

    cv::Mat_<double> R = u * vh;

    if (cv::determinant(R) < 0)
    {
        R -= u.col(2) * (vh.row(2) * 2.0);
    }


    cv::Mat_<double> M = cv::Mat_<double>::eye(4, 4);
    R.copyTo(M.colRange(0, 3).rowRange(0, 3));

    cv::Mat_<double> result = T2 * M * T1;
    result /= result(3, 3);

    cv::Mat finalRes = result.rowRange(0, 3);
    //std::cout << "result=" << result << std::endl;

    return finalRes;
}



//Calculate euqclidean distance between 3d points

float pointCloudStitcher::distBetweenPoints(const cv::Point3f& point1, const cv::Point3f& point2)
{
    return std::sqrtf((point1.x - point2.x) * (point1.x - point2.x) + (point1.y - point2.y) * (point1.y - point2.y) + (point1.z - point2.z) * (point1.z - point2.z));
}



//Find matching points between two 2d images

corrPointsPairType pointCloudStitcher::alignImages(cv::Mat& im1, cv::Mat& im2)
{

 
    // Convert images to grayscale
    cv::Mat im1Gray, im2Gray;
    cv::Mat img1D = im1.clone();
    cv::Mat img2D = im2.clone();
   
    cv::cvtColor(im1, im1Gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(im2, im2Gray, cv::COLOR_BGR2GRAY);

    // Variables to store keypoints and descriptors
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    // Detect ORB features and compute descriptors.
   
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(MAX_FEATURES);
   
    orb->detectAndCompute(im1Gray, cv::Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(im2Gray, cv::Mat(), keypoints2, descriptors2);


    // Match features.
    std::vector<cv::DMatch> matches;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
   
    matcher->match(descriptors1, descriptors2, matches, cv::Mat());

    // Sort matches by score
    std::sort(matches.begin(), matches.end());

    // Remove not so good matches
    const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
    matches.erase(matches.begin() + numGoodMatches, matches.end());

    // Draw top matches
    cv::Mat imMatches;
    cv::drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
    cv::imshow("ImgMathces", imMatches);
    cv::waitKey(0);
    //cv::imwrite("matches.jpg", imMatches);

    // Extract location of good matches
    std::vector<cv::Point2f> points1, points2;
    corrPointsPairType correspondPointsPair;
    for (size_t i = 0; i < matches.size(); i++)
    {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }


    for (int i = 0; i < points1.size(); ++i) {
        cv::circle(img1D, points1[i], 5, cv::Scalar(0, 255, 0), -1);
    }
    for (int i = 0; i < points2.size(); ++i) {
        cv::circle(img2D, points2[i], 5, cv::Scalar(0, 255, 0), -1);
    }
    correspondPointsPair.first = points1;
    correspondPointsPair.second = points2;
    std::cout << "MAtched point clount " << points1.size() << "  second " << points2.size() << std::endl;

    return correspondPointsPair;
   
}


//Feature extraction and matching based on ML, this is not used in code right now.
corrPointsPairType pointCloudStitcher::alignImagesML(cv::Mat& im1, cv::Mat& im2)
{
    cv::Mat img1 = im1.clone();
    cv::Mat img2 = im2.clone();
    std::vector<cv::Point2f> points1, points2;
    corrPointsPairType correspondPointsPair;
    std::vector<std::pair<cv::Point2f, cv::Point2f >> resFinalMatches;
    std::string kniftModelPath = "KNIFT_MODEL/knift_float.tflite";
    KniftFeatureDetectorCalculator calculator(kniftModelPath.c_str());

    const FeatureStore::ImageInfo* targetInfo = calculator.generate("TargetImageName", im1);
    const FeatureStore::ImageInfo* queryInfo = calculator.generate("QueryImageName", im2);

    Knn2BFMatcherBasedFeatureStore matcher;
    matcher.add(targetInfo);

    std::vector<std::vector<cv::Point2f>> matchedKeypoints(2);
    matcher.matchSingleImageInfo(*queryInfo, matchedKeypoints);

    for (size_t i = 0; i < matchedKeypoints[0].size(); ++i) {
        // Get the coordinates of the current pair of keypoints
        const cv::Point2f& point1 = matchedKeypoints[0][i];
        const cv::Point2f& point2 = matchedKeypoints[1][i];
        std::pair<cv::Point2f, cv::Point2f> currMatch;
        points1.push_back(point1);
        points2.push_back(point2);
        resFinalMatches.push_back(currMatch);
        int r = rand() % 256;
        int g = rand() % 256;
        int b = rand() % 256;
        cv::circle(img1, point1, 5, cv::Scalar(r, g, b), cv::FILLED);
        cv::circle(img2, point2, 5, cv::Scalar(r, g, b), cv::FILLED);

     
    }
    cv::imwrite("ResImg1New.png", img1);
    cv::imwrite("ResImg2New.png", img2);
    correspondPointsPair.first = points1;
    correspondPointsPair.second = points2;
    return correspondPointsPair;

}


pointCloudStitcher::pointCloudStitcher()
{
    m_pointCloudStitchedLast.height = 0;
    m_pointCloudStitchedLast.width = 0;
}
pointCloudStitcher::~pointCloudStitcher()
{

}

pcl::PointCloud<pcl::PointXYZRGB> pointCloudStitcher::getLastPointCloud()
{
    return m_pointCloudStitchedLast;
}
cv::Vec3d _CalculateMean(const cv::Mat_<cv::Vec3d>& points)
{
    cv::Mat_<cv::Vec3d> result;
    cv::reduce(points, result, 0, cv::REDUCE_AVG);
    return result(0, 0);
}
cv::Mat_<double> _FindRigidTransform(const cv::Mat_<cv::Vec3d>& points1, const cv::Mat_<cv::Vec3d> points2)
{
    /* Calculate centroids. */
    cv::Vec3d t1 = -_CalculateMean(points1);
    cv::Vec3d t2 = -_CalculateMean(points2);

    cv::Mat_<double> T1 = cv::Mat_<double>::eye(4, 4);
    T1(0, 3) = t1[0];
    T1(1, 3) = t1[1];
    T1(2, 3) = t1[2];

    cv::Mat_<double> T2 = cv::Mat_<double>::eye(4, 4);
    T2(0, 3) = -t2[0];
    T2(1, 3) = -t2[1];
    T2(2, 3) = -t2[2];

    /* Calculate covariance matrix for input points. Also calculate RMS deviation from centroid
    * which is used for scale calculation.
    */
    cv::Mat_<double> C(3, 3, 0.0);
    double p1Rms = 0, p2Rms = 0;
    for (int ptIdx = 0; ptIdx < points1.rows; ptIdx++)
    {
        cv::Vec3d p1 = points1(ptIdx, 0) + t1;
        cv::Vec3d p2 = points2(ptIdx, 0) + t2;
        p1Rms += p1.dot(p1);
        p2Rms += p2.dot(p2);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                C(i, j) += p2[i] * p1[j];
            }
        }
    }

    cv::Mat_<double> u, s, vh;
    cv::SVD::compute(C, s, u, vh);

    cv::Mat_<double> R = u * vh;

    if (cv::determinant(R) < 0)
    {
        R -= u.col(2) * (vh.row(2) * 2.0);
    }


    cv::Mat_<double> M = cv::Mat_<double>::eye(4, 4);
    R.copyTo(M.colRange(0, 3).rowRange(0, 3));

    cv::Mat_<double> result = T2 * M * T1;
    result /= result(3, 3);

    cv::Mat finalRes = result.rowRange(0, 3);
  

    return finalRes;
}


//Stitch two point clouds, and stitch with the result of previous stitches.
void  pointCloudStitcher::stitchTwoPointClouds(const std::string& rgbImgPath1, const std::string& depthImgPath1, const std::string& rgbImgPath2, const std::string& depthImgPath2)
{



    cv::Mat depthImg = cv::imread(depthImgPath1, -1);
    cv::Mat colorImg = cv::imread(rgbImgPath1);

    cv::Mat colorImg1 = cv::imread(rgbImgPath2);
    cv::Mat depthImg1 = cv::imread(depthImgPath2, -1);

   

    
    corrPointsPairType corres2DPointsSet = alignImages(colorImg, colorImg1);


    
    cv::Mat cameraMatrix = cv::Mat(3, 3, CV_32FC1);

#if 1
    cameraMatrix.at<float>(0, 0) = 2742.4927;
    cameraMatrix.at<float>(1, 0) = 0.0;
    cameraMatrix.at<float>(2, 0) = 0.0;

    cameraMatrix.at<float>(0, 1) = 0.0;
    cameraMatrix.at<float>(1, 1) = 2742.4927;
    cameraMatrix.at<float>(2, 1) = 0.0;


    cameraMatrix.at<float>(0, 2) = 2015.7166;
    cameraMatrix.at<float>(1, 2) = 1509.2401;
    cameraMatrix.at<float>(2, 2) = 1.0;
   
#else

    cameraMatrix.at<float>(0, 0) = 2744.995/7.5;
    cameraMatrix.at<float>(1, 0) = 0.0;
    cameraMatrix.at<float>(2, 0) = 0.0;
     
    
    cameraMatrix.at<float>(0, 1) = 0.0;
    cameraMatrix.at<float>(1, 1) = 2744.995/7.5;
    cameraMatrix.at<float>(2, 1) = 0.0;


    cameraMatrix.at<float>(0, 2) = 320;
    cameraMatrix.at<float>(1, 2) = 240;
    cameraMatrix.at<float>(2, 2) = 1.0;

    

#endif


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr resCloud = createPointCloudFromRGBDepth(colorImg, depthImg, cameraMatrix);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr resCloud1 = createPointCloudFromRGBDepth(colorImg1, depthImg1, cameraMatrix);
  


    std::vector<cv::Point3f> point3dSet1;
    std::vector<cv::Point3f> point3dSet2;
    std::vector<cv::Point3f> point3dSet3;


    std::cout << "After cloud create \n";
   
    for (int i = 0; i < corres2DPointsSet.first.size(); ++i) {
        cv::Point3f tmpPoint1;
        tmpPoint1.x = resCloud->points.at(int(corres2DPointsSet.first.at(i).x) + int(corres2DPointsSet.first.at(i).y) * colorImg.cols).x;
        tmpPoint1.y = resCloud->points.at(int(corres2DPointsSet.first.at(i).x) + int(corres2DPointsSet.first.at(i).y) * colorImg.cols).y;
        tmpPoint1.z = resCloud->points.at(int(corres2DPointsSet.first.at(i).x) + int(corres2DPointsSet.first.at(i).y) * colorImg.cols).z;



        cv::Point3f tmpPoint2;
        tmpPoint2.x = resCloud1->points.at(int(corres2DPointsSet.second.at(i).x) + int(corres2DPointsSet.second.at(i).y) * colorImg.cols).x;
        tmpPoint2.y = resCloud1->points.at(int(corres2DPointsSet.second.at(i).x) + int(corres2DPointsSet.second.at(i).y) * colorImg.cols).y;
        tmpPoint2.z = resCloud1->points.at(int(corres2DPointsSet.second.at(i).x) + int(corres2DPointsSet.second.at(i).y) * colorImg.cols).z;

        if ( tmpPoint1.z < 1000 &&
            tmpPoint2.z < 1000   //Getting all points which has less then 1 meter distance from the cam
           ) {
            point3dSet1.push_back(tmpPoint1);
            point3dSet2.push_back(tmpPoint2);
         

        }
        

    }

    auto  bruteforceBestTransformBasedon4Points = [](std::vector<cv::Point3f> point3dSet1,
    std::vector<cv::Point3f> point3dSet2)->cv::Mat {
        const int N = 1000;
        const int nPoints = point3dSet2.size();
        
        for (int i = 0; i < N; ++i) {

            std::vector<int> indices;
           
            for (int j = 0; j < N; ++j) {
                int index = rand() % nPoints;
                if (std::find(indices.begin(), indices.end(), index) == indices.end()) {
                    indices.push_back(index);
                    if (indices.size() == 4) {
                        break;
                    }
                }
            }

                cv::Mat point3dSet1Mat = cv::Mat(1, indices.size(), CV_64FC3);
                cv::Mat point3dSet2Mat = cv::Mat(1, indices.size(), CV_64FC3);
                for (int l = 0; l < indices.size(); ++ l) {
                    int index = indices.at(l);
                    point3dSet1Mat.at<cv::Vec3d>(0, l)[0] = point3dSet1.at(index).x;
                    point3dSet1Mat.at<cv::Vec3d>(0, l)[1] = point3dSet1.at(index).y;
                    point3dSet1Mat.at<cv::Vec3d>(0, l)[2] = point3dSet1.at(index).z;

                    point3dSet2Mat.at<cv::Vec3d>(0, l)[0] = point3dSet2.at(index).x;
                    point3dSet2Mat.at<cv::Vec3d>(0, l)[1] = point3dSet2.at(index).y;
                    point3dSet2Mat.at<cv::Vec3d>(0, l)[2] = point3dSet2.at(index).z;
                }


                cv::Mat_<double> resMatRig = _FindRigidTransform(point3dSet2Mat, point3dSet1Mat);
                if (std::abs(resMatRig.at<double>(2, 3)) > 1 && std::abs(resMatRig.at<double>(2, 3)) < 30) {
                    std::cout << "\n\n\n\n\n Spo code mat " << i << resMatRig <<  std::endl << cv::Mat(indices) << std::endl;
                }
            
        }
        return cv::Mat();
    };

    bruteforceBestTransformBasedon4Points(point3dSet1, point3dSet2);
    
    /*cv::Mat point3dSet1Mat = cv::Mat(1, point3dSet1.size(), CV_64FC3);
    cv::Mat point3dSet2Mat = cv::Mat(1, point3dSet2.size(), CV_64FC3);
    for (int i = 0; i < point3dSet1.size(); ++i) {
        point3dSet1Mat.at<cv::Vec3d>(0, i)[0] = point3dSet1.at(i).x;
        point3dSet1Mat.at<cv::Vec3d>(0, i)[1] = point3dSet1.at(i).y;
        point3dSet1Mat.at<cv::Vec3d>(0, i)[2] = point3dSet1.at(i).z;

        point3dSet2Mat.at<cv::Vec3d>(0, i)[0] = point3dSet2.at(i).x;
        point3dSet2Mat.at<cv::Vec3d>(0, i)[1] = point3dSet2.at(i).y;
        point3dSet2Mat.at<cv::Vec3d>(0, i)[2] = point3dSet2.at(i).z;

    }
    

    cv::Mat_<double> resMatRig = FindRigidTransform(point3dSet2Mat, point3dSet1Mat);
    std::cout << "Spo code mat " << resMatRig << std::endl;
    */


    std::vector<unsigned int> weightsForPointSet1;
    std::vector<unsigned int> weightsForPointSet2;
    checkAndFilter3DMatchedPoints(point3dSet1, point3dSet2, weightsForPointSet1, weightsForPointSet2);
    std::cout << "POints set1 " << point3dSet1.size() << std::endl;
    std::cout << " Points set2 " << point3dSet2.size() << std::endl;
    std::cout << "Weidghts for point set 1   \n";
    for (int g = 0; g < weightsForPointSet1.size(); ++g) {
        std::cout<< weightsForPointSet1[g] << std::endl;
    }
    if (!(point3dSet1.size() > 4 && point3dSet2.size() > 4)) {
        return;
    }
    cv::Mat transformMatInit =    cv::estimateAffine3D(point3dSet2, point3dSet1);
    
   
    Eigen::Matrix4d transformMat = Eigen::Matrix4d::Identity();
    for (int i = 0; i < transformMatInit.rows; ++i) {
        for (int j = 0; j < transformMatInit.cols; ++j) {
            transformMat(i, j) = transformMatInit.at<double>(i, j);
        }
    }
    transformMat(3, 0) = 0;
    transformMat(3, 1) = 0;
    transformMat(3, 2) = 0;
    transformMat(3, 3) = 1;

    std::cout << "Transformation values " << transformMatInit.rows << "  " << transformMatInit.cols << std::endl;
    std::cout << transformMat << std::endl;

    if (m_pointCloudStitchedLast.height == 0 && m_pointCloudStitchedLast.width == 0) {
        m_transformationMatrixLast = transformMat;
       
    }
    else {
        Eigen::Matrix3d currRotMat;
        Eigen::Matrix3d lastRotMat;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                currRotMat(i, j) = transformMat(i, j);
                lastRotMat(i, j) = m_transformationMatrixLast(i, j);
            }
        }
        Eigen::Vector3d currTrVec;
        Eigen::Vector3d lastTrVec;
        for (int i = 0; i < 3; ++i) {
            currTrVec(i) = transformMat(i, 3);
            lastTrVec(i) = m_transformationMatrixLast(i, 3);
             
            //RMatrix = RMatrixTmp * RMatrix;
            //TMatrix = RMatrixTmp * TMatrix + TMatrixTmp;

        }

        lastRotMat = currRotMat * lastRotMat;
        lastTrVec = currRotMat * lastTrVec + currTrVec;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                m_transformationMatrixLast(i, j) = lastRotMat(i, j);
            }
        }

        m_transformationMatrixLast(0, 3) = lastTrVec(0);
        m_transformationMatrixLast(1, 3) = lastTrVec(1);
        m_transformationMatrixLast(2, 3) = lastTrVec(2);

        m_transformationMatrixLast(3, 0) = 0;
        m_transformationMatrixLast(3, 1) = 0;
        m_transformationMatrixLast(3, 2) = 0;
        m_transformationMatrixLast(3, 3) = 1;

    }

    pcl::transformPointCloud(*resCloud1, *resCloud1, m_transformationMatrixLast);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr resCloudFilt(new pcl::PointCloud<pcl::PointXYZRGB>);

    if (m_pointCloudStitchedLast.height == 0 && m_pointCloudStitchedLast.width == 0) {
        for (int j = 0; j < resCloud->points.size(); ++j) {
            if (resCloud->points[j].z < 1000) {
                resCloudFilt->push_back(resCloud->points[j]);
            }
        }
    }
    resCloudFilt->height = 1;
    resCloudFilt->width = resCloudFilt->points.size();
    *resCloud = *resCloudFilt;



    pcl::PointCloud<pcl::PointXYZRGB>::Ptr resCloud1Filt(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (int j = 0; j < resCloud1->points.size(); ++j) {
        if ( resCloud1->points[j].z < 1000) {
            resCloud1Filt->push_back(resCloud1->points[j]);
        }
    }
    resCloud1Filt->height = 1;
    resCloud1Filt->width = resCloud1Filt->points.size();
    *resCloud1 = *resCloud1Filt;

    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    icp.setMaximumIterations(1);

   

   

  
    if (m_pointCloudStitchedLast.height == 0 && m_pointCloudStitchedLast.width == 0) {

        icp.setInputSource(resCloud1);
        icp.setInputTarget(resCloud);
       // icp.align(*resCloud1);   //Open the comment for run icp 
        m_pointCloudStitchedLastOne = *resCloud1;

  
     
       
        *resCloud = *resCloud + *resCloud1;



        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud3F(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (int j = 0; j < resCloud->points.size(); ++j) {
            if (resCloud->points[j].z < /*500*/1000) {
                cloud3F->push_back(resCloud->points[j]);
            }
        }
        cloud3F->height = 1;
        cloud3F->width = cloud3F->points.size();
        *resCloud = *cloud3F;
    }

    if (m_pointCloudStitchedLast.height == 0 && m_pointCloudStitchedLast.width == 0) {
        m_pointCloudStitchedLast = *resCloud;
        pcl::io::savePCDFileASCII("onePointCloud.pcd", m_pointCloudStitchedLast);
        pcl::io::savePLYFileASCII("onePointCloud.ply", m_pointCloudStitchedLast);
    }
    else {

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud5F(new pcl::PointCloud<pcl::PointXYZRGB>);
        *cloud5F = m_pointCloudStitchedLastOne;
        icp.setInputSource(resCloud1);
        icp.setInputTarget(cloud5F);
        ///icp.align(*resCloud1);  //Open the comment for run icp
        m_pointCloudStitchedLastOne = *resCloud1;



        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud4F(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (int j = 0; j < resCloud1->points.size(); ++j) {
            if (resCloud1->points[j].z < /*500*/1000) {
                cloud4F->push_back(resCloud1->points[j]);
            }
        }
        cloud4F->height = 1;
        cloud4F->width = cloud4F->points.size();
        *resCloud1 = *cloud4F;
        m_pointCloudStitchedLast = m_pointCloudStitchedLast + *resCloud1;





    }


}

//Make additional filtering of mathched 3d points based on distance changing between two points from 1 to 2 point cloud
void pointCloudStitcher::checkAndFilter3DMatchedPoints(std::vector<cv::Point3f>& pointSet1, std::vector<cv::Point3f>& pointSet2, std::vector<unsigned int>& weightsForPoints1, std::vector<unsigned int>& weightsForPoints2)
{

    std::vector<cv::Point3f> pointsSet1Tmp;
    std::vector<cv::Point3f> pointsSet2Tmp;
    weightsForPoints1.resize(pointSet1.size());
    weightsForPoints2.resize(pointSet2.size());
    

    for (int i = 0; i < weightsForPoints1.size(); ++i) {
        weightsForPoints1[i] = 0;
    }
    for (int i = 0; i < weightsForPoints2.size(); ++i) {
        weightsForPoints2[i] = 0;
    }

    for (int i = 0; i < pointSet1.size(); ++i) {
        for (int j = 0; j < pointSet2.size(); ++j) {
            if (i != j) {
                if (fabsf(distBetweenPoints(pointSet1[i], pointSet1[j]) -
                    distBetweenPoints(pointSet2[i], pointSet2[j])) < 3) {
                    weightsForPoints1[i]++;
                    weightsForPoints1[j]++;
                    weightsForPoints2[i]++;
                    weightsForPoints2[j]++;
                }
            }
        }
    }
    
    unsigned int meanValueOfWeights1 = 0;
    unsigned int meanValueOfWeights2 = 0;
    unsigned int countOfNonZero1 = 0;
    unsigned int countOfNonZero2 = 0;
    for (int i = 0; i < weightsForPoints1.size(); ++i) {
        if (weightsForPoints1[i] != 0) {
            meanValueOfWeights1 += weightsForPoints1[i];
            ++countOfNonZero1;
        }
    }
    meanValueOfWeights1 = meanValueOfWeights1 / countOfNonZero1;

    for (int i = 0; i < weightsForPoints2.size(); ++i) {
        if (weightsForPoints2[i] != 0) {
            meanValueOfWeights2 += weightsForPoints2[i];
            ++countOfNonZero2;
        }
    }
    meanValueOfWeights2 = meanValueOfWeights2 / countOfNonZero2;

    for (int i = 0; i < pointSet1.size(); ++i) {
        if (weightsForPoints1[i] > meanValueOfWeights1 ) {
            pointsSet1Tmp.push_back(pointSet1[i]);
        }
    }
    for (int i = 0; i < pointSet2.size(); ++i) {
        if (weightsForPoints2[i] > meanValueOfWeights2 ) {
            pointsSet2Tmp.push_back(pointSet2[i]);
        }
    }
    pointSet1.clear();
    pointSet2.clear();
    pointSet1 = pointsSet1Tmp;
    pointSet2 = pointsSet2Tmp;


    std::cout << "Meaan weights " << meanValueOfWeights1 << std::endl;

}


//Write result point cloud file
void pointCloudStitcher::writeResultPointCloud()
{

    pcl::io::savePLYFileASCII("pointCloudForMeasurements.ply", m_pointCloudStitchedLastOne);
    std::cout << "Write point cloud " << m_pointCloudStitchedLast.height << "   " << m_pointCloudStitchedLast.width << std::endl;
    pcl::io::savePCDFileASCII("testCloudStitchedFinal.pcd", m_pointCloudStitchedLast);
    pcl::io::savePLYFileASCII("testCloudStitchedFinal.ply", m_pointCloudStitchedLast);
}
