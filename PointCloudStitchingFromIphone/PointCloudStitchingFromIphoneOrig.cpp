// PointCloudStitchingFromIphone.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "PointCloudStitchingFromIphone.h"

pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloudStitcher::createPointCloudFromRGBDepth(const cv::Mat& rgbImage, const cv::Mat& depthImage, const cv::Mat& calibMatrix)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new  pcl::PointCloud<pcl::PointXYZRGB>());


    cv::Mat points3d;


    for (int i = 0; i < depthImage.rows; ++i) {
        for (int j = 0; j < depthImage.cols; ++j) {
            pcl::PointXYZRGB tmpPoint;

            // const float depthConst = 0.01;

             //else {
            tmpPoint.b = rgbImage.at<cv::Vec3b>(i, j)[0];
            tmpPoint.g = rgbImage.at<cv::Vec3b>(i, j)[1];
            tmpPoint.r = rgbImage.at<cv::Vec3b>(i, j)[2];
            // fx = 2742.4927 
            //cx = 2015.7166
            //cy = 1509.2401,
            //fy = 2742.4927 


            tmpPoint.x = (((float)j - calibMatrix.at<float>(0,2)) * (float)((depthImage.at<unsigned short>(i, j)))) / calibMatrix.at<float>(0,0);
            tmpPoint.y = (((float)i - calibMatrix.at<float>(1, 2)) * (float)((depthImage.at<unsigned short>(i, j)))) / calibMatrix.at<float>(1, 1);
            tmpPoint.z = depthImage.at<unsigned short>(i, j)*0.5;

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
    std::cout << "result=" << result << std::endl;

    return finalRes;
}





float pointCloudStitcher::distBetweenPoints(const cv::Point3f& point1, const cv::Point3f& point2)
{
    return std::sqrtf((point1.x - point2.x) * (point1.x - point2.x) + (point1.y - point2.y) * (point1.y - point2.y) + (point1.z - point2.z) * (point1.z - point2.z));
}





corrPointsPairType pointCloudStitcher::alignImages(cv::Mat& im1, cv::Mat& im2 ,cv::Mat& h  )
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
    cv::imwrite("matches.jpg", imMatches);

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
    cv::imwrite("res1.png", img1D);
    cv::imwrite("res2.png", img2D);
    
    // Find homography
    h = findHomography(points1, points2, cv::RANSAC);
    std::cout << "GGGGGGGGGGGGG \n";
    std::cout << h << std::endl;
    return correspondPointsPair;
   
}

//void stitchTwoPointClouds(const std::string& inputColorImgPath1, const std::string& inputDepthImgPath1, const std::string& inputColorImgPath2, const std::string& inputDepthImgPath2, Eigen::Matrix3d& RMatrix, Eigen::Vector3d& TMatrix, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud1, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud2)
//{
//    cv::Mat depthImg = cv::imread(inputDepthImgPath1, -1);
//    cv::Mat colorImg = cv::imread(inputColorImgPath1);
//
//    cv::Mat colorImg1 = cv::imread(inputColorImgPath2);
//    cv::Mat depthImg1 = cv::imread(inputDepthImgPath2, -1);
//
//
//
//    cv::Mat transformMat2d;
//    corrPointsPairType corres2DPointsSet = alignImages(colorImg, colorImg1, transformMat2d);
//    // return 0;
//
//
//    cv::Mat cameraMatrix = cv::Mat(3, 3, CV_32FC1);
//
//
//    cameraMatrix.at<float>(0, 0) = 2742.4927;
//    cameraMatrix.at<float>(1, 0) = 0.0;
//    cameraMatrix.at<float>(2, 0) = 0.0;
//
//    cameraMatrix.at<float>(0, 1) = 0.0;
//    cameraMatrix.at<float>(1, 1) = 2742.4927;
//    cameraMatrix.at<float>(2, 1) = 0.0;
//
//
//    cameraMatrix.at<float>(0, 2) = 2015.7166;
//    cameraMatrix.at<float>(1, 2) = 1509.2401;
//    cameraMatrix.at<float>(2, 2) = 1.0;
//
//
//    cloud1 = createPointCloudFromRGBDepth(colorImg, depthImg, cameraMatrix);
//    cloud2 = createPointCloudFromRGBDepth(colorImg1, depthImg1, cameraMatrix);
//    std::vector<cv::Point3f> point3dSet1;
//    std::vector<cv::Point3f> point3dSet2;
//
//
//    for (int i = 0; i < corres2DPointsSet.first.size(); ++i) {
//        cv::Point3f tmpPoint1;
//        tmpPoint1.x = cloud1->points.at(corres2DPointsSet.first.at(i).x + corres2DPointsSet.first.at(i).y * colorImg.cols).x;
//        tmpPoint1.y = cloud1->points.at(corres2DPointsSet.first.at(i).x + corres2DPointsSet.first.at(i).y * colorImg.cols).y;
//        tmpPoint1.z = cloud1->points.at(corres2DPointsSet.first.at(i).x + corres2DPointsSet.first.at(i).y * colorImg.cols).z;
//
//
//
//
//        cv::Point3f tmpPoint2;
//        tmpPoint2.x = cloud2->points.at(corres2DPointsSet.second.at(i).x + corres2DPointsSet.second.at(i).y * colorImg.cols).x;
//        tmpPoint2.y = cloud2->points.at(corres2DPointsSet.second.at(i).x + corres2DPointsSet.second.at(i).y * colorImg.cols).y;
//        tmpPoint2.z = cloud2->points.at(corres2DPointsSet.second.at(i).x + corres2DPointsSet.second.at(i).y * colorImg.cols).z;
//
//        //if (distBetweenPoints(tmpPoint1, tmpPoint2) <= 10 &&  tmpPoint1.z < 700 && tmpPoint2.z < 700) {
//
//            if (std::fabsf(tmpPoint1.z - tmpPoint2.z) <= 10) {
//                point3dSet1.push_back(tmpPoint1);
//                point3dSet2.push_back(tmpPoint2);
//                std::cout << "Point1 " << tmpPoint1 << std::endl;
//                std::cout << "Point 2 " << tmpPoint2 << std::endl;
//
//            }
//       // }
//        
//
//    }
//
//    cv::Mat point3dSet1Mat = cv::Mat(1, point3dSet1.size(), CV_64FC3);
//    cv::Mat point3dSet2Mat = cv::Mat(1, point3dSet2.size(), CV_64FC3);
//    for (int i = 0; i < point3dSet1.size(); ++i) {
//        point3dSet1Mat.at<cv::Vec3d>(0, i)[0] = point3dSet1.at(i).x;
//        point3dSet1Mat.at<cv::Vec3d>(0, i)[1] = point3dSet1.at(i).y;
//        point3dSet1Mat.at<cv::Vec3d>(0, i)[2] = point3dSet1.at(i).z;
//
//        point3dSet2Mat.at<cv::Vec3d>(0, i)[0] = point3dSet2.at(i).x;
//        point3dSet2Mat.at<cv::Vec3d>(0, i)[1] = point3dSet2.at(i).y;
//        point3dSet2Mat.at<cv::Vec3d>(0, i)[2] = point3dSet2.at(i).z;
//
//    }
//
//
//
//    cv::Mat transformMatInit = cv::estimateAffine3D(point3dSet2, point3dSet1);
//    
//
//    Eigen::Matrix4d transformMat = Eigen::Matrix4d::Identity();
//    for (int i = 0; i < transformMatInit.rows; ++i) {
//        for (int j = 0; j < transformMatInit.cols; ++j) {
//            if (j < 3 && j < 3) {
//                RMatrix(i, j) = transformMatInit.at<double>(i, j);
//            }
//            else if (j == 3 && i < 3) {
//                TMatrix(i, 0) = transformMatInit.at<double>(i, j);
//            }
//        }
//    }
// 
//}
//
//int mainExper()
//{
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZRGB>);
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZRGB>);
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudRes(new pcl::PointCloud<pcl::PointXYZRGB>);
//
//    Eigen::Matrix3d RMatrix;
//    Eigen::Vector3d TMatrix;
//    stitchTwoPointClouds("E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/3/Documents/color_185.png", "E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/3/Documents/depth_185.png", "E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/3/Documents/color_195.png", "E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/2/Documents/depth_195.png", RMatrix, TMatrix, cloud1, cloud2);
//
//    std::cout << "Rotation vecgtor " << std::endl;
//    std::cout << RMatrix << std::endl;
//    std::cout << "Translation vector " << std::endl;
//    std::cout << TMatrix << std::endl;
//
//    Eigen::Matrix4d transformMat = Eigen::Matrix4d::Identity();
//    for (int i = 0; i < 3; ++i) {
//        for (int j = 0; j < 3; ++j) {
//            transformMat(i, j) = RMatrix(i, j);
//        }
//    }
//    transformMat(0, 3) = TMatrix(0, 0);
//    transformMat(1, 3) = TMatrix(1, 0);
//    transformMat(2, 3) = TMatrix(2, 0);
//    transformMat(3, 3) = 1;
//
//    std::cout << "Final transformation matrix !!!!! " << std::endl;
//    std::cout << transformMat << std::endl;
//
//    pcl::transformPointCloud(*cloud2, *cloud2, transformMat);
//    *cloud1 = *cloud1 + *cloud2;
//    *cloudRes = *cloud1;
//
//
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud3F(new pcl::PointCloud<pcl::PointXYZRGB>);
//
//    /*for (int j = 0; j < cloudRes->points.size(); ++j) {
//        if (cloudRes->points[j].z < 700) {
//            cloud3F->push_back(cloudRes->points[j]);
//        }
//    }
//    cloud3F->height = 1;
//    cloud3F->width = cloud3F->points.size();
//    *cloudRes = *cloud3F;
//    */
//
//    pcl::io::savePLYFileASCII("resCloud.ply", *cloudRes);
//    pcl::io::savePCDFileASCII("resCloud.pcd", *cloudRes);
//    return 0;
//
//    Eigen::Matrix3d RMatrixTmp;
//    Eigen::Vector3d TMatrixTmp;
//    stitchTwoPointClouds("E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/2/Documents/color_71.png", "E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/2/Documents/depth_71.png", "E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/2/Documents/color_72.png", "E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/2/Documents/depth_72.png", RMatrixTmp, TMatrixTmp, cloud1, cloud2);
//    RMatrix = RMatrixTmp * RMatrix;
//    TMatrix = RMatrixTmp * TMatrix + TMatrixTmp;
//
//   // Eigen::Matrix4d transformMat = Eigen::Matrix4d::Identity();
//    for (int i = 0; i < 3; ++i) {
//        for (int j = 0; j < 3; ++j) {
//            transformMat(i, j) = RMatrix(i, j);
//        }
//    }
//    transformMat(0, 3) = TMatrix(0, 0);
//    transformMat(1, 3) = TMatrix(1, 0);
//    transformMat(2, 3) = TMatrix(2, 0);
//    transformMat(3, 3) = 1;
//
//    std::cout << "Final transformation matrix !!!!! " << std::endl;
//    std::cout << transformMat << std::endl;
//
//    pcl::transformPointCloud(*cloud2, *cloud2, transformMat);
//
//   // *cloudRes = *cloudRes + *cloud2;
//
//    //pcl::io::savePLYFileASCII("testCloud.ply", *cloud1);
//    pcl::io::savePLYFileASCII("resCloud.ply", *cloudRes);
//
//}


//void simpleStitchWithICP(const std::string& inputColorImgPath1, const std::string& inputDepthImgPath1, const std::string& inputColorImgPath2, const std::string& inputDepthImgPath2)
//
//{
//
//    cv::Mat depthImg = cv::imread("E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/3/Documents/depth_100.png", -1);
//    cv::Mat colorImg = cv::imread("E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/3/Documents/color_100.png");
//
//
//   
//
//
//    cv::Mat depthImg1 = cv::imread("E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/3/Documents/depth_105.png", -1);
//    cv::Mat colorImg1 = cv::imread("E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/3/Documents/color_105.png");
//
//
//    cv::Mat cameraMatrix = cv::Mat(3, 3, CV_32FC1);
//
//
//    cameraMatrix.at<float>(0, 0) = 2742.4927;
//    cameraMatrix.at<float>(1, 0) = 0.0;
//    cameraMatrix.at<float>(2, 0) = 0.0;
//
//    cameraMatrix.at<float>(0, 1) = 0.0;
//    cameraMatrix.at<float>(1, 1) = 2742.4927;
//    cameraMatrix.at<float>(2, 1) = 0.0;
//
//
//    cameraMatrix.at<float>(0, 2) = 2015.7166;
//    cameraMatrix.at<float>(1, 2) = 1509.2401;
//    cameraMatrix.at<float>(2, 2) = 1.0;
//
//
//
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1 = createPointCloudFromRGBDepth(colorImg, depthImg, cameraMatrix);
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr  cloud2 = createPointCloudFromRGBDepth(colorImg1, depthImg1, cameraMatrix);
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1F(new pcl::PointCloud<pcl::PointXYZRGB>);
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2F(new pcl::PointCloud<pcl::PointXYZRGB>);
//
//    for (int i = 0; i < cloud1->points.size(); ++i) {
//        if (cloud1->points[i].z < 500) {
//            cloud1F->push_back(cloud1->points[i]);
//        }
//    }
//    cloud1F->height = 1;
//    cloud1F->width = cloud1F->points.size();
//
//    for (int i = 0; i < cloud2->points.size(); ++i) {
//        if (cloud2->points[i].z < 500) {
//            cloud2F->push_back(cloud2->points[i]);
//        }
//    }
//    cloud2F->height = 1;
//    cloud2F->width = cloud2F->points.size();
//    *cloud1 = *cloud1F;
//    *cloud2 = *cloud2F;
//
//    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
//    icp.setMaximumIterations(1);
//    icp.setInputSource(cloud2);
//    icp.setInputTarget(cloud1);
//    icp.align(*cloud2);
//    icp.setMaximumIterations(1);  // We set this variable to 1 for the next time we will call .align () function
//   // std::cout << "Applied " << iterations << " ICP iteration(s) in " << time.toc() << " ms" << std::endl;
//    //*cloud1 = *cloud1 + *cloud2;
//    if (icp.hasConverged())
//    {
//        std::cout << "\nICP has converged, score is " << icp.getFitnessScore() << std::endl;
//        std::cout << "\nICP transformation " << 3 << " : cloud_icp -> cloud_in" << std::endl;
//        Eigen::Matrix4d transformation_matrix = icp.getFinalTransformation().cast<double>();
//        //pcl::transformPointCloud(*cloud2, *cloud2, transformation_matrix);
//        *cloud1 = *cloud1 + *cloud2;
//        
//        std::cout<<"Transform Mat\n"<<transformation_matrix<<"\n";
//        std::cout << "POints clout " << cloud1->points.size() << std::endl;
//    }
//    else
//    {
//        PCL_ERROR("\nICP has not converged.\n");
//        return ;
//    }
//    
//
//    for (int i =110; i < 125; i += 5) {
//        std::string currDepthPath = "E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/3/Documents/depth_" + std::to_string(long(i)) + ".png";
//        std::string currColorPath = "E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/3/Documents/color_" + std::to_string(long(i)) + ".png";
//
//
//        cv::Mat depthImg2 = cv::imread(currDepthPath, -1);
//        cv::Mat colorImg2 = cv::imread(currColorPath);
//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr  cloud3 = createPointCloudFromRGBDepth(colorImg2, depthImg2, cameraMatrix);
//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud3F (new pcl::PointCloud<pcl::PointXYZRGB>);
//
//        for (int j = 0; j < cloud3->points.size(); ++j) {
//            if (cloud3->points[j].z < 500) {
//                cloud3F->push_back(cloud3->points[j]);
//            }
//        }
//        cloud3F->height = 1;
//        cloud3F->width = cloud3F->points.size();
//        *cloud3 = *cloud3F;
//        
//        
//        
//        
//        
//        icp.setMaximumIterations(1);
//        icp.setInputSource(cloud3);
//        icp.setInputTarget(cloud2);
//        std::cout << "Points cout before aliogn " << cloud1->points.size() << std::endl;
//        icp.align(*cloud3);
//        std::cout << "POiots count after align " << cloud1->points.size() << std::endl;
//        // 
//          //icp.setMaximumIterations(15);  // We set this variable to 1 for the next time we will call .align () function
//        // std::cout << "Applied " << iterations << " ICP iteration(s) in " << time.toc() << " ms" << std::endl;
//
//        if (icp.hasConverged())
//        {
//            std::cout << "\nICP has converged, score is " << icp.getFitnessScore() << std::endl;
//            std::cout << "\nICP transformation " << 3 << " : cloud_icp -> cloud_in" << std::endl;
//            Eigen::Matrix4d transformation_matrix = icp.getFinalTransformation().cast<double>();
//            // pcl::transformPointCloud(*cloud3, *cloud3, transformation_matrix);
//            std::cout << "POints clout before " << cloud1->points.size() << std::endl;
//            *cloud1 = *cloud1 + *cloud3;
//            std::cout << "CLoud 3 poitns count " << cloud3->points.size() << std::endl;
//            std::cout << "Transform Mat\n" << transformation_matrix << "\n";
//            std::cout << "POints clout " << cloud1->points.size() << std::endl;
//        }
//        else
//        {
//            PCL_ERROR("\nICP has not converged.\n");
//            return;
//        }
//        *cloud2 = *cloud3;
//
//
//    }
//
//    /*
//    cv::Mat depthImg3 = cv::imread("E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/2/Documents/depth_55.png", -1);
//    cv::Mat colorImg3 = cv::imread("E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/2/Documents/color_55.png");
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr  cloud4 = createPointCloudFromRGBDepth(colorImg3, depthImg3, cameraMatrix);
//    icp.setMaximumIterations(10);
//    icp.setInputSource(cloud4);
//    icp.setInputTarget(cloud3);
//    std::cout << "Points cout before aliogn " << cloud1->points.size() << std::endl;
//    icp.align(*cloud4);
//    std::cout << "POiots count after align " << cloud1->points.size() << std::endl;
//    // 
//      //icp.setMaximumIterations(15);  // We set this variable to 1 for the next time we will call .align () function
//    // std::cout << "Applied " << iterations << " ICP iteration(s) in " << time.toc() << " ms" << std::endl;
//
//    if (icp.hasConverged())
//    {
//        std::cout << "\nICP has converged, score is " << icp.getFitnessScore() << std::endl;
//        std::cout << "\nICP transformation " << 3 << " : cloud_icp -> cloud_in" << std::endl;
//        Eigen::Matrix4d transformation_matrix = icp.getFinalTransformation().cast<double>();
//        // pcl::transformPointCloud(*cloud3, *cloud3, transformation_matrix);
//        std::cout << "POints clout before " << cloud1->points.size() << std::endl;
//        *cloud1 = *cloud1 + *cloud4;
//        std::cout << "CLoud 3 poitns count " << cloud3->points.size() << std::endl;
//        std::cout << "Transform Mat\n" << transformation_matrix << "\n";
//        std::cout << "POints clout " << cloud1->points.size() << std::endl;
//    }
//    else
//    {
//        PCL_ERROR("\nICP has not converged.\n");
//        return;
//    }
//
//
//
//
//
//
//
//
//
//    cv::Mat depthImg4 = cv::imread("E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/2/Documents/depth_60.png", -1);
//    cv::Mat colorImg4 = cv::imread("E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/2/Documents/color_60.png");
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr  cloud5 = createPointCloudFromRGBDepth(colorImg4, depthImg4, cameraMatrix);
//    icp.setMaximumIterations(10);
//    icp.setInputSource(cloud5);
//    icp.setInputTarget(cloud4);
//    std::cout << "Points cout before aliogn " << cloud1->points.size() << std::endl;
//    icp.align(*cloud5);
//    std::cout << "POiots count after align " << cloud1->points.size() << std::endl;
//    // 
//      //icp.setMaximumIterations(15);  // We set this variable to 1 for the next time we will call .align () function
//    // std::cout << "Applied " << iterations << " ICP iteration(s) in " << time.toc() << " ms" << std::endl;
//
//    if (icp.hasConverged())
//    {
//        std::cout << "\nICP has converged, score is " << icp.getFitnessScore() << std::endl;
//        std::cout << "\nICP transformation " << 3 << " : cloud_icp -> cloud_in" << std::endl;
//        Eigen::Matrix4d transformation_matrix = icp.getFinalTransformation().cast<double>();
//        // pcl::transformPointCloud(*cloud3, *cloud3, transformation_matrix);
//        std::cout << "POints clout before " << cloud1->points.size() << std::endl;
//        *cloud1 = *cloud1 + *cloud5;
//        std::cout << "CLoud 3 poitns count " << cloud3->points.size() << std::endl;
//        std::cout << "Transform Mat\n" << transformation_matrix << "\n";
//        std::cout << "POints clout " << cloud1->points.size() << std::endl;
//    }
//    else
//    {
//        PCL_ERROR("\nICP has not converged.\n");
//        return;
//    }
//
//
//
//    */
//
//    pcl::io::savePLYFileASCII("icpRes.ply", *cloud1);
//    pcl::io::savePCDFileASCII("icpResPCD.pcd", *cloud1);
//
//}







//int mainLastWorking()
//{
//
//
//  //  simpleStitchWithICP("E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/2/Documents/color_60.png", "E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/2/Documents/depth_60.png", "E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/2/Documents/color_65.png", "E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/2/Documents/depth_65.png");
//   // return 0;
//    // cv::Mat depthImg = cv::imread("C:/Users/artas/Downloads/new16UC1/Documents/depth_120.png", -1);
//    //cv::Mat colorImg = cv::imread("C:/Users/artas/Downloads/new16UC1/Documents/color_120.png");
//
//    //cv::Mat colorImg1 = cv::imread("C:/Users/artas/Downloads/new16UC1/Documents/color_70.png");
//   // cv::Mat depthImg1 = cv::imread("C:/Users/artas/Downloads/new16UC1/Documents/depth_70.png", -1);
//    cv::Mat depthImg = cv::imread("E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/3/Documents/depth_515.png", -1);
//    cv::Mat colorImg = cv::imread("E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/3/Documents/color_515.png");
//
//    cv::Mat colorImg1 = cv::imread("E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/3/Documents/color_520.png");
//    cv::Mat depthImg1 = cv::imread("E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/3/Documents/depth_520.png", -1);
//    
//    cv::Mat depthImg2 = cv::imread("E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/3/Documents/depth_205.png", -1);
//    cv::Mat colorImg2 = cv::imread("E:/UpworkProjects/IOSBodyScanner/TrueDepthProjectWithArtash-20230710T131709Z-001/TrueDepthProjectWithArtash/3/Documents/color_205.png");
//    
//    cv::Mat transformMat2d;
//    corrPointsPairType corres2DPointsSet = alignImages(colorImg, colorImg1, transformMat2d);
//
//
//   // cv::Mat transformMat2d1;
//   // corrPointsPairType corres2DPointsSet1 = alignImages(colorImg1, colorImg2, transformMat2d1);
//
//   // return 0;
//
//
//    cv::Mat cameraMatrix = cv::Mat(3, 3, CV_32FC1);
//   
//
//    cameraMatrix.at<float>(0, 0) = 2742.4927;
//    cameraMatrix.at<float>(1, 0) = 0.0;
//    cameraMatrix.at<float>(2, 0) = 0.0;
//
//    cameraMatrix.at<float>(0, 1) = 0.0;
//    cameraMatrix.at<float>(1, 1) = 2742.4927;
//    cameraMatrix.at<float>(2, 1) = 0.0;
//
//
//    cameraMatrix.at<float>(0, 2) = 2015.7166;
//    cameraMatrix.at<float>(1, 2) = 1509.2401;
//    cameraMatrix.at<float>(2, 2) = 1.0;
//
//
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr resCloud = createPointCloudFromRGBDepth(colorImg, depthImg, cameraMatrix);
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr resCloud1 = createPointCloudFromRGBDepth(colorImg1, depthImg1, cameraMatrix);
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr resCloud2 = createPointCloudFromRGBDepth(colorImg2, depthImg2, cameraMatrix);
//    
//    
//    std::vector<cv::Point3f> point3dSet1;
//    std::vector<cv::Point3f> point3dSet2;
//    std::vector<cv::Point3f> point3dSet3;
//
//
//    
//    for (int i = 0; i < corres2DPointsSet.first.size(); ++i) {
//        cv::Point3f tmpPoint1;
//        tmpPoint1.x = resCloud->points.at(corres2DPointsSet.first.at(i).x + corres2DPointsSet.first.at(i).y * colorImg.cols).x;
//        tmpPoint1.y = resCloud->points.at(corres2DPointsSet.first.at(i).x + corres2DPointsSet.first.at(i).y * colorImg.cols).y;
//        tmpPoint1.z = resCloud->points.at(corres2DPointsSet.first.at(i).x + corres2DPointsSet.first.at(i).y * colorImg.cols).z;
//        
//       
//       
//
//        cv::Point3f tmpPoint2;
//        tmpPoint2.x = resCloud1->points.at(corres2DPointsSet.second.at(i).x + corres2DPointsSet.second.at(i).y * colorImg.cols).x;
//        tmpPoint2.y = resCloud1->points.at(corres2DPointsSet.second.at(i).x + corres2DPointsSet.second.at(i).y * colorImg.cols).y;
//        tmpPoint2.z = resCloud1->points.at(corres2DPointsSet.second.at(i).x + corres2DPointsSet.second.at(i).y * colorImg.cols).z;
//       
//        if (distBetweenPoints(tmpPoint1, tmpPoint2) < 15   /* && std::abs(corres2DPointsSet.first.at(i).x - corres2DPointsSet.second.at(i).x) < 15 &&
//            std::abs(corres2DPointsSet.first.at(i).y - corres2DPointsSet.second.at(i).y) < 15 
//            &&*/
//             && tmpPoint1.z < 500 &&
//            tmpPoint2.z < 500 
//           /* tmpPoint1.z < 1000 && tmpPoint2.z  < 1000*/) {
//            point3dSet1.push_back(tmpPoint1);
//            point3dSet2.push_back(tmpPoint2);
//            std::cout << "Point1 " << tmpPoint1 << std::endl;
//            std::cout << "Point 2 " << tmpPoint2 << std::endl;
//
//        }
//      ;
//
//    }
//    
//    cv::Mat point3dSet1Mat = cv::Mat( 1, point3dSet1.size(), CV_64FC3);
//    cv::Mat point3dSet2Mat = cv::Mat( 1, point3dSet2.size(), CV_64FC3);
//    for (int i = 0; i < point3dSet1.size(); ++i) {
//        point3dSet1Mat.at<cv::Vec3d>( 0,i)[0] = point3dSet1.at(i).x;
//        point3dSet1Mat.at<cv::Vec3d>( 0,i)[1] = point3dSet1.at(i).y;
//        point3dSet1Mat.at<cv::Vec3d>(0,i)[2] = point3dSet1.at(i).z;
//
//        point3dSet2Mat.at<cv::Vec3d>( 0,i)[0] = point3dSet2.at(i).x;
//        point3dSet2Mat.at<cv::Vec3d>( 0,i)[1] = point3dSet2.at(i).y;
//        point3dSet2Mat.at<cv::Vec3d>(0,i)[2] = point3dSet2.at(i).z;
//
//    }
//
//   // cv::Mat_<double> resMatRig = FindRigidTransform(point3dSet1Mat, point3dSet2Mat);
//   // std::cout << "Res mat from SPO CODE \n";
//   // std::cout << resMatRig << std::endl;
//
//    cv::Mat transformMatInit = cv::estimateAffine3D(point3dSet2, point3dSet1);
//   // cv::Mat transformMatInit2d = cv::estimateAffine2D(corres2DPointsSet.first, corres2DPointsSet.second );
//   // std::cout << "Estimate affind 2d " << std::endl;
//   // std::cout << transformMat2d << std::endl;
//    Eigen::Matrix4d transformMat = Eigen::Matrix4d::Identity();
//    for (int i = 0; i < transformMatInit.rows; ++i) {
//        for (int j = 0; j < transformMatInit.cols; ++j) {
//            transformMat(i, j) = transformMatInit.at<double>(i, j);
//        }
//    }
//    transformMat(3, 0) = 0;
//    transformMat(3, 1) = 0;
//    transformMat(3, 2) = 0;
//    transformMat(3, 1) = 1;
//
//    std::cout << "Transformation values " << transformMatInit.rows << "  " << transformMatInit.cols << std::endl;
//    std::cout << transformMat << std::endl;
//
//    //std::cout << "Transform mat 2d \n";
//    //std::cout << transformMat2d << std::endl;
//    //pcl::transformPointCloud()
//   /* pcl::io::savePLYFileASCII("testCloud.ply", *resCloud);
//    pcl::io::savePLYFileASCII("testCloud1.ply", *resCloud1);*/
//    pcl::transformPointCloud(*resCloud1, *resCloud1, transformMat);
//
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr resCloudFilt(new pcl::PointCloud<pcl::PointXYZRGB>);
//
//    for (int j = 0; j < resCloud->points.size(); ++j) {
//        if (resCloud->points[j].z < 1000) {
//            resCloudFilt->push_back(resCloud->points[j]);
//        }
//    }
//    resCloudFilt->height = 1;
//    resCloudFilt->width = resCloudFilt->points.size();
//    *resCloud = *resCloudFilt;
//
//
//
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr resCloud1Filt(new pcl::PointCloud<pcl::PointXYZRGB>);
//
//    for (int j = 0; j < resCloud1->points.size(); ++j) {
//        if (resCloud1->points[j].z < 1000) {
//            resCloud1Filt->push_back(resCloud1->points[j]);
//        }
//    }
//    resCloud1Filt->height = 1;
//    resCloud1Filt->width = resCloud1Filt->points.size();
//    *resCloud1 = *resCloud1Filt;
//
//
//    
//    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
//    icp.setMaximumIterations(7);
//    icp.setInputSource(resCloud1);
//    icp.setInputTarget(resCloud);
//   // icp.align(*resCloud1);
//    //icp.setmaximumiterations(1);  // we set this variable to 1 for the next time we will call .align () function
//
//    if (icp.hasConverged())
//    {
//        std::cout << "\nICP has converged, score is " << icp.getFitnessScore() << std::endl;
//        std::cout << "\nICP transformation " << 3 << " : cloud_icp -> cloud_in" << std::endl;
//        Eigen::Matrix4d transformation_matrix = icp.getFinalTransformation().cast<double>();
//        //pcl::transformPointCloud(*cloud2, *cloud2, transformation_matrix);
//        *resCloud = *resCloud + *resCloud1;
//
//        std::cout << "Transform Mat\n" << transformation_matrix << "\n";
//      
//    }
//
//    
//
//
//
//   *resCloud = *resCloud + *resCloud1;
//
//
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud3F(new pcl::PointCloud<pcl::PointXYZRGB>);
//
//    for (int j = 0; j < resCloud->points.size(); ++j) {
//        if (resCloud->points[j].z < 500) {
//            cloud3F->push_back(resCloud->points[j]);
//        }
//    }
//    cloud3F->height = 1;
//    cloud3F->width = cloud3F->points.size();
//    *resCloud = *cloud3F;
//
//
//
//
//
//    pcl::io::savePLYFileASCII("testCloudStitched.ply", *resCloud);
//    pcl::io::savePCDFileASCII("testCloudStitched.pcd", *resCloud);
//    std::cout << "Hello World!\n";
//    
//}








// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file








pointCloudStitcher::pointCloudStitcher()
{
    m_pointCloudStitchedLast.height = 0;
    m_pointCloudStitchedLast.width = 0;
}
pointCloudStitcher::~pointCloudStitcher()
{

}

void  pointCloudStitcher::stitchTwoPointClouds(const std::string& rgbImgPath1, const std::string& depthImgPath1, const std::string& rgbImgPath2, const std::string& depthImgPath2)
{



    cv::Mat depthImg = cv::imread(depthImgPath1, -1);
    cv::Mat colorImg = cv::imread(rgbImgPath1);

    cv::Mat colorImg1 = cv::imread(rgbImgPath2);
    cv::Mat depthImg1 = cv::imread(depthImgPath2, -1);

   

    cv::Mat transformMat2d;
    corrPointsPairType corres2DPointsSet = alignImages(colorImg, colorImg1, transformMat2d);


    
    cv::Mat cameraMatrix = cv::Mat(3, 3, CV_32FC1);


    cameraMatrix.at<float>(0, 0) = 2742.4927;
    cameraMatrix.at<float>(1, 0) = 0.0;
    cameraMatrix.at<float>(2, 0) = 0.0;

    cameraMatrix.at<float>(0, 1) = 0.0;
    cameraMatrix.at<float>(1, 1) = 2742.4927;
    cameraMatrix.at<float>(2, 1) = 0.0;


    cameraMatrix.at<float>(0, 2) = 2015.7166;
    cameraMatrix.at<float>(1, 2) = 1509.2401;
    cameraMatrix.at<float>(2, 2) = 1.0;


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr resCloud = createPointCloudFromRGBDepth(colorImg, depthImg, cameraMatrix);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr resCloud1 = createPointCloudFromRGBDepth(colorImg1, depthImg1, cameraMatrix);
  


    std::vector<cv::Point3f> point3dSet1;
    std::vector<cv::Point3f> point3dSet2;
    std::vector<cv::Point3f> point3dSet3;



    for (int i = 0; i < corres2DPointsSet.first.size(); ++i) {
        cv::Point3f tmpPoint1;
        tmpPoint1.x = resCloud->points.at(corres2DPointsSet.first.at(i).x + corres2DPointsSet.first.at(i).y * colorImg.cols).x;
        tmpPoint1.y = resCloud->points.at(corres2DPointsSet.first.at(i).x + corres2DPointsSet.first.at(i).y * colorImg.cols).y;
        tmpPoint1.z = resCloud->points.at(corres2DPointsSet.first.at(i).x + corres2DPointsSet.first.at(i).y * colorImg.cols).z;



        cv::Point3f tmpPoint2;
        tmpPoint2.x = resCloud1->points.at(corres2DPointsSet.second.at(i).x + corres2DPointsSet.second.at(i).y * colorImg.cols).x;
        tmpPoint2.y = resCloud1->points.at(corres2DPointsSet.second.at(i).x + corres2DPointsSet.second.at(i).y * colorImg.cols).y;
        tmpPoint2.z = resCloud1->points.at(corres2DPointsSet.second.at(i).x + corres2DPointsSet.second.at(i).y * colorImg.cols).z;

        if (distBetweenPoints(tmpPoint1, tmpPoint2) < 15   /* && std::abs(corres2DPointsSet.first.at(i).x - corres2DPointsSet.second.at(i).x) < 15 &&
            std::abs(corres2DPointsSet.first.at(i).y - corres2DPointsSet.second.at(i).y) < 15
            &&*/
            && tmpPoint1.z < 500 &&
            tmpPoint2.z < 500
            /* tmpPoint1.z < 1000 && tmpPoint2.z  < 1000*/) {
            point3dSet1.push_back(tmpPoint1);
            point3dSet2.push_back(tmpPoint2);
            std::cout << "Point1 " << tmpPoint1 << std::endl;
            std::cout << "Point 2 " << tmpPoint2 << std::endl;

        }
        ;

    }

    cv::Mat point3dSet1Mat = cv::Mat(1, point3dSet1.size(), CV_64FC3);
    cv::Mat point3dSet2Mat = cv::Mat(1, point3dSet2.size(), CV_64FC3);
    for (int i = 0; i < point3dSet1.size(); ++i) {
        point3dSet1Mat.at<cv::Vec3d>(0, i)[0] = point3dSet1.at(i).x;
        point3dSet1Mat.at<cv::Vec3d>(0, i)[1] = point3dSet1.at(i).y;
        point3dSet1Mat.at<cv::Vec3d>(0, i)[2] = point3dSet1.at(i).z;

        point3dSet2Mat.at<cv::Vec3d>(0, i)[0] = point3dSet2.at(i).x;
        point3dSet2Mat.at<cv::Vec3d>(0, i)[1] = point3dSet2.at(i).y;
        point3dSet2Mat.at<cv::Vec3d>(0, i)[2] = point3dSet2.at(i).z;

    }

 

    cv::Mat transformMatInit = cv::estimateAffine3D(point3dSet2, point3dSet1);
 
    Eigen::Matrix4d transformMat = Eigen::Matrix4d::Identity();
    for (int i = 0; i < transformMatInit.rows; ++i) {
        for (int j = 0; j < transformMatInit.cols; ++j) {
            transformMat(i, j) = transformMatInit.at<double>(i, j);
        }
    }
    transformMat(3, 0) = 0;
    transformMat(3, 1) = 0;
    transformMat(3, 2) = 0;
    transformMat(3, 1) = 1;

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
        m_transformationMatrixLast(3, 1) = 1;

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
        if (resCloud1->points[j].z < 1000) {
            resCloud1Filt->push_back(resCloud1->points[j]);
        }
    }
    resCloud1Filt->height = 1;
    resCloud1Filt->width = resCloud1Filt->points.size();
    *resCloud1 = *resCloud1Filt;

    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    icp.setMaximumIterations(2);

   

   

  
    if (m_pointCloudStitchedLast.height == 0 && m_pointCloudStitchedLast.width == 0) {

        icp.setInputSource(resCloud1);
        icp.setInputTarget(resCloud);
       // icp.align(*resCloud1);













        *resCloud = *resCloud + *resCloud1;



        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud3F(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (int j = 0; j < resCloud->points.size(); ++j) {
            if (resCloud->points[j].z < 500) {
                cloud3F->push_back(resCloud->points[j]);
            }
        }
        cloud3F->height = 1;
        cloud3F->width = cloud3F->points.size();
        *resCloud = *cloud3F;
    }

    if (m_pointCloudStitchedLast.height == 0 && m_pointCloudStitchedLast.width == 0) {
        m_pointCloudStitchedLast = *resCloud;
    }
    else {

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud5F(new pcl::PointCloud<pcl::PointXYZRGB>);
        *cloud5F = m_pointCloudStitchedLast;
        icp.setInputSource(resCloud1);
        icp.setInputTarget(cloud5F);
       // icp.align(*resCloud1);



        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud4F(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (int j = 0; j < resCloud1->points.size(); ++j) {
            if (resCloud1->points[j].z < 500) {
                cloud4F->push_back(resCloud1->points[j]);
            }
        }
        cloud4F->height = 1;
        cloud4F->width = cloud4F->points.size();
        *resCloud1 = *cloud4F;
        m_pointCloudStitchedLast = m_pointCloudStitchedLast + *resCloud1;
    }



    pcl::io::savePCDFileASCII("testCloudStitchedOne.pcd", *resCloud1);
    pcl::io::savePCDFileASCII("testCloudStitched.pcd", m_pointCloudStitchedLast);
    std::cout << "Hello World!\n";




}
