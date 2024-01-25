
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/transforms.h>
#include <pcl/registration/icp.h>

#include <pcl/filters/statistical_outlier_removal.h>



#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/extract_indices.h>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

using namespace cv::xfeatures2d;

const int MAX_FEATURES = /*2500*/800;
const float GOOD_MATCH_PERCENT = 0.25f;



typedef  std::pair< std::vector<cv::Point2f>, std::vector<cv::Point2f>> corrPointsPairType;

#pragma once


class pointCloudStitcher {
public:

	pointCloudStitcher();
	~pointCloudStitcher();
	corrPointsPairType alignImages(cv::Mat& im1, cv::Mat& im2);
	corrPointsPairType alignImagesML(cv::Mat& im1, cv::Mat& im2);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr createPointCloudFromRGBDepth(const cv::Mat& rgbImage, const cv::Mat& depthImage, const cv::Mat& calibMatrix);
	void stitchTwoPointClouds(const std::string& rgbImgPath1, const std::string& depthImgPath1, const std::string& rgbImgPath2, const std::string& depthImgPath2);
	void writeResultPointCloud();
	pcl::PointCloud<pcl::PointXYZRGB> getLastPointCloud();

private:

	float distBetweenPoints(const cv::Point3f& point1, const cv::Point3f& point2);
	
	void checkAndFilter3DMatchedPoints(std::vector<cv::Point3f>& pointSet1, std::vector<cv::Point3f>& pointSet2, std::vector<unsigned int>& weightsForPoints1, std::vector<unsigned int>& weightsForPoints2);
	
	cv::Mat_<double> FindRigidTransform(const cv::Mat_<cv::Vec3d>& points1, const cv::Mat_<cv::Vec3d> points2);
	cv::Vec3d CalculateMean(const cv::Mat_<cv::Vec3d>& points);

private:
	pcl::PointCloud<pcl::PointXYZRGB> m_pointCloudStitchedLast;
	pcl::PointCloud<pcl::PointXYZRGB> m_pointCloudStitchedLastOne;
	Eigen::Matrix4d m_transformationMatrixLast;




};