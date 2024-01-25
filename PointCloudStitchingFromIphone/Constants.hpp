#pragma once

#include <opencv2/features2d.hpp>

namespace constants 
{
	static const float					fRatioTestFeatureMatching{ 0.35f };
	
	static const int					maxFeatures{ 300 };
	
	static const int					pyramidLevel{ 12 };
	// Pyramid decimation ratio.
	static const float					scaleFactor{ 1.2f };
	static const cv::ORB::ScoreType		scoreType{ cv::ORB::FAST_SCORE };


	static const int					kPatchSize{ 32 };
	static const int					singleImageDescriptorSizeKNIFT{ 40 };
	static const int					distanceBetweenMatchedKeypointsError{ 5 };

}

