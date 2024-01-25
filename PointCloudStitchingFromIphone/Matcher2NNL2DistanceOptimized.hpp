#pragma once

#include <vector>

#include <opencv2/core/types.hpp>

class Matcher2NNL2DistanceOptimized
{

public:	
	void knnMatch(const std::vector<std::vector<float>>& queryDescriptors, const std::vector<std::vector<float>>& trainDescriptors,
				  std::vector<std::vector<cv::DMatch>>& matches1to2, std::vector<std::vector<cv::DMatch>>& matches2to1);

private:
	inline static float normL2(const float* a, const float* b, const int n = 40);

};