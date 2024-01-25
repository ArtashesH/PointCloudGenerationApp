#pragma once

#include "FeatureStore.hpp"

class ImageInfoGenerator
{
public:
	virtual ~ImageInfoGenerator() {}
public:
	virtual FeatureStore::ImageInfo* generate(const std::string& uniqueIdentifier, const cv::Mat& img) const = 0;
};