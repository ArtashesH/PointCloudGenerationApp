#pragma once

#include "FeatureStore.hpp"
#include "ImageInfoGenerator.hpp"

class TfLiteModel;

class KniftFeatureDetectorCalculator : public ImageInfoGenerator
{
public:
	KniftFeatureDetectorCalculator(const char* kniftModelPath);
	FeatureStore::ImageInfo* generate(const std::string& uniqueIdentifier, const cv::Mat& img) const;

private:
	// Create image pyramid based on input image.
	void computeImagePyramid(const cv::Mat& inputImage, std::vector<cv::Mat>* imagePyramid) const;

	// Extract the patch for single feature with image pyramid.
	void extractPatch(const std::vector<cv::KeyPoint>& feature, const std::vector<cv::Mat>& imagePyramid, int s, int e, std::vector<cv::Mat>& patchMat) const;


	std::vector<float> descriptorFromTfLiteTensor(const std::vector<float>& inputData) const;

private:

	cv::Ptr<cv::Feature2D> m_FeatureDetector;

private:
	TfLiteModel* m_kniftModel;

};