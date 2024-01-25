#pragma once

#include <string>
#include <vector>
#include <set>
#include <opencv2/features2d.hpp>
#include <tensorflow/lite/interpreter.h>

enum Feature2dType 
{
	eKNIFT_ORB,
	eORB,
	eSIFT
};

class FeatureStore 
{
public:
	virtual ~FeatureStore() {}

public:
	typedef std::vector<float> FloatDescriptor;
	typedef std::vector<cv::KeyPoint> KeyPoints;
	typedef std::vector<cv::Point2f> NormalizedLandmarks;

	typedef std::vector<FloatDescriptor> FloatDescriptorsCollection;


	struct ImageInfo {

		std::string m_uniqueIdentifier;
		FloatDescriptorsCollection m_descriptorsCollection;
		KeyPoints m_keyPoints;
		NormalizedLandmarks m_normalizedLandmarks;
		Feature2dType m_type;

		cv::Size m_imageSize;
		cv::Size_<float> m_phisicalSizeInMM;	
		std::string m_url;
		std::string m_localPath;
		bool m_isPlanar;
	};

public:
	virtual bool load(const std::string& serializedFeatureStorePath) = 0;
	virtual bool save(const std::string& path) = 0;

	/// <summary>
	/// deep copy of the entire info
	/// </summary>
	/// <param name="info"></param>
	virtual void add(const ImageInfo* info) = 0;


	virtual const ImageInfo* matchSingleImageInfo(const ImageInfo& s, std::vector<std::vector<cv::Point2f>>& matchedKeypoints) const = 0;

protected:
	std::vector<const ImageInfo*> m_allImagesInfo;
};





