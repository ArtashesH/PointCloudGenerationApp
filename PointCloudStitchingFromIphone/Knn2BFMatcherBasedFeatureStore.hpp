#pragma once

#include "FeatureStore.hpp"

class Knn2BFMatcherBasedFeatureStore : public FeatureStore
{
public:
	Knn2BFMatcherBasedFeatureStore();
	~Knn2BFMatcherBasedFeatureStore();

	virtual bool load(const std::string& serializedFeatureStorePath) { return false; }
	virtual bool save(const std::string& path) { return false; }


	void add(const ImageInfo* info);

	
	const ImageInfo* matchSingleImageInfo(const ImageInfo& s, std::vector<std::vector<cv::Point2f>>& matchedKeypoints) const;

private:

	struct MatchInfo
	{
		int maxMatchedCount;
		int maxMatchedCountOccurrence;
		int matchedIndex;
	};

private:
	void makeMatches(const ImageInfo& queryImage, int s, int e, std::vector<int>& maxMatchedCounts, MatchInfo& ans, std::vector<std::vector<cv::Point2i>>& goodMatches) const;
	int findDistantMatchesCount(std::vector<cv::Point2i>& goodMatches, const std::vector<cv::KeyPoint>& queryKeypoints, const std::vector<cv::KeyPoint>& initialKeypoints) const;
};
