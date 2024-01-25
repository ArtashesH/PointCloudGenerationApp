#include "Knn2BFMatcherBasedFeatureStore.hpp"
#include "Matcher2NNL2DistanceOptimized.hpp"
#include "Constants.hpp"
#include <iostream>
#include <thread>

#include <opencv2/calib3d/calib3d.hpp>

Knn2BFMatcherBasedFeatureStore::Knn2BFMatcherBasedFeatureStore()
{
	m_allImagesInfo.reserve(300);
}

Knn2BFMatcherBasedFeatureStore::~Knn2BFMatcherBasedFeatureStore()
{
	for (int i{ 0 }; i < (int)m_allImagesInfo.size(); ++i) {
		delete m_allImagesInfo[i];
	}
}

void Knn2BFMatcherBasedFeatureStore::add(const ImageInfo* info)
{
	m_allImagesInfo.push_back(info);
}

int Knn2BFMatcherBasedFeatureStore::findDistantMatchesCount(std::vector<cv::Point2i>& goodMatches, const std::vector<cv::KeyPoint>& queryKeypoints, const std::vector<cv::KeyPoint>& initialKeypoints) const
{
	if (goodMatches.size() < 4) {
		return 0;
	}
	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;
	for (int i{ 0 }; i < goodMatches.size(); ++i)
	{
		obj.push_back(queryKeypoints[goodMatches[i].x].pt);
		scene.push_back(initialKeypoints[goodMatches[i].y].pt);
	}
	cv::Mat H{ cv::findHomography(obj, scene, cv::RANSAC) };
	if (H.empty()) {
		return goodMatches.size();
	}
	int count{ 0 };
	for (int i{ 0 }; i < obj.size(); ++i)
	{
		cv::Point2f p{ obj[i] };
		float m[3][1];
		m[0][0] = p.x;
		m[1][0] = p.y;
		m[2][0] = 1.0;

		cv::Mat v(3, 1, CV_64FC1);
		v.at<double>(0, 0) = p.x;
		v.at<double>(1, 0) = p.y;
		v.at<double>(2, 0) = 1.0f;

		cv::Mat r = H * v;

		cv::Point2f q1(r.at<double>(0, 0) / r.at<double>(2, 0),
					   r.at<double>(1, 0) / r.at<double>(2, 0));

		cv::Point2f q2{ scene[i] };

		float d = std::sqrt((q1.x - q2.x) * (q1.x - q2.x) + (q1.y - q2.y) * (q1.y - q2.y));
		if (d > constants::distanceBetweenMatchedKeypointsError)
		{
			++count;
			goodMatches.erase(goodMatches.begin() + i);
			obj.erase(obj.begin() + i);
			scene.erase(scene.begin() + i);
			i--;
		}
	}
	return count;
}

void Knn2BFMatcherBasedFeatureStore::makeMatches(const ImageInfo& queryImage, int s, int e, std::vector<int>& matchedCounts, MatchInfo& ans, std::vector<std::vector<cv::Point2i>>& goodMatches) const
{
	std::vector<std::vector<cv::DMatch>> matches1to2(queryImage.m_descriptorsCollection.size(), std::vector<cv::DMatch>(2));
	Matcher2NNL2DistanceOptimized matcher;
	
	int* matchedFeaturesArrayPtr = new int[queryImage.m_descriptorsCollection.size()];
	int maxMatchedCount{ 0 }, matchedIndex{ 0 };
	
	ans.matchedIndex = 0;
	ans.maxMatchedCountOccurrence = (int)1e9;
	ans.maxMatchedCount = -1;
	
	for (int i{ s }; i <= e; ++i)
	{
		int matchedCount{ 0 };

		const ImageInfo* initialImage{ m_allImagesInfo[i] };
		
		std::vector<std::vector<cv::DMatch>> matches2to1(initialImage->m_descriptorsCollection.size(), std::vector<cv::DMatch>(2));
		
		matcher.knnMatch(queryImage.m_descriptorsCollection, initialImage->m_descriptorsCollection, matches1to2, matches2to1);
	
		memset(matchedFeaturesArrayPtr, -1, sizeof(int) * queryImage.m_descriptorsCollection.size());
		goodMatches[i].reserve((int)queryImage.m_descriptorsCollection.size());

		for (int it{ 0 }; it < (int)queryImage.m_descriptorsCollection.size(); ++it)
		{
			const cv::DMatch& one{ matches1to2[it][0] };
			const cv::DMatch& two{ matches1to2[it][1] };
	
			const int queryIdx1{ one.queryIdx };
			const int trainIdx1{ one.trainIdx };
			const float distance1{ one.distance };
			const float distance2{ two.distance };
	
			//if (distance1 < constants::fRatioTestFeatureMatching * distance2)
			{
				matchedFeaturesArrayPtr[queryIdx1] = trainIdx1;
			}
		}
		
		for (int it{ 0 }; it < initialImage->m_descriptorsCollection.size(); ++it)
		{
			const cv::DMatch& one{ matches2to1[it][0] };
			const cv::DMatch& two{ matches2to1[it][1] };
	
			const int queryIdx1{ one.queryIdx };
			const int trainIdx1{ one.trainIdx };
	
			if (matchedFeaturesArrayPtr[trainIdx1] == queryIdx1 
				&& trainIdx1 < queryImage.m_keyPoints.size() && queryIdx1 < initialImage->m_keyPoints.size()) 
			{
				++matchedCount;
				goodMatches[i].push_back(cv::Point2f(trainIdx1, queryIdx1));
			}
		}
		
		//matchedCount -= findDistantMatchesCount(goodMatches[i], queryImage.m_keyPoints, initialImage->m_keyPoints);
		
		if (matchedCount > ans.maxMatchedCount)
		{
			ans.maxMatchedCount = matchedCount;
			ans.maxMatchedCountOccurrence = 1;
			ans.matchedIndex = i;
		}
		else if (matchedCount == ans.maxMatchedCount)
		{
			++ans.maxMatchedCountOccurrence;
			ans.matchedIndex = -1;
		}

		matchedCounts[i] = matchedCount;
	}
	
	delete[] matchedFeaturesArrayPtr;
}

const FeatureStore::ImageInfo* Knn2BFMatcherBasedFeatureStore::matchSingleImageInfo(const ImageInfo& queryImage, std::vector<std::vector<cv::Point2f>>& matchedKeypoints) const
{
	MatchInfo ans{ 0, (int)1e9, -1 };
	std::vector<int> matchedCounts(m_allImagesInfo.size());
	std::vector<std::vector<cv::Point2i>> goodMatches(m_allImagesInfo.size());

//#define ThreadingApproach // here multithreading is useless because m_allImagesInfo.size() == 1
#ifdef ThreadingApproach

	const int N{ (int)m_allImagesInfo.size() };
	int nThreads{ (int)std::thread::hardware_concurrency() };
	if (nThreads > 1) {
		--nThreads;
	}
	std::vector<std::thread> pool;
	const int len{ N / nThreads };
	int cur{ 0 };
	std::vector<MatchInfo> matchesEachThread(nThreads);
	for (int i{ 0 }; i < nThreads; ++i)
	{
		int to_add{ i < N % nThreads ? 0 : -1 };
		pool.push_back(std::thread(&Knn2BFMatcherBasedFeatureStore::makeMatches, this, std::ref(queryImage), cur, cur + len + to_add, std::ref(matchedCounts), std::ref(matchesEachThread[i]), std::ref(goodMatches)));
		cur += len + 1 + to_add;
	}
	for (int i{ 0 }; i < nThreads; ++i)
	{
		if (pool[i].joinable()) {
			pool[i].join();
		}
		if (matchesEachThread[i].maxMatchedCount > ans.maxMatchedCount)
		{
			ans.maxMatchedCount = matchesEachThread[i].maxMatchedCount;
			ans.maxMatchedCountOccurrence = matchesEachThread[i].maxMatchedCountOccurrence;
			ans.matchedIndex = matchesEachThread[i].matchedIndex;
		}
		else if (matchesEachThread[i].maxMatchedCount == ans.maxMatchedCount) {
			ans.maxMatchedCountOccurrence += matchesEachThread[i].maxMatchedCountOccurrence;
			ans.matchedIndex = -1;
		}
	}
#else

	makeMatches(queryImage, 0, m_allImagesInfo.size() - 1, matchedCounts, ans, goodMatches);

#endif

	for (int i = 0; i < goodMatches[0].size(); ++i)
	{
		int targetKeypointIndex = goodMatches[0][i].x;
		int queryKeypointIndex = goodMatches[0][i].y;
		matchedKeypoints[0].push_back(m_allImagesInfo[0]->m_keyPoints[queryKeypointIndex].pt);
		matchedKeypoints[1].push_back(queryImage.m_keyPoints[targetKeypointIndex].pt);
	}

	return m_allImagesInfo[ans.matchedIndex];

}