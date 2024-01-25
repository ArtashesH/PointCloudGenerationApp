#include "Matcher2NNL2DistanceOptimized.hpp"
#include <iostream>
#include <immintrin.h>

inline float Matcher2NNL2DistanceOptimized::normL2(const float* a, const float* b, const int n)
{
    int j{ 0 };
    float result{ 0.f };
    float CV_DECL_ALIGNED(16) buf[4];
    __m128 d0 = _mm_setzero_ps(), d1 = _mm_setzero_ps();

    for (; j <= n - 8; j += 8)
    {
        __m128 t0 = _mm_sub_ps(_mm_loadu_ps(a + j), _mm_loadu_ps(b + j));
        __m128 t1 = _mm_sub_ps(_mm_loadu_ps(a + j + 4), _mm_loadu_ps(b + j + 4));
        d0 = _mm_add_ps(d0, _mm_mul_ps(t0, t0));
        d1 = _mm_add_ps(d1, _mm_mul_ps(t1, t1));
    }
    _mm_store_ps(buf, _mm_add_ps(d0, d1));
    result = buf[0] + buf[1] + buf[2] + buf[3];

    return std::sqrt(result);
}

void Matcher2NNL2DistanceOptimized::knnMatch(const std::vector<std::vector<float>>& queryDescriptors, const std::vector<std::vector<float>>& trainDescriptors,
												  std::vector<std::vector<cv::DMatch>>& matches1to2, std::vector<std::vector<cv::DMatch>>& matches2to1)
{
    std::vector<const float*> matches1to2Ptr(queryDescriptors.size());
    std::vector<const float*> matches2to1Ptr(trainDescriptors.size());

	for (int i{ 0 }; i < (int)trainDescriptors.size(); ++i)
	{
		matches2to1[i][0].queryIdx = matches2to1[i][1].queryIdx = i;
		matches2to1[i][0].distance = matches2to1[i][1].distance = FLT_MAX;
        matches2to1Ptr[i] = trainDescriptors[i].data();
	}

    for (int i{ 0 }; i < (int)queryDescriptors.size(); ++i)
    {
        matches1to2Ptr[i] = queryDescriptors[i].data();
    }

	float minDistance1, minDistance2, distance;
	int index1, index2;
	cv::DMatch match1, match2;

	for (int i{ 0 }; i < (int)queryDescriptors.size(); ++i)
	{
		minDistance1 = minDistance2 = FLT_MAX;
		index1 = index2 = -1;
		match1.queryIdx = match2.queryIdx = i;

		for (int j{ 0 }; j < (int)trainDescriptors.size(); ++j)
		{
            distance = normL2(matches1to2Ptr[i], matches2to1Ptr[j]);
            
            if (distance < minDistance1)
            {
                minDistance2 = minDistance1;
                index2 = index1;

                minDistance1 = distance;
                index1 = j;
            }

            else if (distance < minDistance2)
            {
                minDistance2 = distance;
                index2 = j;
            }

            auto& matches2to1_j_0 = matches2to1[j][0];
            auto& matches2to1_j_1 = matches2to1[j][1];

            if (distance < matches2to1_j_0.distance)
            {
                matches2to1_j_1 = matches2to1_j_0;

                matches2to1_j_0.distance = distance;
                matches2to1_j_0.trainIdx = i;
            }

            else if (distance < matches2to1_j_1.distance)
            {
                matches2to1_j_1.distance = distance;
                matches2to1_j_1.trainIdx = i;
            }
		}

        match1.trainIdx = index1;
        match1.distance = minDistance1;

        match2.trainIdx = index2;
        match2.distance = minDistance2;

        matches1to2[i][0] = match1;
        matches1to2[i][1] = match2;

	}
}