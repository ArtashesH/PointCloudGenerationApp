#include "KniftFeatureDetectorCalculator.hpp"

#include "Constants.hpp"

#include <opencv2/imgproc/imgproc.hpp>

#include <tensorflow/lite/c/c_api.h>
#include <iostream>


KniftFeatureDetectorCalculator::KniftFeatureDetectorCalculator(const char* kniftModelPath)
{
	m_FeatureDetector = cv::ORB::create(
		constants::maxFeatures, constants::scaleFactor,
		constants::pyramidLevel, constants::kPatchSize - 1, 0, 2, constants::scoreType);

	m_kniftModel = TfLiteModelCreateFromFile(kniftModelPath);
}

void KniftFeatureDetectorCalculator::computeImagePyramid(const cv::Mat& inputImage, std::vector<cv::Mat>* imagePyramid) const
{
	if (imagePyramid == nullptr) {
		std::cerr << "Image Pyramid Vector Is Null\n";
		return;
	}
	cv::Mat srcImage{ inputImage };
	for (int i{ 0 }; i < constants::pyramidLevel; ++i) 
	{
		imagePyramid->push_back(srcImage);
		cv::resize(srcImage, srcImage, cv::Size(), 1.0f / constants::scaleFactor,
			1.0f / constants::scaleFactor);
	}
}

void KniftFeatureDetectorCalculator::extractPatch(const std::vector<cv::KeyPoint>& feature, const std::vector<cv::Mat>& imagePyramid, int s, int e, std::vector<cv::Mat>& patchMat) const
{
	for (int i{ s }; i <= e; ++i)
	{
		cv::Mat img{ imagePyramid[feature[i].octave] };
		float scaleFactor{ 1.f / (float)pow(constants::scaleFactor, feature[i].octave) };
		cv::Point2f center{ cv::Point2f(feature[i].pt.x * scaleFactor, feature[i].pt.y * scaleFactor) };
		cv::Mat rot{ cv::getRotationMatrix2D(center, feature[i].angle, 1.0) };
		rot.at<double>(0, 2) += constants::kPatchSize / 2 - center.x;
		rot.at<double>(1, 2) += constants::kPatchSize / 2 - center.y;
		// perform the affine transformation
		cv::warpAffine(img, patchMat[i], rot, cv::Size(constants::kPatchSize, constants::kPatchSize),
			cv::INTER_LINEAR);
	}
}

static TfLiteIntArray* TfLiteIntArrayCreate(int size) 
{
	TfLiteIntArray* ret =
		(TfLiteIntArray*)malloc(sizeof(*ret) + sizeof(ret->data[0]) * size);
	ret->size = size;
	return ret;
}

std::vector<float> KniftFeatureDetectorCalculator::descriptorFromTfLiteTensor(const std::vector<float>& inputData) const
{
	TfLiteInterpreterOptions* options{ TfLiteInterpreterOptionsCreate() };
	int nThreads;
#define ThreadingApproach
#ifdef ThreadingApproach
	nThreads = std::thread::hardware_concurrency();
	if (nThreads > 1) {
		--nThreads;
	}
#else
	nThreads = 1;
#endif
	TfLiteInterpreterOptionsSetNumThreads(options, nThreads);
	static TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(m_kniftModel, options);
	TfLiteInterpreterAllocateTensors(interpreter);
	TfLiteTensor* inputTensor =
		TfLiteInterpreterGetInputTensor(interpreter, 0);
	const TfLiteTensor* outputTensor =
		TfLiteInterpreterGetOutputTensor(interpreter, 0);
	TfLiteStatus from_status = TfLiteTensorCopyFromBuffer(
		inputTensor,
		inputData.data(),
		TfLiteTensorByteSize(inputTensor));
	TfLiteStatus interpreter_invoke_status = TfLiteInterpreterInvoke(interpreter);
	std::vector<float> outputData(constants::maxFeatures * constants::singleImageDescriptorSizeKNIFT);
	TfLiteStatus to_status = TfLiteTensorCopyToBuffer(
		outputTensor,
		outputData.data(),
		TfLiteTensorByteSize(outputTensor));
	
	return outputData;
}

FeatureStore::ImageInfo* KniftFeatureDetectorCalculator::generate(const std::string& uniqueIdentifier, const cv::Mat& img) const
{
	FeatureStore::ImageInfo* info = new FeatureStore::ImageInfo;

	cv::Mat grayscaleView;
	cv::cvtColor(img, grayscaleView, cv::COLOR_BGR2GRAY);

	// keypoints
	std::vector<cv::KeyPoint> keypoints;
	m_FeatureDetector->detect(grayscaleView, keypoints);
	if (keypoints.size() > constants::maxFeatures) {
		keypoints.resize(constants::maxFeatures);
	}

	// normalizedLandmarks
	std::vector<cv::Point2f> normalizedLandmarks((int)keypoints.size());
	for (int i{ 0 }; i < (int)keypoints.size(); ++i)
	{
		normalizedLandmarks[i].x = keypoints[i].pt.x / grayscaleView.cols;
		normalizedLandmarks[i].y = keypoints[i].pt.y / grayscaleView.rows;
	}

	// patches
	std::vector<cv::Mat> imagePyramid;
	computeImagePyramid(grayscaleView, &imagePyramid);
	std::vector<cv::Mat> patchMat;
	patchMat.resize(keypoints.size());
#define ThreadingApproach
#ifdef ThreadingApproach
	int nThreads = std::thread::hardware_concurrency();
	if (nThreads > 1) {
		--nThreads;
	}
	const int N{ (int)keypoints.size() };
	std::vector<std::thread> pool;
	const int len{ N / nThreads };
	int cur{ 0 };
	for (int i{ 0 }; i < nThreads; ++i)
	{
		int to_add{ i < N % nThreads ? 0 : -1};
		pool.push_back(std::thread(&KniftFeatureDetectorCalculator::extractPatch, this, std::ref(keypoints), std::ref(imagePyramid), cur, cur + len + to_add, std::ref(patchMat)));
		cur += len + 1 + to_add;
	}
	for (int i{ 0 }; i < nThreads; ++i) {
		if (pool[i].joinable()) {
			pool[i].join();
		}
	}
#else
	extractPatch(keypoints, imagePyramid, 0, keypoints.size() - 1, patchMat);

#endif
	
	const int batchSize{ constants::maxFeatures };
	TfLiteTensor tensor;
	tensor.type = kTfLiteFloat32;
	tensor.dims = TfLiteIntArrayCreate(4);
	tensor.dims->data[0] = batchSize;
	tensor.dims->data[1] = constants::kPatchSize;
	tensor.dims->data[2] = constants::kPatchSize;
	tensor.dims->data[3] = 1;
	int numBytes{ batchSize * constants::kPatchSize * constants::kPatchSize * sizeof(float) };
	tensor.data.data = malloc(numBytes);
	tensor.bytes = numBytes;
	tensor.allocation_type = kTfLiteArenaRw;
	float* tensorBuffer{ tensor.data.f };
	for (int i{ 0 }; i < keypoints.size(); i++) {
		for (int j{ 0 }; j < patchMat[i].rows; ++j) {
			for (int k{ 0 }; k < patchMat[i].cols; ++k) {
				*tensorBuffer++ = patchMat[i].at<uchar>(j, k) / 128.0f - 1.0f;
			}
		}
	}
	for (int i{ (int)keypoints.size() * constants::kPatchSize * constants::kPatchSize }; i < numBytes / 4; i++) {
		*tensorBuffer++ = 0;
	}
	
	std::vector<float> descriptors{ descriptorFromTfLiteTensor(std::vector<float>(tensor.data.f, tensor.data.f + numBytes / 4)) };
	std::vector<std::vector<float>> descriptorsCollection(constants::maxFeatures);
	for (int i{ 0 }; i < constants::maxFeatures; ++i)
	{
		std::vector<float> descriptor(constants::singleImageDescriptorSizeKNIFT);
		for (int j{ 0 }; j < constants::singleImageDescriptorSizeKNIFT; ++j) {
			descriptor[j] = descriptors[i * constants::singleImageDescriptorSizeKNIFT + j];
		}
		descriptorsCollection[i] = std::move(descriptor);
	}

	info->m_uniqueIdentifier = uniqueIdentifier; 
	info->m_keyPoints = std::move(keypoints);
	info->m_normalizedLandmarks = std::move(normalizedLandmarks);
	info->m_descriptorsCollection = std::move(descriptorsCollection);
	info->m_type = Feature2dType::eKNIFT_ORB;

	return info;
}

