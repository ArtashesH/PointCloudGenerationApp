#include <unordered_map>
#include <unordered_set>
#include "PointCloudStitchingFromIphone.h"


int main()
{


	pointCloudStitcher cloudStitcher;

	std::string rootFolderPath = "InputFolderPathWithDepthAndColorImages";


	for (int i = 0; i < 100; i+=1) {
		std::string firstColorImgPath = rootFolderPath + "color_" + std::to_string(i) + ".png";
		std::string firstDepthImgPath = rootFolderPath + "depth_" + std::to_string(i) + ".png";

		std::string secondColorImgPath = rootFolderPath + "color_" + std::to_string(i+1) + ".png";
		std::string secondDepthImgPath = rootFolderPath + "depth_" + std::to_string(i+1) + ".png";

		cloudStitcher.stitchTwoPointClouds(firstColorImgPath, firstDepthImgPath, secondColorImgPath, secondDepthImgPath);


	}

	cloudStitcher.writeResultPointCloud();

	return 0;
}


int main1()
{
	   
	    std::unordered_map<int, std::string> unorderedMap;
		unorderedMap[3] = "Three";
		unorderedMap[1] = "One";
		unorderedMap[2] = "Two";

		for (const auto& pair : unorderedMap) {
			std::cout << pair.first << ": " << pair.second << std::endl;
		}

		return 0;
	
}