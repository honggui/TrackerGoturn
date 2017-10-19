#include "buildH5Dataset.h"
#include <iostream>
using namespace H5;
using namespace std;
using namespace cv;
using namespace caffe;


#define INPUT_SIZE 227
#define NUM_CHANNELS 3

static double generateRandomLaplacian(double b, double m)
{
    double t = (double)rand() / (RAND_MAX);
    double n = (double)rand() / (RAND_MAX);

    if (t > 0.5)
        return m + b*log(n);
    else
        return m - b*log(n);
}


static Rect2f my_anno2rect(vector<Point2f> annoBB)
{
    Rect2f rectBB;
    rectBB.x = min(annoBB[0].x, annoBB[1].x);
    rectBB.y = min(annoBB[0].y, annoBB[2].y);
    rectBB.width = fabs(annoBB[0].x - annoBB[1].x);
    rectBB.height = fabs(annoBB[0].y - annoBB[2].y);

    return rectBB;
}

//Number of samples in batch
const int samplesInBatch = 50;

//Number of samples to mine from video frame
const int samplesInFrame = 10;

//Number of samples to mine from still image
const int samplesInImage = 10;

//Padding coefficients for Target/Search Region
const double padTarget = 2.0;
const double padSearch = 2.0;

//Scale parameters for Laplace distribution for Translation/Scale
const double bX = 1.0/10;
const double bY = 1.0/10;
const double bS = 1.0/15;

//Limits of scale changes
const double Ymax = 1.4;
const double Ymin = 0.6;

//Lower boundary constraints for random samples (sample should include X% of target BB)
const double minX = 0.5;
const double minY = 0.5;

//Structure of sample for training
struct TrainingSample
{
    Mat targetPatch;
    Mat searchPatch;
    //Output bounding box on search patch
    Rect2f targetBB;
};

//Laplacian distribution
double generateRandomLaplacian(double b, double m);


static vector <TrainingSample> my_gatherFrameSamples(Mat prevFrame, Mat currFrame, Rect2f prevBB, Rect2f currBB)
{
    vector <TrainingSample> trainingSamples;
    Point2f currCenter, prevCenter;
    Rect2f targetPatchRect, searchPatchRect;
    Mat targetPatch, searchPatch;
    Mat prevFramePadded, currFramePadded;

    //Crop Target Patch

    //Padding

    //Previous frame GTBBs center
    prevCenter.x = prevBB.x + prevBB.width / 2;
    prevCenter.y = prevBB.y + prevBB.height / 2;

    targetPatchRect.width = (float)(prevBB.width*padTarget);
    targetPatchRect.height = (float)(prevBB.height*padTarget);
    targetPatchRect.x = (float)(prevCenter.x - prevBB.width*padTarget / 2.0 + targetPatchRect.width);
    targetPatchRect.y = (float)(prevCenter.y - prevBB.height*padTarget / 2.0 + targetPatchRect.height);

    copyMakeBorder(prevFrame, prevFramePadded, (int)targetPatchRect.height, (int)targetPatchRect.height, (int)targetPatchRect.width, (int)targetPatchRect.width, BORDER_REPLICATE);

    targetPatch = prevFramePadded(targetPatchRect);


    for (int i = 0; i < samplesInFrame; i++)
    {
        TrainingSample sample;

        //Current frame GTBBs center
        currCenter.x = (float)(currBB.x + currBB.width / 2.0);
        currCenter.y = (float)(currBB.y + currBB.height / 2.0);

        //Generate and add random Laplacian distribution (Scaling from target size)
        double dx, dy, ds;
        dx = generateRandomLaplacian(bX, 0)*prevBB.width;
        dy = generateRandomLaplacian(bY, 0)*prevBB.height;
        ds = generateRandomLaplacian(bS, 1);

        //Limit coefficients
        dx = min(dx, (double)prevBB.width);
        dx = max(dx, (double)-prevBB.width);
        dy = min(dy, (double)prevBB.height);
        dy = max(dy, (double)-prevBB.height);
        ds = min(ds, Ymax);
        ds = max(ds, Ymin);

        searchPatchRect.width = (float)(prevBB.width*padSearch*ds);
        searchPatchRect.height =(float)(prevBB.height*padSearch*ds);
        searchPatchRect.x = (float)(currCenter.x + dx - searchPatchRect.width / 2.0 + searchPatchRect.width);
        searchPatchRect.y = (float)(currCenter.y + dy - searchPatchRect.height / 2.0 + searchPatchRect.height);
        copyMakeBorder(currFrame, currFramePadded, (int)searchPatchRect.height, (int)searchPatchRect.height, (int)searchPatchRect.width, (int)searchPatchRect.width, BORDER_REPLICATE);
        searchPatch = currFramePadded(searchPatchRect);

        //Calculate Relative GTBB in search patch
        Rect2f relGTBB;
        relGTBB.width = currBB.width;
        relGTBB.height = currBB.height;
        relGTBB.x = currBB.x - searchPatchRect.x + searchPatchRect.width;
        relGTBB.y = currBB.y - searchPatchRect.y + searchPatchRect.height;

        //Link to the sample struct
        sample.targetPatch = targetPatch.clone();
        sample.searchPatch = searchPatch.clone();
        sample.targetBB = relGTBB;

        trainingSamples.push_back(sample);
    }

    return trainingSamples;
}

void buildH5Datasets(string fileName, int samplesNum)
{
	//Create HDF5 database file
	hid_t fileID = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

	//Create an empty chunked extensible datasets
	hsize_t dims[4] = { 0, NUM_CHANNELS, INPUT_SIZE, INPUT_SIZE };
	hsize_t maxDims[4] = { H5S_UNLIMITED, NUM_CHANNELS, INPUT_SIZE, INPUT_SIZE };
	hid_t fileSpace = H5Screate_simple(4, dims, maxDims);
	hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
	H5Pset_layout(plist, H5D_CHUNKED);
	hsize_t chunkDims[4] = { 10, NUM_CHANNELS, INPUT_SIZE, INPUT_SIZE };
	H5Pset_chunk(plist, 4, chunkDims);
	hid_t data1 = H5Dcreate(fileID, "data1", H5T_NATIVE_FLOAT, fileSpace, H5P_DEFAULT, plist, H5P_DEFAULT);
	hid_t data2 = H5Dcreate(fileID, "data2", H5T_NATIVE_FLOAT, fileSpace, H5P_DEFAULT, plist, H5P_DEFAULT);
	dims[0] = 0;
	dims[1] = 4;
	maxDims[0] = H5S_UNLIMITED;
	maxDims[1] = 4;
	fileSpace = H5Screate_simple(2, dims, maxDims);
	chunkDims[0] = 10;
	chunkDims[1] = 4;
	H5Pset_chunk(plist, 2, chunkDims);
	hid_t label = H5Dcreate(fileID, "label", H5T_NATIVE_FLOAT, fileSpace, H5P_DEFAULT, plist, H5P_DEFAULT);

	//Create DATA/LABEL pointers and allocate memory
	int width, height;
	width = INPUT_SIZE;
	height = INPUT_SIZE;

	//Open ALOV300++ Dataset
	Ptr<cv::datasets::TRACK_alov> alovDataset = cv::datasets::TRACK_alov::create();
	alovDataset->loadAnnotatedOnly("/opt/imagedata++");
	//Extract patches & lables
	for (int k = 0; k < samplesNum; k++)
	{
		Mat prevFrame, currFrame;
		vector <vector <Mat> > targetPatchesSplitted;
		vector <vector <Mat> > searchPatchesSplitted;
		vector <TrainingSample> trainingSamples, tmpSamples;
		Rect2f prevGTBB, currGTBB;
		int datasetID = (rand() % alovDataset->getDatasetsNum()) + 1;
		//cout << alovDataset->getDatasetsNum() << " " << datasetID << endl;
		int datasetLength = alovDataset->getDatasetLength(datasetID);
		int frameID = (rand() % (datasetLength-1)) + 1;
		//cout << datasetLength << " " << frameID << endl;
		//prevGTBB = (Rect2f)gtr::anno2rect((vector<Point2f>)alovDataset->getGT(datasetID, frameID));
		//currGTBB = (Rect2f)gtr::anno2rect((vector<Point2f>)alovDataset->getGT(datasetID, frameID + 1));
		prevGTBB = my_anno2rect((vector<Point2f>)alovDataset->getGT(datasetID, frameID));
		currGTBB = my_anno2rect((vector<Point2f>)alovDataset->getGT(datasetID, frameID + 1));

		alovDataset->getFrame(prevFrame, datasetID, frameID);
		alovDataset->getFrame(currFrame, datasetID, frameID + 1);
		//trainingSamples = gtr::gatherFrameSamples(prevFrame, currFrame, prevGTBB, currGTBB);
                trainingSamples = my_gatherFrameSamples(prevFrame, currFrame, prevGTBB, currGTBB);
		int N = trainingSamples.size();


		//Shuffle data from one dataset
		random_shuffle(trainingSamples.begin(), trainingSamples.end());

		/*for (int i = 0; i < trainingSamples.size(); i++)
		{
		cv::gtr::TrainingSample  sample = trainingSamples[i];
		rectangle(sample.searchPatch, sample.targetBB, Scalar(0, 0, 255));
		imshow("1", sample.targetPatch);
		imshow("2", sample.searchPatch);
		waitKey();
		}*/

		//Warp DATA/LABEL pointers to their data
		int fullVolume = N * NUM_CHANNELS * INPUT_SIZE * INPUT_SIZE;
		float* targetData = new float[fullVolume];
		float* searchData = new float[fullVolume];
		float* labelData = new float[N * 4];
		float* pointer;
		int offset = 0;

		for (int i = 0; i < N; i++)
		{
			offset = i*NUM_CHANNELS*INPUT_SIZE*INPUT_SIZE;

			vector <Mat> targetPatchSplitted;
			pointer = targetData + offset;
			for (int j = 0; j < 3; ++j)
			{
				Mat channel(height, width, CV_32FC1, pointer);
				targetPatchSplitted.push_back(channel);
				pointer += width * height;
			}
			targetPatchesSplitted.push_back(targetPatchSplitted);

			vector <Mat> searchPatchSplitted;
			pointer = searchData + offset;
			for (int j = 0; j < 3; ++j)
			{
				Mat channel(height, width, CV_32FC1, pointer);
				searchPatchSplitted.push_back(channel);
				pointer += width * height;
			}
			searchPatchesSplitted.push_back(searchPatchSplitted);
		}

		//Split channels and map to the DATA/LABELS
		for (int i = 0; i < N; i++)
		{
			Mat targetPatch, searchPatch;
			Rect2f relBB;
			targetPatch = trainingSamples[i].targetPatch;
			searchPatch = trainingSamples[i].searchPatch;
			relBB = trainingSamples[i].targetBB;

			//Scale parameters
			float dx = (float)INPUT_SIZE / searchPatch.cols;
			float dy = (float)INPUT_SIZE / searchPatch.rows;
			//cout << dx <<  "   " << dy << endl;

			//Preprocess
			//Resize
			resize(targetPatch, targetPatch, Size(INPUT_SIZE, INPUT_SIZE));
			resize(searchPatch, searchPatch, Size(INPUT_SIZE, INPUT_SIZE));

			//Mean Subtract
			targetPatch = targetPatch - 128;
			searchPatch = searchPatch - 128;

			//Convert to Float type
			targetPatch.convertTo(targetPatch, CV_32FC1);
			searchPatch.convertTo(searchPatch, CV_32FC1);

			//Split data to mapped memory
			split(targetPatch, targetPatchesSplitted[i]);
			split(searchPatch, searchPatchesSplitted[i]);

			//Labels mapping
			labelData[i * 4] = dx * relBB.x;
			labelData[i * 4 + 1] = dy * relBB.y;
			labelData[i * 4 + 2] = dx * (relBB.x + relBB.width);
			labelData[i * 4 + 3] = dy * (relBB.y + relBB.height);
        //print out
        //cout << "labeldata " << labelData[i * 4] << " " << labelData[i * 4 + 1] << " " << labelData[i * 4 + 2] << " " << labelData[i * 4 + 3] << endl;
		}
		//Write Patches Data
		//Memory Space
		dims[0] = N;
		dims[1] = NUM_CHANNELS;
		dims[2] = INPUT_SIZE;
		dims[3] = INPUT_SIZE;
		hid_t memSpace = H5Screate_simple(4, dims, NULL);
		//Get current dims
		fileSpace = H5Dget_space(data1);
		H5Sget_simple_extent_dims(fileSpace, dims, NULL);
		//cout << "Current data size: " << dims[0] << endl;
		//Extend dataset by N elements
		int currNum = dims[0];
		dims[0] = dims[0] + N;
		dims[1] = NUM_CHANNELS;
		dims[2] = INPUT_SIZE;
		dims[3] = INPUT_SIZE;
		H5Dset_extent(data1, dims);
		H5Dset_extent(data2, dims);
		//File Space
		fileSpace = H5Dget_space(data1);
		H5Sget_simple_extent_dims(fileSpace, dims, NULL);
		cout << "New data size: " << dims[0] << endl;
		hsize_t start[4] = { currNum, 0, 0, 0 };
		hsize_t count[4] = { N, NUM_CHANNELS, INPUT_SIZE, INPUT_SIZE };
		H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, start, NULL, count, NULL);
		H5Dwrite(data1, H5T_NATIVE_FLOAT, memSpace, fileSpace, H5P_DEFAULT, targetData);
		H5Dwrite(data2, H5T_NATIVE_FLOAT, memSpace, fileSpace, H5P_DEFAULT, searchData);

		//Write Lable Data
		//Memory Space
		dims[0] = N;
		dims[1] = 4;
		memSpace = H5Screate_simple(2, dims, NULL);
		//Get current dims
		fileSpace = H5Dget_space(label);
		H5Sget_simple_extent_dims(fileSpace, dims, NULL);
		//cout << "Current label size: " << dims[0] << endl;
		//Extend dataset by N elements
		currNum = dims[0];
		dims[0] = dims[0] + N;
		dims[1] = 4;
		H5Dset_extent(label, dims);
		//File Space
		fileSpace = H5Dget_space(label);
		H5Sget_simple_extent_dims(fileSpace, dims, NULL);
		//cout << "New label size: " << dims[0] << endl;
		start[0] = currNum;
		start[1] = 0;
		count[0] = N;
		count[1] = 4;
		H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, start, NULL, count, NULL);
		H5Dwrite(label, H5T_NATIVE_FLOAT, memSpace, fileSpace, H5P_DEFAULT, labelData);

		H5Sclose(memSpace);

		delete[] targetData;
		delete[] searchData;
		delete[] labelData;

		cout << "Added data from Dataset: " << datasetID  << "    Frame: " << frameID << endl;
	}
	H5Sclose(fileSpace);
	H5Pclose(plist);
	H5Dclose(data1);
	H5Dclose(data2);
	H5Dclose(label);
	H5Fclose(fileID);
	cout << "HDF5 Dataset Generation Finished..." << endl;
	//getchar();
}
