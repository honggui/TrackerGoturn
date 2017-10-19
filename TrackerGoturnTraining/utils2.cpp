#include "utils.h"

using namespace std;
using namespace cv;
using namespace cv::datasets;
using namespace caffe;

void trainNet()
{
	boost::shared_ptr <Net <float> > net;
	NetParameter caffenetParam, goturnParam;
	SolverParameter solverParam;
	ReadSolverParamsFromTextFileOrDie("goturnSolver.prototxt", &solverParam);
	ReadNetParamsFromBinaryFileOrDie("squeezenet_v1.1.caffemodel", &caffenetParam);

	boost::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solverParam));

	//Replace CONV layers weights with SqueezeNet pretrain model values
	int numSourceLayers = caffenetParam.layer_size();
	net = solver->net();
	for (int i = 0; i < numSourceLayers; i++)
	{
		const LayerParameter& sourceLayer = caffenetParam.layer(i);
		boost::shared_ptr <Layer<float> > targetLayer[2];

		if (sourceLayer.name() == "conv1")
		{
			targetLayer[0] = net->layer_by_name("conv1");
			targetLayer[1] = net->layer_by_name("conv1_p");
		}
		else if (sourceLayer.name() == "fire2/squeeze1x1")
		{
			targetLayer[0] = net->layer_by_name("fire2/squeeze1x1");
			targetLayer[1] = net->layer_by_name("fire2/squeeze1x1_p");
		}
                else if (sourceLayer.name() == "fire2/expand1x1")
		{
			targetLayer[0] = net->layer_by_name("fire2/expand1x1");
			targetLayer[1] = net->layer_by_name("fire2/expand1x1_p");
                }
                else if (sourceLayer.name() == "fire2/expand3x3")
		{
			targetLayer[0] = net->layer_by_name("fire2/expand3x3");
			targetLayer[1] = net->layer_by_name("fire2/expand3x3_p");
                }
                else if (sourceLayer.name() == "fire3/squeeze1x1")
		{
			targetLayer[0] = net->layer_by_name("fire3/squeeze1x1");
			targetLayer[1] = net->layer_by_name("fire3/squeeze1x1_p");
		}
                else if (sourceLayer.name() == "fire3/expand1x1")
		{
			targetLayer[0] = net->layer_by_name("fire3/expand1x1");
			targetLayer[1] = net->layer_by_name("fire3/expand1x1_p");
                }
                else if (sourceLayer.name() == "fire3/expand3x3")
		{
			targetLayer[0] = net->layer_by_name("fire3/expand3x3");
			targetLayer[1] = net->layer_by_name("fire3/expand3x3_p");
                }
                else if (sourceLayer.name() == "fire4/squeeze1x1")
		{
			targetLayer[0] = net->layer_by_name("fire4/squeeze1x1");
			targetLayer[1] = net->layer_by_name("fire4/squeeze1x1_p");
		}
                else if (sourceLayer.name() == "fire4/expand1x1")
		{
			targetLayer[0] = net->layer_by_name("fire4/expand1x1");
			targetLayer[1] = net->layer_by_name("fire4/expand1x1_p");
                }
                else if (sourceLayer.name() == "fire4/expand3x3")
		{
			targetLayer[0] = net->layer_by_name("fire4/expand3x3");
			targetLayer[1] = net->layer_by_name("fire4/expand3x3_p");
                }
		else if (sourceLayer.name() == "fire5/squeeze1x1")
		{
			targetLayer[0] = net->layer_by_name("fire5/squeeze1x1");
			targetLayer[1] = net->layer_by_name("fire5/squeeze1x1_p");
		}
                else if (sourceLayer.name() == "fire5/expand1x1")
		{
			targetLayer[0] = net->layer_by_name("fire5/expand1x1");
			targetLayer[1] = net->layer_by_name("fire5/expand1x1_p");
                }
                else if (sourceLayer.name() == "fire5/expand3x3")
		{
			targetLayer[0] = net->layer_by_name("fire5/expand3x3");
			targetLayer[1] = net->layer_by_name("fire5/expand3x3_p");
                }
                else if (sourceLayer.name() == "fire6/squeeze1x1")
		{
			targetLayer[0] = net->layer_by_name("fire6/squeeze1x1");
			targetLayer[1] = net->layer_by_name("fire6/squeeze1x1_p");
		}
                else if (sourceLayer.name() == "fire6/expand1x1")
		{
			targetLayer[0] = net->layer_by_name("fire6/expand1x1");
			targetLayer[1] = net->layer_by_name("fire6/expand1x1_p");
                }
                else if (sourceLayer.name() == "fire6/expand3x3")
		{
			targetLayer[0] = net->layer_by_name("fire6/expand3x3");
			targetLayer[1] = net->layer_by_name("fire6/expand3x3_p");
                }
                else if (sourceLayer.name() == "fire7/squeeze1x1")
		{
			targetLayer[0] = net->layer_by_name("fire7/squeeze1x1");
			targetLayer[1] = net->layer_by_name("fire7/squeeze1x1_p");
		}
                else if (sourceLayer.name() == "fire7/expand1x1")
		{
			targetLayer[0] = net->layer_by_name("fire7/expand1x1");
			targetLayer[1] = net->layer_by_name("fire7/expand1x1_p");
                }
                else if (sourceLayer.name() == "fire7/expand3x3")
		{
			targetLayer[0] = net->layer_by_name("fire7/expand3x3");
			targetLayer[1] = net->layer_by_name("fire7/expand3x3_p");
                }
                else if (sourceLayer.name() == "fire8/squeeze1x1")
		{
			targetLayer[0] = net->layer_by_name("fire8/squeeze1x1");
			targetLayer[1] = net->layer_by_name("fire8/squeeze1x1_p");
		}
                else if (sourceLayer.name() == "fire8/expand1x1")
		{
			targetLayer[0] = net->layer_by_name("fire8/expand1x1");
			targetLayer[1] = net->layer_by_name("fire8/expand1x1_p");
                }
                else if (sourceLayer.name() == "fire8/expand3x3")
		{
			targetLayer[0] = net->layer_by_name("fire8/expand3x3");
			targetLayer[1] = net->layer_by_name("fire8/expand3x3_p");
                }
                else if (sourceLayer.name() == "fire9/squeeze1x1")
		{
			targetLayer[0] = net->layer_by_name("fire9/squeeze1x1");
			targetLayer[1] = net->layer_by_name("fire9/squeeze1x1_p");
		}
                else if (sourceLayer.name() == "fire9/expand1x1")
		{
			targetLayer[0] = net->layer_by_name("fire9/expand1x1");
			targetLayer[1] = net->layer_by_name("fire9/expand1x1_p");
                }
                else if (sourceLayer.name() == "fire9/expand3x3")
		{
			targetLayer[0] = net->layer_by_name("fire9/expand3x3");
			targetLayer[1] = net->layer_by_name("fire9/expand3x3_p");
                }
		//LinF
                //else if (sourceLayer.name() == "conv10")
		//{
		//	targetLayer[0] = net->layer_by_name("conv10");
		//	targetLayer[1] = net->layer_by_name("conv10_p");
                //}
		//LinF
		else
			continue;

		if (sourceLayer.blobs_size() != 2) continue;

		if (targetLayer[0]->blobs().size() != 2 ||
			targetLayer[1]->blobs().size() != 2)
		{
			cout << "Target Blobs are not Conv";
			getchar();
			break;
		}

		for (int j = 0; j < targetLayer[0]->blobs().size(); ++j) {
			if (!targetLayer[0]->blobs()[j]->ShapeEquals(sourceLayer.blobs(j))) {
				Blob<float > source_blob;
				const bool kReshape = true;
				source_blob.FromProto(sourceLayer.blobs(j), kReshape);
			}
			const bool kReshape = false;
			targetLayer[0]->blobs()[j]->FromProto(sourceLayer.blobs(j), kReshape);
		}
		for (int j = 0; j < sourceLayer.blobs_size(); j++)
		{
			//LF
			const bool kReshape = false;
			targetLayer[0]->blobs()[j]->FromProto(sourceLayer.blobs(j), kReshape);
			targetLayer[1]->blobs()[j]->FromProto(sourceLayer.blobs(j), kReshape);
		}

		cout << " replaced by: " << sourceLayer.name() << endl;
		cout << " replaced by: " << sourceLayer.name() << endl;
		//LF
	}

	solver->Solve();
}

void testNet(string modelPath)
{
	Caffe::set_mode(Caffe::GPU);
	Net<float> net("goturnDeploy.prototxt", TEST);
	net.CopyTrainedLayersFrom(modelPath);

	boost::shared_ptr<Blob <float>> data1Layer;
	boost::shared_ptr<Blob <float> > data2Layer;
	boost::shared_ptr<Blob <float> > labelLayer;
	boost::shared_ptr<Blob <float> > outputLayer;

	outputLayer = net.blob_by_name("out");
	data1Layer = net.blob_by_name("data1");
	data2Layer = net.blob_by_name("data2");
	labelLayer = net.blob_by_name("label");

	net.Forward();

	float *out = outputLayer->mutable_cpu_data();
	float *data1 = data1Layer->mutable_cpu_data();
	float *data2 = data2Layer->mutable_cpu_data();
	float *label = labelLayer->mutable_cpu_data();

	for (int k = 0; k < 100; k++)
	{
		//Print GTBB
		cout << "GTBB x,y: " << label[k * 4 + 0] << " " << label[k * 4 + 1] << " " << label[k * 4 + 2] << " " << label[k * 4 + 3] << endl;
		//Print GOTURN_BB
		cout << "estimate x,y: " << out[k * 4 + 0] << " " << out[k * 4 + 1] << " " << out[k * 4 + 2] << " " << out[k * 4 + 3] << endl;

		//Make GTBB and GOTURN_BB
		Rect2f gtbb, res_bb;
		gtbb.x = label[k * 4 + 0];
		gtbb.y = label[k * 4 + 1];
		gtbb.width = label[k * 4 + 2] - label[k * 4 + 0];
		gtbb.height = label[k * 4 + 3] - label[k * 4 + 1];

		res_bb.x = out[k * 4 + 0];
		res_bb.y = out[k * 4 + 1];
		res_bb.width = out[k * 4 + 2] - out[k * 4 + 0];
		res_bb.height = out[k * 4 + 3] - out[k * 4 + 1];

		//Construct Target/Search patches from data1/data2
		vector <Mat> channelsTargetPatch;
		vector <Mat> channelsSearchPatch;
		Mat targetPatch;
		Mat searchPatch;

		for (int i = 0; i < 3; i++)
		{
			Mat channelTarget(227, 227, CV_32FC1, data1 + i * 227 * 227 + k * 3 * 227 * 227);
			Mat channelSearch(227, 227, CV_32FC1, data2 + i * 227 * 227 + k * 3 * 227 * 227);
			channelsTargetPatch.push_back(channelTarget);
			channelsSearchPatch.push_back(channelSearch);
		}
		//RGB -> BGR and Merge
		//reverse(channelsTargetPatch.begin(), channelsTargetPatch.end());
		//reverse(channelsSearchPatch.begin(), channelsSearchPatch.end());

		merge(channelsTargetPatch, targetPatch);
		merge(channelsSearchPatch, searchPatch);

		//Add mean
		targetPatch = targetPatch + 128;
		searchPatch = searchPatch + 128;

		targetPatch.convertTo(targetPatch, CV_8U);
		searchPatch.convertTo(searchPatch, CV_8U);

		//Draw GT/GOTURN bounding boxes and show patches
		rectangle(searchPatch, gtbb, Scalar(0, 255, 0));
		rectangle(searchPatch, res_bb, Scalar(0, 0, 255));
		imshow("Target", targetPatch);
		imshow("Search", searchPatch);
		waitKey();
	}
}

void buildDB()
{
	//Generate training datasets
	for (int i = 1; i <= 10; i++)
	{
		string fileName = "/opt/imagedata++/trainDataset_" + to_string(i) + ".h5";
		buildH5Datasets(fileName, 500);
	}
}
