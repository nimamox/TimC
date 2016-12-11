//
//  Network.cpp
//  TimC
//
//  Created by Nima Mohammadi
//  Copyright © 2016 Nima Mohammadi. All rights reserved.
//

#include "Network.hpp"
#include <iostream>
#include <stdlib.h>
#include <fstream>

#if defined(_WIN32)
#include <direct.h>
#define spl "\\"
#define splc '\\'
#else
#include <sys/stat.h>
#define spl "/"
#define splc '/'
#endif


//#define EIGEN_INITIALIZE_MATRICES_BY_NAN

Network::Network(){
    
}

Network::Network(std::vector<float> scales, std::vector<float> orientations, int gaborRFSize, float gaborDiv, int gaborCRF, int c1InhibRFsize, float c1InhibPercentClose, float c1InhibPercentFar){
    Scales = scales;
    Orientations = orientations;
    GaborRFSize = gaborRFSize;
    GaborDiv = gaborDiv;
    GaborCRF = gaborCRF;
    C1InhibRFsize = c1InhibRFsize;
    C1InhibPercentClose = c1InhibPercentClose;
    C1InhibPercentFar = c1InhibPercentFar;
}

std::shared_ptr<GaborLayer> Network::ApplyGabor(std::string imageFolder, std::vector<float> scales, std::vector<float> orientations, int rfSize, float div, int cRF){
    std::string serializedPath = imageFolder;
    std::replace(serializedPath.begin(), serializedPath.end(), splc, '_');
    serializedPath = "GaborLayer_" + serializedPath + "-" + std::to_string(rfSize) + "-" + std::to_string(div) + "-" + std::to_string(cRF) + ".bin";
    
    std::shared_ptr<GaborLayer> gabor;
    
    std::ifstream input_stream(serializedPath, std::ios::binary);
    if (input_stream.good()){
        std::cout << "Loading cached gabored spike times..." << std::endl;
        cereal::BinaryInputArchive archive(input_stream);
        gabor = std::make_shared<GaborLayer>();
        archive(*gabor);
    } else {
        gabor = std::make_shared<GaborLayer>(imageFolder, scales, orientations, rfSize, div, cRF);
        std::ofstream output_stream(serializedPath, std::ios::binary);
        cereal::BinaryOutputArchive archive(output_stream);
        archive(*gabor);
    }
    
    return gabor;
    
}

SpikeVec1D Network::dummy(){
    SpikeVec1D q;
//    q.push_back(SpikeVec1D());
    std::shared_ptr<SpikeData> w = std::make_shared<SpikeData>(12.34, 5, 6, 7, 8);
    std::shared_ptr<SpikeData> e = std::make_shared<SpikeData>(34.12, 8, 9, 10, 11);
    q.push_back(w);
    q.push_back(e);
    return q;
}

std::vector<std::vector<std::vector<float>>> Network::TestNetworkSpikeTiming(std::string imageFolder){
    Eigen::MatrixXi spikeCounts;
    std::cout << "Preparing network for test..." << std::endl;
    //applying gabor
    std::cout << "Applying gabor..." << std::endl;
    gaborLayer = ApplyGabor(imageFolder, Scales, Orientations, GaborRFSize, GaborDiv, GaborCRF);
    OrderedSpikes = gaborLayer->OrderedData;
    Spikes5D = gaborLayer->SpikeData5D;
    std::cout << " Done." << std::endl;
    
    //applying previous kernels
    std::cout << "Applying previous kernels..." << std::endl;
    for (int ker = 0; ker < Kernels.size(); ker++){
        std::cout << Kernels[ker].Name  << "..." << std::endl;
        SpikeVec2D tempOrderedSpikes;
        SpikeVec5D tempSpikes5D;
        for (int i = 0; i < OrderedSpikes.size(); i++){
            SpikeVec4D temp;
            tempOrderedSpikes.push_back(Kernels[ker].TestKernel(OrderedSpikes[i], Spikes5D[i], temp, (ker != (Kernels.size() - 1))));
            tempSpikes5D.push_back(temp);
        }
        OrderedSpikes = tempOrderedSpikes;
        Spikes5D = tempSpikes5D;
        
        std::cout << " Done." << std::endl;
    }
    
    Kernel lastKernel = Kernels[Kernels.size()-1];
    
    std::vector<std::vector<std::vector<float>>> spikeTiming;
    for (int img = 0; img < Spikes5D.size(); img++){
        spikeTiming.push_back(std::vector<std::vector<float>>());
        spikeTiming[img].resize(lastKernel.NumberOfFeature);
        for (int scl = 0; scl < Spikes5D[img].size(); scl++)
            for (int r = 0; r < Spikes5D[img][scl][0].size(); r++)
                for (int c = 0; c < Spikes5D[img][scl][0][0].size(); c++){
                    float mini_time = 0;
                    int feature = -1;
                    for (int f = 0; f < Spikes5D[img][scl].size(); f++)
                        if (Spikes5D[img][scl][f][r][c])
                            if (!mini_time || Spikes5D[img][scl][f][r][c]->Time < mini_time){
                                mini_time = Spikes5D[img][scl][f][r][c]->Time;
                                feature = f;
                            }
                    if (mini_time)
                        spikeTiming[img][feature].push_back(mini_time);
                }
        for (int f=0; f<lastKernel.NumberOfFeature; f++) {
            std::sort(spikeTiming[img][f].begin(), spikeTiming[img][f].end());
        }
    }
    return spikeTiming;
}


Eigen::MatrixXi Network::TestNetwork(std::string imageFolder){
    Eigen::MatrixXi spikeCounts;
    std::cout << "Preparing network for test..." << std::endl;
    //applying gabor
    std::cout << "Applying gabor..." << std::endl;
	gaborLayer = ApplyGabor(imageFolder, Scales, Orientations, GaborRFSize, GaborDiv, GaborCRF);
    OrderedSpikes = gaborLayer->OrderedData;
    Spikes5D = gaborLayer->SpikeData5D;
    std::cout << " Done." << std::endl;
    
    //applying previous kernels
    std::cout << "Applying previous kernels..." << std::endl;
    for (int ker = 0; ker < Kernels.size(); ker++){
        std::cout << Kernels[ker].Name  << "..." << std::endl;
        SpikeVec2D tempOrderedSpikes;
        SpikeVec5D tempSpikes5D;
        for (int i = 0; i < OrderedSpikes.size(); i++){
            SpikeVec4D temp;
            tempOrderedSpikes.push_back(Kernels[ker].TestKernel(OrderedSpikes[i], Spikes5D[i], temp, (ker != (Kernels.size() - 1))));
            tempSpikes5D.push_back(temp);
        }
        OrderedSpikes = tempOrderedSpikes;
        Spikes5D = tempSpikes5D;
        
        std::cout << " Done." << std::endl;
    }
    
    std::cout << "Counting spikes..." << std::endl;
    Kernel lastKernel = Kernels[Kernels.size()-1];
    spikeCounts.resize(Spikes5D.size(), lastKernel.NumberOfFeature);
    spikeCounts.setZero();
    for (int img = 0; img < Spikes5D.size(); img++){
        for (int i = 0; i < lastKernel.NumberOfFeature; i++){
            int cnt = 0;
            for (int scl = 0; scl < Spikes5D[img].size(); scl++){
                for (int r = 0; r < Spikes5D[img][scl][i].size(); r++){
                    for(int c = 0; c < Spikes5D[img][scl][i][0].size(); c++){
                        if (Spikes5D[img][scl][i][r][c])
                            cnt++;
                    }
                }
            }
            spikeCounts(img, i) = cnt;
        }
    }
    return spikeCounts;
}

void Network::TrainNewKernel(std::string imageFolder, Kernel& kernel, int numberOfEpoch, int numberOfImageRepeat, float apLimit){
    std::cout << "Preparing network to train kernel " << kernel.Name << "..." << std::endl;
    
    //applying gabor
    std::cout << "Applying gabor..." << std::endl;
    gaborLayer = ApplyGabor(imageFolder, Scales, Orientations, GaborRFSize, GaborDiv, GaborCRF);
    OrderedSpikes = gaborLayer->OrderedData;
    Spikes5D = gaborLayer->SpikeData5D;
    std::cout << " Done." << std::endl;
    
    //applying previous kernels
    std::cout << "Applying previous kernels..." << std::endl;
    for (int ker = 0; ker < Kernels.size(); ker++){
        std::cout << Kernels[ker].Name  << "..." << std::endl;
        SpikeVec2D tempOrderedSpikes;
        SpikeVec5D tempSpikes5D;
        for (int i = 0; i < OrderedSpikes.size(); i++){
            SpikeVec4D temp;
            tempOrderedSpikes.push_back(Kernels[ker].TestKernel(OrderedSpikes[i], Spikes5D[i], temp, true));
            tempSpikes5D.push_back(temp);
        }
        OrderedSpikes = tempOrderedSpikes;
        Spikes5D = tempSpikes5D;
        
        std::cout << " Done." << std::endl;
    }
    
    //training new kernel
    std::cout << "Training the new kernel " << kernel.Name << std::endl;
    
    int cnt = 1;
    for (int i = 0; i < numberOfEpoch; i++){
        std::cout << "Epoch: " << i+1 << std::endl;
        for (int j = 0; j < OrderedSpikes.size(); j++)
            for (int k = 0; k < numberOfImageRepeat; k++, cnt++){
//                SaveKernelMergedWeights(kernel, i);
                if (kernel.Ap < apLimit)
                    if (cnt % 400 == 0){
                        kernel.Ap *= 2;
                        kernel.An = -3.0 / 4.0 * kernel.Ap;
                    }
                kernel.TrainKernel(OrderedSpikes[j], Spikes5D[j]);
            }
    }
    std::cout << "Done." << std::endl;
    
    //saving
    std::cout << "Saving kernel " << kernel.Name << std::endl;
//    kernel.SaveKernel($"{kernel.Name}.kernel");
    Kernels.push_back(kernel);
    std::cout << " Done." << std::endl;
    SaveKernelMergedWeights(kernel);
}

void Network::SaveKernelMergedWeights(Kernel ker, int epoch){
    std::string path;
    
    #if defined(_WIN32)
        _mkdir(ker.Name.c_str());
    #else
        mode_t mode = 0755;
        mkdir(ker.Name.c_str(), mode);
    #endif
    
    path = (std::string(ker.Name) + spl + ker.Name + "_SumWeightsPlot.plt");
    std::ofstream plt_outfile;
    plt_outfile.open(path, std::ofstream::out);
    
    plt_outfile << "set xrange [-0.5:" << ker.Weights[0][0].cols()-.5 <<"]; set yrange [" << ker.Weights[0][0].rows()-.5 <<":-0.5]" << std::endl;
    plt_outfile << "set size ratio 1" << std::endl;
    plt_outfile << "set cbrange [0:1]" << std::endl;
    plt_outfile << "set pm3d map" << std::endl;
    plt_outfile << "set palette gray" << std::endl;
    plt_outfile << "set terminal png" << std::endl;
    
    plt_outfile << "do for [i = 0:" << (ker.NumberOfFeature - 1) << "]{" << std::endl;
    plt_outfile << "\tt = sprintf('Weights | Kernel: " << ker.Name << ", Neuron: %d', i)" << std::endl;
    plt_outfile << "\tset title t" << std::endl;
    plt_outfile << "\toutfile = sprintf('" << ker.Name << "_SumWeights_n%d.png', i)" << std::endl;
    plt_outfile << "\tset output outfile" << std::endl;
    plt_outfile << "\tinfile = sprintf('" << ker.Name << "_SumWeights_n%d.txt', i)" << std::endl;
    plt_outfile << "\tsplot infile matrix with image" << std::endl;
    plt_outfile << "}" << std::endl;
    
    
    for (int i = 0; i < ker.Weights.size(); i++){
        if (epoch==-1)
            path = (std::string(ker.Name) + spl + ker.Name + "_SumWeights_n" + std::to_string(i) + ".txt");
        else
            path = (std::string(ker.Name) + spl + ker.Name + "_SumWeights_n" + std::to_string(i) + +"_" + std::to_string(epoch) + ".txt");
        std::cout << "Saving merged weights" << path << std::endl;
        std::ofstream outfile;
        outfile.open(path, std::ofstream::out);
        for (int j = 0; j < ker.Weights[i][0].rows(); j++){
            for (int k = 0; k < ker.Weights[i][0].cols(); k++){
                float sum = 0;
                for (int m = 0; m < ker.Weights[i].size(); m++)
                    sum += ker.Weights[i][m](j, k);
                if (sum > 1)
                    sum = 1;
                outfile << std::to_string(sum) << " ";
            }
            outfile << std::endl;
        }
        outfile.close();
    }
}