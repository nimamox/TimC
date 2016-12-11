//
//  Network.hpp
//  TimC
//
//  Created by Nima Mohammadi
//  Copyright © 2016 Nima Mohammadi. All rights reserved.
//

#ifndef Network_hpp
#define Network_hpp

#include <stdio.h>
#include <vector>
#include <string>
#include "GaborLayer.hpp"
#include "Kernel.hpp"
#include "common.hpp"
#include <memory>

class Network{
public:
    std::vector<Kernel> Kernels;
    std::vector<float> Scales;
    std::vector<float> Orientations;
    int GaborRFSize;
    float GaborDiv;
    int GaborCRF;
    int C1InhibRFsize;
    float C1InhibPercentClose;
    float C1InhibPercentFar;
    std::shared_ptr<GaborLayer> gaborLayer;
    Network();
    Network(std::vector<float>, std::vector<float>, int, float, int, int=5, float=.15, float=.05);
    void TrainNewKernel(std::string, Kernel&, int, int, float);
    Eigen::MatrixXi TestNetwork(std::string);
    std::vector<std::vector<std::vector<float>>> TestNetworkSpikeTiming(std::string);
    SpikeVec1D dummy();
    static std::shared_ptr<GaborLayer> ApplyGabor(std::string, std::vector<float>, std::vector<float>, int, float, int);
    static void SaveKernelMergedWeights(Kernel, int=-1);
private:
    SpikeVec2D OrderedSpikes;
    SpikeVec5D Spikes5D;
};

#endif /* Network_hpp */
