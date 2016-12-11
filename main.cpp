
//
//  main.cpp
//  TimC
//
//  Created by Nima Mohammadi
//  Copyright © 2016 Nima Mohammadi. All rights reserved.
//

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "Network.hpp"
#include "common.hpp"
#include "Gabor.hpp"
#include "INIReader/INIReader.h"
#include <sys/types.h>
#include <sys/stat.h>

#include "Eigen/Dense"

int main(int argc, const char * argv[]) {
    
//    Eigen::VectorXd v = Eigen::VectorXd::LinSpaced(5, 0.15, 0.05);
//    v.setLinSpaced(5, 0.15, 0.05);
//    std::cout << v;
//    exit(0);
    INIReader reader("config.ini");
    
    if (reader.ParseError() < 0) {
        std::cout << "Can't load 'config.ini'\n";
        return 1;
    }
    std::string trainFolder = reader.Get("General", "TrainFolder", "");
    std::string testFolder = reader.Get("General", "TestFolder", "");
    struct stat info;
    if(stat(trainFolder.c_str(), &info)){
        std::cout << "No such train folder!" << std::endl;
        exit(1);
    }
    if (!bool(info.st_mode & S_IFDIR)){
        std::cout << "Invalid train folder!" << std::endl;
        exit(1);
    }
    if (!testFolder.empty()){
        if(stat(testFolder.c_str(), &info)){
            std::cout << "No such test folder!" << std::endl;
            exit(1);
        }
        if (!bool(info.st_mode & S_IFDIR)){
            std::cout << "Invalid test folder!" << std::endl;
            exit(1);
        }
    }
    int gaborRF = (int)reader.GetInteger("General", "GaborRF", 5);
    int gaborCRF = (int)reader.GetInteger("General", "GaborCRF", 7);
    int gaborDiv = (int)reader.GetInteger("General", "GaborDiv", 4);
    int C1InhibRFsize = (int)reader.GetInteger("General", "C1InhibRFsize", 5);
    float C1InhibPercentClose = (float)reader.GetReal("General", "C1InhibPercentClose", 0.15);
    float C1InhibPercentFar = (float)reader.GetReal("General", "C1InhibPercentFar", 0.05);
    auto scales_str = std::stringstream(reader.Get("General", "Scales", "1 0.7 0.5 0.36 0.25"));
    auto orientations_str = std::stringstream(reader.Get("General", "Orientations", "22.5 67.5 112.5 157.5"));
    auto layers_str = std::stringstream(reader.Get("General", "Layers", "S2"));
    std::vector<float> scales;
    std::vector<float> orientations;
    float t;
    while (scales_str >> t){
        scales.push_back(t);
    }
    while (orientations_str >> t){
        orientations.push_back(t);
    }
    std::cout << "-------------------------" << std::endl;
    std::cout << "NET PARAMS:" << std::endl;
    std::cout << "GaborRF: " << gaborRF << ", GaborCRF: " << gaborCRF << ", GaborDiv: " << gaborDiv << std::endl;
    std::cout << "Scales: ";
    for (auto i: scales)
        std::cout << i << " ";
    std::cout << std::endl;
    std::cout << "Orientations: ";
    for (auto i: orientations)
        std::cout << i << " ";
    std::cout << std::endl;
    std::cout << "C1 Lateral Inhibitions: " << Eigen::VectorXd::LinSpaced(C1InhibRFsize, C1InhibPercentClose, C1InhibPercentFar).transpose() << std::endl;
    
    std::cout << "-------------------------" << std::endl;
    
    Network network(scales, orientations, gaborRF, gaborDiv, gaborCRF, C1InhibRFsize, C1InhibPercentClose, C1InhibPercentFar);
    std::string layer;
    while (layers_str >> layer){
        std::cout << "~~> Layer " << layer << std::endl;
        float threshold = reader.GetReal(layer, "Threshold", 64.0);
        float meanWeight = reader.GetReal(layer, "meanWeight", 0.8f);
        float stdDevWeight = reader.GetReal(layer, "stdDevWeight", 0.05f);
        float numberOfPreKernelFeatures = reader.GetInteger(layer, "NumberOfPreKernelFeatures", 4);
        int srf = (int)reader.GetInteger(layer, "SRF", 17);
        int crf = (int)reader.GetInteger(layer, "CRF", 3);
        int cstride = (int)reader.GetInteger(layer, "CSTRIDE", 2);
        int kwta = (int)reader.GetInteger(layer, "kWTA", 2);
        int epochs = (int)reader.GetInteger(layer, "Epochs", 30);
        int presentationPerEpoch = (int)reader.GetInteger(layer, "PresentationPerEpoch", 1);
        int numberOfFeatures = (int)reader.GetInteger(layer, "NumberOfFeatures", 10);
        float ap = reader.GetReal(layer, "Ap", 1.0f / 64.0);
        float an = reader.GetReal(layer, "An", -3.0 / 4.0 * ap);
        float apLimit = reader.GetReal(layer, "ApLimit", 0.25);
        auto inhibitionPercents_str = std::stringstream(reader.Get("General", "InhibitionPercents", "0.15 0.12"));
        std::vector<float> inhibitionPercents;
        while (inhibitionPercents_str >> t){
            inhibitionPercents.push_back(t);
        }
        std::cout << "-------------------------" << std::endl;
        std::cout << layer << " Params:" << std::endl;
        std::cout << "Epochs: " << epochs << " SRF: " << srf << " CRF: " << crf << " CSTRIDE: " << cstride << " KWTA:" << kwta << std::endl;
        std::cout << "Threshold: " << threshold << " meanWeight: " << meanWeight << " stdDevWeight: " << stdDevWeight << std::endl;
        std::cout << "Ap: " << ap << " An: " << an << " ApLimit: " << apLimit << std::endl;
        std::cout << "inhibitionPercents: ";
        for (auto i: inhibitionPercents)
            std::cout << i << " ";
        std::cout << std::endl;
        std::cout << "-------------------------" << std::endl;
        Kernel ker(layer, numberOfFeatures, numberOfPreKernelFeatures, srf, crf, cstride, inhibitionPercents, threshold, kwta, ap, an, meanWeight, stdDevWeight);
        network.TrainNewKernel(trainFolder, ker, epochs, presentationPerEpoch, apLimit);
    }
    
    
//    std::ifstream input_stream(serializedPath, std::ios::binary);
//    if (input_stream.good()){
//        std::cout << "Loading cached gabored spike times..." << std::endl;
//        cereal::BinaryInputArchive archive(input_stream);
//        gabor = std::make_shared<GaborLayer>();
//        archive(*gabor);
//    } else {
//        gabor = std::make_shared<GaborLayer>(imageFolder, scales, orientations, rfSize, div, cRF);
//        std::ofstream output_stream(serializedPath, std::ios::binary);
//        cereal::BinaryOutputArchive archive(output_stream);
//        archive(*gabor);
//    }
    
    if (!testFolder.empty()){
        Eigen::MatrixXi res = network.TestNetwork(testFolder);
        std::cout << res << std::endl;
    }
    else{
        std::cout << "Omit test!" << std::endl;
    }
    
    std::cout << "FINI" << std::endl;
    return 0;
}

