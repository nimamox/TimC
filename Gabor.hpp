//
//  Gabor.hpp
//  TimC
//
//  Created by Nima Mohammadi
//  Copyright © 2016 Nima Mohammadi. All rights reserved.
//

#ifndef Gabor_hpp
#define Gabor_hpp

#include <stdio.h>
#include <vector>
#include <string>
#include "SpikeData.hpp"
#include <math.h>
#include "Eigen/Dense"
#include "common.hpp"

class Gabor{
public:
    std::vector<float> Scales;
    std::vector<float> Orientations;
    int RFSize;
    float Div;
    std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> gaborFilters;
    
    Gabor(std::vector<float>, std::vector<float>, int, float);
    SpikeVec4D GetGaboredTimes(std::string);
    
private:
    void GenerateGaborFilters();
};

#endif /* Gabor_hpp */
