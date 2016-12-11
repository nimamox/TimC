//
//  GaborLayer.hpp
//  TimC
//
//  Created by Nima Mohammadi
//  Copyright © 2016 Nima Mohammadi. All rights reserved.
//

#ifndef GaborLayer_hpp
#define GaborLayer_hpp

#include <stdio.h>
#include <string>
#include <vector>
#include "SpikeData.hpp"
#include "Gabor.hpp"
#include "common.hpp"
#include <memory>

#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

class GaborLayer{
public:
    int ImageIdx;
    std::vector<std::string> FileNames;
    int NumberOfScales;
    SpikeVec2D OrderedData;
    SpikeVec5D SpikeData5D;
    std::shared_ptr<Gabor> gabor;
    Eigen::VectorXd inhibitionPercents;
    GaborLayer();
    GaborLayer(std::string, std::vector<float>, std::vector<float>, int, int, int, int=5, float=.15, float=.05);
    
    template <class Archive>
    void save(Archive & ar) const
    {
        ar(ImageIdx, NumberOfScales, FileNames, OrderedData, SpikeData5D, inhibitionPercents);
    }
    
    template <class Archive>
    void load(Archive & ar)
    {
        ar(ImageIdx, NumberOfScales, FileNames, OrderedData, SpikeData5D, inhibitionPercents);
    }
    
private:
    SpikeVec1D GetGaboredTimes(std::string, SpikeVec4D&, int);
    SpikeVec4D GetPooledInhibitedTimes(int, const SpikeVec4D&);
};

#endif /* GaborLayer_hpp */
