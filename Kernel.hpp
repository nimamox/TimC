//
//  Kernel.hpp
//  TimC
//
//  Created by Nima Mohammadi
//  Copyright © 2016 Nima Mohammadi. All rights reserved.
//

#ifndef Kernel_hpp
#define Kernel_hpp

#include <stdio.h>
#include <string>
#include <vector>
#include "GaborLayer.hpp"
#include "Eigen/Dense"
#include "SpikeData.hpp"
#include "common.hpp"
//#include "Eigen/StdVector"

class Kernel{
public:
    std::string Name;
    int NumberOfFeature;
    std::vector<std::vector<Eigen::MatrixXf>> Weights;
    int CRF;
    int SRF;
    int centerRow;
    int centerCol;
    int CStride;
    float Threshold;
    std::vector<float> InhibitionsPercent;
    int KWTA;
    float Ap;
    float An;
    Kernel();
    Kernel(std::string, int, int, int, int, int, std::vector<float>, float, int, float, float, float, float);
    void TrainKernel(SpikeVec1D&, SpikeVec4D&);
    SpikeVec1D TestKernel(SpikeVec1D&, SpikeVec4D&, SpikeVec4D&, bool);
    
    template <class Archive>
    void save(Archive & ar) const
    {
        ar(Name, NumberOfFeature, Weights, CRF, SRF, centerRow, centerCol, CStride, Threshold, InhibitionsPercent, KWTA, Ap, An, sOffset, cOffset, iOffset);
    }
    
    template <class Archive>
    void load( Archive & ar )
    {
        ar(Name, NumberOfFeature, Weights, CRF, SRF, centerRow, centerCol, CStride, Threshold, InhibitionsPercent, KWTA, Ap, An, sOffset, cOffset, iOffset);
    }
    
private:
    int sOffset;
    int cOffset;
    int iOffset;
    SpikeVec4D GetPooledInhibitedTimes(const SpikeVec4D&);
    float getWeight(int, int, int, int);
    void ApplySTDP(int, int, int, SpikeVec3D&, std::shared_ptr<SpikeData>);
    void increaseWeight(int, int, int, int);
    void decreaseWeight(int, int, int, int);
    
};

#endif /* Kernel_hpp */
