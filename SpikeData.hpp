//
//  SpikeData.hpp
//  TimC
//
//  Created by Nima Mohammadi
//  Copyright © 2016 Nima Mohammadi. All rights reserved.
//

#ifndef SpikeData_hpp
#define SpikeData_hpp

#include <stdio.h>

//#include <cereal/types/unordered_map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>
#include <fstream>

class SpikeData{
public:
    float Time;
    int Row;
    int Column;
    int Feature;
    int Scale;
    SpikeData();
    SpikeData(float, int, int, int, int);
    
    template <class Archive>
    void serialize( Archive & ar )
    {
        ar(Time, Row, Column, Feature, Scale);
    }
};

#endif /* SpikeData_hpp */
