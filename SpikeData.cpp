//
//  SpikeData.cpp
//  TimC
//
//  Created by Nima Mohammadi
//  Copyright © 2016 Nima Mohammadi. All rights reserved.
//

#include "SpikeData.hpp"

SpikeData::SpikeData(){
    
}

SpikeData::SpikeData(float time, int row, int col, int feature, int scale){
    this->Time = time;
    this->Row = row;
    this->Column = col;
    this->Feature = feature;
    this->Scale = scale;
}