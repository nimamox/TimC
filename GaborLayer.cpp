//
//  GaborLayer.cpp
//  TimC
//
//  Created by Nima Mohammadi
//  Copyright © 2016 Nima Mohammadi. All rights reserved.
//

#include "GaborLayer.hpp"
#include "tinydir/tinydir.h"
#include <iostream>
#include <algorithm>
#include <math.h>

using namespace std;

GaborLayer::GaborLayer(){
    
}

GaborLayer::GaborLayer(std::string folderAddress, std::vector<float> scales, std::vector<float> orientations, int rfSize, int div, int cRF, int C1InhibRFsize, float C1InhibPercentClose, float C1InhibPercentFar){
    ImageIdx = 0;
    NumberOfScales = (int)scales.size();
    inhibitionPercents = Eigen::VectorXd::LinSpaced(C1InhibRFsize, C1InhibPercentClose, C1InhibPercentFar);
    tinydir_dir dir;
    tinydir_open(&dir, folderAddress.c_str());
    std::string fname;
    while (dir.has_next){
        tinydir_file file;
        tinydir_readfile(&dir, &file);
        fname = file.name;
        //std::cout<<fname<<std::endl;
        if (!file.is_dir && fname.length() > 4)
            if (fname.compare(fname.length()-4, 4, ".png")==0)
                FileNames.push_back(folderAddress + "/" + fname);
        tinydir_next(&dir);
    }
    this->gabor = std::make_shared<Gabor>(scales, orientations, rfSize, div);
    
    while(ImageIdx < FileNames.size()){
        std::cout << ImageIdx << ": " << FileNames[ImageIdx] << std::endl;
        SpikeVec4D temp;
        OrderedData.push_back(GetGaboredTimes(FileNames[ImageIdx], temp, cRF));
        SpikeData5D.push_back(temp);
        ImageIdx++;
    }
}

SpikeVec1D GaborLayer::GetGaboredTimes(std::string imageAddress, SpikeVec4D& spike4DOriPooled, int complexField){
    SpikeVec1D result;
    SpikeVec4D spike4D = gabor->GetGaboredTimes(imageAddress);
    spike4DOriPooled = GetPooledInhibitedTimes(complexField, spike4D);
    
    for (int scl = 0; scl < NumberOfScales; scl++)
        for (int r = 0; r < spike4DOriPooled[scl][0].size(); r++)
            for (int c = 0; c < spike4DOriPooled[scl][0][0].size(); c++){
				std::shared_ptr<SpikeData> mini;
                for (int ori = 0; ori < spike4DOriPooled[scl].size(); ori++){
                    if (spike4DOriPooled[scl][ori][r][c]){
                        if (!mini || spike4DOriPooled[scl][ori][r][c]->Time < mini->Time){
                            mini = std::make_shared<SpikeData>(
                                                 spike4DOriPooled[scl][ori][r][c]->Time,
                                                 r, c,
                                                 spike4DOriPooled[scl][ori][r][c]->Feature,
                                                 spike4DOriPooled[scl][ori][r][c]->Scale);
                        }
                    }
                }
                if (mini)
                    result.push_back(mini);
            }
    std::sort(result.begin(), result.end(), [](std::shared_ptr<SpikeData> const &a, std::shared_ptr<SpikeData> const &b){
        return a->Time < b->Time;
    });
    return result;
    
}

SpikeVec4D GaborLayer::GetPooledInhibitedTimes(int complexField, const SpikeVec4D& spike4D){
    std::vector<std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>>> inhibitions;
    
    SpikeVec4D spike4DOriPooled;
    for (int scl = 0; scl < NumberOfScales; scl++){
        spike4DOriPooled.push_back(SpikeVec3D());
        for (int ori = 0; ori < spike4D[scl].size(); ori++){
            spike4DOriPooled[scl].push_back(SpikeVec2D());
            for (int t1 = 0; t1 < ceil((float)spike4D[scl][ori].size() / (complexField - 1)); t1++){
                spike4DOriPooled[scl][ori].push_back(SpikeVec1D());
                for (int t2 = 0; t2 < ceil((float)spike4D[scl][ori][0].size() / (complexField - 1)); t2++)
                    spike4DOriPooled[scl][ori][t1].push_back(std::shared_ptr<SpikeData>());
            }
            for (int r = 0; r < spike4DOriPooled[scl][ori].size(); r++)
                for (int c = 0; c < spike4DOriPooled[scl][ori][0].size(); c++){
                    int minCol = c * (complexField - 1);
                    int minRow = r * (complexField - 1);
                    
                    int maxCol = std::min(minCol + complexField, (int)spike4D[scl][ori][0].size());
                    int maxRow = std::min(minRow + complexField, (int)spike4D[scl][ori].size());
                    
                    std::shared_ptr<SpikeData> mini;
                    
                    for (int rr = minRow; rr < maxRow; rr++)
                        for (int cc = minCol; cc < maxCol; cc++)
                            if (spike4D[scl][ori][rr][cc])
                                if (!mini || spike4D[scl][ori][rr][cc]->Time < mini->Time){
                                    mini = std::make_shared<SpikeData>(
                                                         spike4D[scl][ori][rr][cc]->Time,
                                                         r, c,
                                                         spike4D[scl][ori][rr][cc]->Feature,
                                                         spike4D[scl][ori][rr][cc]->Scale);
                                }
                    spike4DOriPooled[scl][ori][r][c] = mini;
                }
        }
    }
    
    int inhibOffset = (int)inhibitionPercents.size();
    
    //apply lateral inhibition
    //compute percentage
    for (int scl = 0; scl < NumberOfScales; scl++){
        inhibitions.push_back(std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>>());
        for (int ori = 0; ori < spike4DOriPooled[scl].size(); ori++){
            inhibitions[scl].push_back(Eigen::MatrixXf(spike4DOriPooled[scl][ori].size(), spike4DOriPooled[scl][ori][0].size()));
            inhibitions[scl][ori].setZero();
            for (int row = 0; row < spike4DOriPooled[scl][ori].size(); row++){
                int minRow = max(0, row - inhibOffset);
                int maxRow = min(row + inhibOffset + 1, (int)spike4DOriPooled[scl][ori].size());
                
                for (int col = 0; col < spike4DOriPooled[scl][ori][0].size(); col++){
                    if (spike4DOriPooled[scl][ori][row][col]){
                        int minCol = std::max(0, col - inhibOffset);
                        int maxCol = std::min(col + inhibOffset + 1, (int)spike4DOriPooled[scl][ori][0].size());
                        
                        for (int r = minRow; r < maxRow; r++)
                            for (int c = minCol; c < maxCol; c++)
                                if (spike4DOriPooled[scl][ori][r][c]){
                                    int dist = std::max(std::abs(row - r), std::abs(col - c));
                                    if (dist)
                                        if (spike4DOriPooled[scl][ori][r][c]->Time > spike4DOriPooled[scl][ori][row][col]->Time)
                                        inhibitions[scl][ori](r, c) += inhibitionPercents(dist-1);
                                }
                    }
                }
            }
        }
    }
    
    //apply inhibition
    for (int scl = 0; scl < NumberOfScales; scl++)
        for (int ori = 0; ori < spike4DOriPooled[scl].size(); ori++)
            for (int row = 0; row < spike4DOriPooled[scl][ori].size(); row++)
                for (int col = 0; col < spike4DOriPooled[scl][ori][0].size(); col++)
                    if (spike4DOriPooled[scl][ori][row][col])
                        spike4DOriPooled[scl][ori][row][col]->Time += spike4DOriPooled[scl][ori][row][col]->Time * inhibitions[scl][ori](row, col);
    
    return spike4DOriPooled;
}