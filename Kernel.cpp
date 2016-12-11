//
//  Kernel.cpp
//  TimC
//
//  Created by Nima Mohammadi
//  Copyright © 2016 Nima Mohammadi. All rights reserved.
//

#include "Kernel.hpp"
#include <math.h>
#include <iostream>
#include <random>

inline float uniform_rand(float a, float b)
{
    static std::default_random_engine generator;
    static std::uniform_real_distribution<float> distribution(a, b);
    return distribution(generator);
}

Kernel::Kernel(){
    
}

Kernel::Kernel(std::string name, int numberOfFeatures, int numberOfPreKernelFeatures, int srf, int crf, int cStride, std::vector<float> inhibitionsPercent, float threshold, int kwta, float ap, float an, float meanWeight, float stdDevWeight){
    Name = name;
    NumberOfFeature = numberOfFeatures;
    SRF = srf;
    CRF = crf;
    CStride = cStride;
    InhibitionsPercent = inhibitionsPercent;
    sOffset = SRF / 2;
    cOffset = CRF / 2;
    centerRow = SRF / 2;
    centerCol = SRF / 2;
    iOffset = (int)InhibitionsPercent.size();
    Threshold = threshold;
    KWTA = kwta;
    Ap = ap;
    An = an;
    
    for (int i = 0; i < NumberOfFeature; i++){
        Weights.push_back(std::vector<Eigen::MatrixXf>());
        for (int p = 0; p < numberOfPreKernelFeatures; p++){
            Weights[i].push_back(Eigen::MatrixXf(SRF, SRF));
            Weights[i][p].resize(SRF, SRF);
            for (int r = 0; r < SRF; r++)
                for (int c = 0; c < SRF; c++){
                    float u1 = uniform_rand(0, 1);
                    float u2 = uniform_rand(0, 1);
                    float randStdNormal = std::sqrt(-2.0 * std::log(u1)) * std::sin(2.0 * PI * u2);
                    float randNormal = meanWeight + stdDevWeight * randStdNormal;
                    Weights[i][p](r, c) = randNormal;
                }
        }
    }
}

void Kernel::TrainKernel(SpikeVec1D& spikes, SpikeVec4D& spikes4D){
    std::vector<std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>>> potentials;
    std::vector<bool> hasFired;
    std::vector<int> scaleFireCounts;
    
    for (int i=0; i<NumberOfFeature; i++)
        hasFired.push_back(false);
    
    for (int i=0; i<spikes4D.size(); i++)
        scaleFireCounts.push_back(0);
    
    //scales
    for (int i = 0; i < spikes4D.size(); i++){
        int r = (int)spikes4D[i][0].size();
        int c = (int)spikes4D[i][0][0].size();
        potentials.push_back(std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>>());
        for (int j = 0; j < NumberOfFeature; j++){
            potentials[i].push_back(Eigen::MatrixXf(r, c));
            potentials[i][j].setZero();
        }
    }
    
    for (int i = 0; i < spikes.size(); i++){
        std::shared_ptr<SpikeData> current = spikes[i];
        for (int j = 0; j < NumberOfFeature; j++){
            if (scaleFireCounts[current->Scale] < KWTA && !hasFired[j]){
                int minRow = std::max(0, current->Row - sOffset);
                int maxRow = std::min((int)potentials[current->Scale][j].rows(), current->Row + sOffset + 1);
                
                int minCol = std::max(0, current->Column - sOffset);
                int maxCol = std::min((int)potentials[current->Scale][j].cols(), current->Column + sOffset + 1);
                
                //updating
                for (int r = minRow; r < maxRow; r++)
                    for (int c = minCol; c < maxCol; c++)
                        potentials[current->Scale][j](r, c) += getWeight(j, (current->Row - r),
                                                                        (current->Column - c), current->Feature);
                //check fire
                for (int r = minRow; r < maxRow && !hasFired[j]; r++)
                {
                    for (int c = minCol; c < maxCol && !hasFired[j]; c++)
                    {
                        if (potentials[current->Scale][j](r, c) >= Threshold){
                            ++scaleFireCounts[current->Scale];
                            hasFired[j] = true;
                            //Apply STDP
                            ApplySTDP(j, r, c, spikes4D[current->Scale], current);
                        }
                    }
                }
            }}
    }
    

    
    
}

SpikeVec1D Kernel::TestKernel(SpikeVec1D& spikes, SpikeVec4D& spikes4DIn, SpikeVec4D& spikes4DPooledOut, bool applyPooling){
    std::vector<std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>>> potentials;
    SpikeVec4D spikes4D;
    
    //initializing
    for (int i = 0; i < spikes4DIn.size(); i++){
        int r = (int)spikes4DIn[i][0].size();
        int c = (int)spikes4DIn[i][0][0].size();
        potentials.push_back(std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>>());
        spikes4D.push_back(SpikeVec3D());
        for (int j = 0; j < NumberOfFeature; j++){
            potentials[i].push_back(Eigen::MatrixXf(r, c));
            potentials[i][j].setZero();
            spikes4D[i].push_back(SpikeVec2D());
            for (int rr=0; rr < r; rr++){
                spikes4D[i][j].push_back(SpikeVec1D());
                for (int cc=0; cc < c; cc++)
                    spikes4D[i][j][rr].push_back(std::shared_ptr<SpikeData>());
            }
        }
    }
    
    //S firings
    for (int i = 0; i < spikes.size(); i++){
        std::shared_ptr<SpikeData> current = spikes[i];
        for (int j = 0; j < NumberOfFeature; j++){
            int minRow = std::max(0, current->Row - sOffset);
            int maxRow = std::min((int)potentials[current->Scale][j].rows(), current->Row + sOffset + 1);
            
            int minCol = std::max(0, current->Column - sOffset);
            int maxCol = std::min((int)potentials[current->Scale][j].cols(), current->Column + sOffset + 1);
            
            //updating
            for (int r = minRow; r < maxRow; r++)
                for (int c = minCol; c < maxCol; c++)
                    potentials[current->Scale][j](r, c) += getWeight(j, (current->Row - r),
                                                                    (current->Column - c), current->Feature);
            //check fire
            for (int r = minRow; r < maxRow; r++)
                for (int c = minCol; c < maxCol; c++)
                    if (spikes4D[current->Scale][j][r][c]==nullptr && potentials[current->Scale][j](r, c) >= Threshold)
                        spikes4D[current->Scale][j][r][c] = std::make_shared<SpikeData>(current->Time, r, c, j, current->Scale);
                        
        }
    }
    if (applyPooling)
        spikes4DPooledOut = GetPooledInhibitedTimes(spikes4D);
    else
        spikes4DPooledOut = spikes4D;
    
    //flattening features
    SpikeVec1D result;
    for (int scl = 0; scl < spikes4DIn.size(); scl++)
        for (int r = 0; r < spikes4DPooledOut[scl][0].size(); r++)
            for (int c = 0; c < spikes4DPooledOut[scl][0][0].size(); c++){
                std::shared_ptr<SpikeData> mini;
                for (int f = 0; f < spikes4DPooledOut[scl].size(); f++)
                    if (spikes4DPooledOut[scl][f][r][c])
                        if (!mini || spikes4DPooledOut[scl][f][r][c]->Time < mini->Time)
                            mini = std::make_shared<SpikeData>(
                                                 spikes4DPooledOut[scl][f][r][c]->Time,
                                                 r, c,
                                                 spikes4DPooledOut[scl][f][r][c]->Feature,
                                                 spikes4DPooledOut[scl][f][r][c]->Scale);
                if (mini)
                    result.push_back(mini);
            }
    std::sort(result.begin(), result.end(), [](std::shared_ptr<SpikeData> const &a, std::shared_ptr<SpikeData> const &b){
        return a->Time < b->Time;
    });
    return result;
}

SpikeVec4D Kernel::GetPooledInhibitedTimes(const SpikeVec4D& spikes4D){
    SpikeVec4D spikes4DPooled;
    std::vector<std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>>> inhibitions;

    //pooling on each feature
    for (int scl = 0; scl < spikes4D.size(); scl++){
        spikes4DPooled.push_back(SpikeVec3D());
        for (int f = 0; f < spikes4D[scl].size(); f++){
            spikes4DPooled[scl].push_back(SpikeVec2D());
            int rr = (int)ceil((float)spikes4D[scl][f].size() / CStride);
            int cc = (int)ceil((float)spikes4D[scl][f][0].size() / CStride);
            for (int r=0; r<rr; r++){
                spikes4DPooled[scl][f].push_back(SpikeVec1D());
                for (int c=0; c<cc; c++)
                    spikes4DPooled[scl][f][r].push_back(std::shared_ptr<SpikeData>());
            }
            for (int r = 0; r < spikes4DPooled[scl][f].size(); r++)
                for (int c = 0; c < spikes4DPooled[scl][f][0].size(); c++){
                    int minCol = c * CStride;
                    int minRow = r * CStride;
                    
                    int maxCol = std::min(minCol + CRF, (int)spikes4D[scl][f][0].size());
                    int maxRow = std::min(minRow + CRF, (int)spikes4D[scl][f].size());
                    
                    std::shared_ptr<SpikeData> mini;
                    for (int rr = minRow; rr < maxRow; rr++)
                        for (int cc = minCol; cc < maxCol; cc++)
                            if (spikes4D[scl][f][rr][cc])
                                if (!mini || spikes4D[scl][f][rr][cc]->Time < mini->Time)
                                    mini = std::make_shared<SpikeData>(
                                                         spikes4D[scl][f][rr][cc]->Time,
                                                         r, c,
                                                         spikes4D[scl][f][rr][cc]->Feature,
                                                         spikes4D[scl][f][rr][cc]->Scale);
                    spikes4DPooled[scl][f][r][c] = mini;
                }
        }
    }
    
    //int inhibZone = 11;
    int inhibOffset = (int)InhibitionsPercent.size();
    
    //apply lateral inhibition
    //compute percentage
    for (int scl = 0; scl < spikes4D.size(); scl++){
        inhibitions.push_back(std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>>());
        for (int f = 0; f < spikes4DPooled[scl].size(); f++){
            inhibitions[scl].push_back(Eigen::MatrixXf(spikes4DPooled[scl][f].size(),
                                                       spikes4DPooled[scl][f][0].size()));
            inhibitions[scl][f].setZero();
            for (int row = 0; row < spikes4DPooled[scl][f].size(); row++){
                int minRow = std::max(0, row - inhibOffset);
                int maxRow = std::min(row + inhibOffset + 1, (int)spikes4DPooled[scl][f].size());
                
                for (int col = 0; col < spikes4DPooled[scl][f][0].size(); col++)
                    if (spikes4DPooled[scl][f][row][col]){
                        int minCol = std::max(0, col - inhibOffset);
                        int maxCol = std::min(col + inhibOffset + 1, (int)spikes4DPooled[scl][f][0].size());
                        
                        for (int r = minRow; r < maxRow; r++)
                            for (int c = minCol; c < maxCol; c++)
                                if (spikes4DPooled[scl][f][r][c]){
                                    int dist = std::max(std::abs(row - r), std::abs(col - c));
                                    if (dist > 0 && spikes4DPooled[scl][f][r][c]->Time > spikes4DPooled[scl][f][row][col]->Time)
                                        inhibitions[scl][f](r, c) += InhibitionsPercent[dist - 1];
                                }
                    }
            }
            
        }
    }
    
    //apply inhibition
    for (int scl = 0; scl < spikes4D.size(); scl++)
        for (int f = 0; f < spikes4DPooled[scl].size(); f++)
            for (int row = 0; row < spikes4DPooled[scl][f].size(); row++)
                for (int col = 0; col < spikes4DPooled[scl][f][0].size(); col++)
                    if (spikes4DPooled[scl][f][row][col])
                        spikes4DPooled[scl][f][row][col]->Time +=
                        spikes4DPooled[scl][f][row][col]->Time *
                        inhibitions[scl][f](row, col);
    return spikes4DPooled;
}

inline void Kernel::increaseWeight(int f, int r, int c, int pref){
    Weights[f][pref](centerRow + r, centerCol + c) +=
				Ap * (Weights[f][pref](centerRow + r, centerCol + c)) * (1.0f - Weights[f][pref](centerRow + r, centerCol + c));
}

inline void Kernel::decreaseWeight(int f, int r, int c, int pref){
    Weights[f][pref](centerRow + r, centerCol + c) +=
				An * (Weights[f][pref](centerRow + r, centerCol + c)) * (1.0f - Weights[f][pref](centerRow + r, centerCol + c));
}

void Kernel::ApplySTDP(int f, int r, int c, SpikeVec3D& spikes3D, std::shared_ptr<SpikeData> currentSpike){
    for (int feature = 0; feature < Weights[f].size(); feature++)
        for (int i = -SRF / 2; i <= SRF / 2; i++){
            int row = r + i;
            for (int j = -SRF / 2; j <= SRF / 2; j++){
                int col = c + j;
                if (row >= 0 && row < spikes3D[feature].size() &&
                    col >= 0 && col < spikes3D[feature][0].size()){
                        if (spikes3D[feature][row][col] && spikes3D[feature][row][col]->Time <= currentSpike->Time)
                            increaseWeight(f, i, j, feature);
                        else
                            decreaseWeight(f, i, j, feature);
                }
            }
        }
    
}

inline float Kernel::getWeight(int f, int r, int c, int pref){
    return Weights[f][pref](centerRow + r, centerCol + c);
}