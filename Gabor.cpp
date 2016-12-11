//
//  Gabor.cpp
//  TimC
//
//  Created by Nima Mohammadi
//  Copyright © 2016 Nima Mohammadi. All rights reserved.
//

#include <iostream>
#include "Gabor.hpp"
#include "Utility.hpp"
#include "png/lodepng.hpp"
//#include "Eigen/StdVector"

Gabor::Gabor(std::vector<float> scales, std::vector<float> orientations, int rfSize, float div){
    Scales = scales;
    Orientations = orientations;
    RFSize = rfSize;
    Div = div;
    
    GenerateGaborFilters();
}

void Gabor::GenerateGaborFilters(){
    float lambda = RFSize * 2 / Div;
    float sigma = lambda * 0.8f;
    float sigmaSq = sigma * sigma;
    float g = 0.3f;
    int offset = RFSize / 2;
    for (int ori = 0; ori < Orientations.size(); ori++){
        gaborFilters.push_back(Eigen::MatrixXf(RFSize, RFSize));
        gaborFilters[ori].setZero();
        float sumSq = 0;
        float sum = 0;
        
        float theta = (Orientations[ori] * (float)PI) / 180;
        std::cout << "THETA: " << theta << std::endl;
        for (int i = -offset; i <= offset; i++){
            for (int j = -offset; j <= offset; j++){
                float value = 0;
                if (std::sqrt(i * i + j * j) <= RFSize / 2.0f){
                    float x = (float)(i * std::cos(theta) - j * std::sin(theta));
                    float y = (float)(i * std::sin(theta) + j * std::cos(theta));
                    value = (float)(std::exp(-(x * x + g * g * y * y) / (2 * sigmaSq))) *
                    std::cos(2 * PI * x / lambda);
                    sum += value;
                    sumSq += value * value;
                }
                gaborFilters[ori](i + offset, offset - j) = value;
            }
        }
//        std::cout << "1GF[" << ori << "]=" << gaborFilters[ori].sum() << std::endl;
        float mean = sum / (RFSize * RFSize);
        sumSq = std::sqrt(sumSq);
        
//        gaborFilters[ori].block(0, 0, 7, 7) = (gaborFilters[ori].block(0, 0, 7, 7).array()-mean) / sumSq;
//        std::cout << "2GF[" << ori << "]=" << ((gaborFilters[ori].block(0, 0, 7, 7).array()-mean) / sumSq).sum() << std::endl;
        for (int i = -offset; i <= offset; i++)
            for (int j = -offset; j <= offset; j++){
                gaborFilters[ori](i + offset, j + offset) -= mean;
                gaborFilters[ori](i + offset, j + offset) /= sumSq;
                //q += gaborFilters[ori](i + offset, j + offset);
                //std::cout << "[" << i + offset << ", " << j+offset << "]" << gaborFilters[ori](i + offset, j + offset) << std::endl;
            }
//        std::cout << "3GF[" << ori << "]=" << gaborFilters[ori].sum() << std::endl;
        
    }
}

SpikeVec4D Gabor::GetGaboredTimes(std::string imageAddress){
    SpikeVec4D  spike4D;
    std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> scaledImages;
    std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> normedImages;
    std::vector<std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>>> gaboredImages;
    
    int offset = RFSize / 2;
    
    //compute normal images
    for (int scl = 0; scl < Scales.size(); scl++){
        scaledImages.push_back(ReadImageGrayScale(imageAddress, Scales[scl]));
        normedImages.push_back(Eigen::MatrixXf(scaledImages[scl].rows(), scaledImages[scl].cols()));
        normedImages[scl].setZero();
        gaboredImages.push_back(std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>>());
        spike4D.push_back(SpikeVec3D());
        
        //compute normals
        for (int i = 0; i < scaledImages[scl].rows(); i++){
            for (int j = 0; j < scaledImages[scl].cols(); j++){
                for (int row = -offset; row <= offset; row++)
                    if (i + row >= 0 && i + row < scaledImages[scl].rows())
                        for (int col = -offset; col <= offset; col++)
                            if (j + col >= 0 && j + col < scaledImages[scl].cols())
                                normedImages[scl](i, j) += scaledImages[scl](i + row, j + col) * scaledImages[scl](i + row, j + col);
                normedImages[scl](i, j) = (float)sqrt(normedImages[scl](i, j));
                if (normedImages[scl](i, j) == 0)
                    normedImages[scl](i, j) = 1;
            }
        }
        
//        std::cout << "nrSUM: " << normedImages[scl].sum() << std::endl;
        
        for (int ori = 0; ori < Orientations.size(); ori++){
            spike4D[scl].push_back(SpikeVec2D());
            for (int i = 0; i < scaledImages[scl].rows(); i++){
                spike4D[scl][ori].push_back(SpikeVec1D());
                for (int j = 0; j < scaledImages[scl].cols() - offset; j++){
                    spike4D[scl][ori][i].push_back(std::shared_ptr<SpikeData>());
                }
            }
        }
        
        //compute gabored
        for (int ori = 0; ori < Orientations.size(); ori++){
            gaboredImages[scl].push_back(Eigen::MatrixXf(scaledImages[scl].rows(), scaledImages[scl].cols()));
            gaboredImages[scl][ori].setZero();
            for (int i = offset; i < scaledImages[scl].rows() - offset; i++)
                for (int j = offset; j < scaledImages[scl].cols() - offset; j++){
                    for (int row = i - offset, fr = 0; row <= i + offset; row++, fr++)
                        for (int col = j - offset, fc = 0; col <= j + offset; col++, fc++){
                            gaboredImages[scl][ori](i, j) += gaborFilters[ori](fr, fc) * scaledImages[scl](row, col);
                        }
                    gaboredImages[scl][ori](i, j) = std::abs(gaboredImages[scl][ori](i, j));
                    gaboredImages[scl][ori](i, j) /= normedImages[scl](i, j);
                    if (gaboredImages[scl][ori](i, j) >= 0.01){
                        spike4D[scl][ori][i][j] = std::make_shared<SpikeData>(1 / gaboredImages[scl][ori](i, j), i, j, ori, scl);
                    }
                }
//            gaboredImages[scl][ori] = gaboredImages[scl][ori].cwiseAbs().cwiseQuotient(normedImages[scl]);
//            for (int i = offset; i < scaledImages[scl].rows() - offset; i++)
//                for (int j = offset; j < scaledImages[scl].cols() - offset; j++){
//                        spike4D[scl][ori][i][j] = new SpikeData(1 / gaboredImages[scl][ori](i, j), i, j, ori, scl);
//                }
//            std::cout << "grSUM: " << scl << "_" << ori << "_" << gaboredImages[scl][ori].sum() << std::endl;
            
//            //Save gabor images on disk
//            gaboredImages[scl][ori] = gaboredImages[scl][ori] / gaboredImages[scl][ori].maxCoeff() * 255;
//            std::vector<unsigned char> tv;
//            for (int i=0; i<gaboredImages[scl][ori].rows(); i++){
//                for (int j=0; j<gaboredImages[scl][ori].cols(); j++){
//                    for (int k=0; k<3; k++)
//                        tv.push_back(gaboredImages[scl][ori](i, j));
//                    tv.push_back(0xff);
//                }
//            }
//            lodepng::encode((std::string("/tmp/out") + std::to_string(ori) + std::string(".png")).c_str(), tv, gaboredImages[scl][ori].cols(), gaboredImages[scl][ori].rows());
        }
    }
    return spike4D;
}