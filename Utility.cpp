//
//  Utility.cpp
//  TimC
//
//  Created by Nima Mohammadi
//  Copyright © 2016 Nima Mohammadi. All rights reserved.
//

#include "Utility.hpp"
#include "png/lodepng.hpp"

#include <iostream>

#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include "Eigen/Dense"
#include <fstream>

std::vector<int> resizeBilinear(std::vector<int> pixels, int w, int h, int w2, int h2) {
    std::vector<int> temp;
    temp.reserve(w2 * h2);
    int a, b, c, d, x, y, index, grey;
    float x_ratio = ((float)(w-1))/w2;
    float y_ratio = ((float)(h-1))/h2;
    float x_diff, y_diff;
    
    for (int i=0; i<h2; i++){
        for (int j=0; j<w2; j++){
            x = (int)(x_ratio * j);
            y = (int)(y_ratio * i);
            x_diff = (x_ratio * j) - x;
            y_diff = (y_ratio * i) - y;
            index = (y*w+x);
            
            a = pixels[index] & 0xff;
            b = pixels[index+1] & 0xff;
            c = pixels[index+w] & 0xff;
            d = pixels[index+w+1] & 0xff;
            
            grey = (int)(
                         a*(1-x_diff)*(1-y_diff) +  b*(x_diff)*(1-y_diff) +
                         c*(y_diff)*(1-x_diff)   +  d*(x_diff*y_diff)
                         );
            
            temp.push_back(grey);
        }
    }
    return temp;
}

Eigen::MatrixXf ReadImageGrayScale(std::string imageAddress, float scale){
    std::vector<unsigned char> origImage;
    unsigned origWidth, origHeight;
    unsigned long origSize;
    lodepng::decode(origImage, origWidth, origHeight, imageAddress);
    
    std::vector<int> greyImage;
    origSize = origImage.size();
    for (int i = 0; i < origSize/4; i++){
        greyImage.push_back((int)(origImage[i*4] + origImage[i*4+1] + origImage[i*4+2]) / 3);
    }
    
    int scaledWidth = (int)(origWidth * scale);
    int scaledHeight = (int)(origHeight * scale);
    
    std::vector<int> scaledImage;
    
    if (scale!=1.0)
        scaledImage = resizeBilinear(greyImage, origWidth, origHeight, scaledWidth, scaledHeight);
    else
        scaledImage = greyImage;
    
//    std::vector<unsigned char> tv;
//    for (int i=0; i<scaledImage.size(); i++){
//        tv.push_back(scaledImage[i]);
//        tv.push_back(scaledImage[i]);
//        tv.push_back(scaledImage[i]);
//        tv.push_back(0xff);
//    }
//    lodepng::encode("/tmp/out.png", tv, scaledWidth, scaledHeight);
    
    Eigen::MatrixXf img(scaledHeight, scaledWidth);
    for (int i=0; i<scaledImage.size(); i++){
        img(i/scaledWidth, i%scaledWidth) = scaledImage[i] / 255.0;
    }
    return img;
}