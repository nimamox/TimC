//
//  common.hpp
//  TimC
//
//  Created by Nima Mohammadi
//  Copyright © 2016 Nima Mohammadi. All rights reserved.
//

#ifndef common_h
#define common_h

#define PI 3.141592653589793

#define EIGEN_INITIALIZE_MATRICES_BY_NAN

#include <memory>

typedef std::vector<std::vector<std::vector<std::vector<std::vector<std::shared_ptr<SpikeData>>>>>> SpikeVec5D;
typedef std::vector<std::vector<std::vector<std::vector<std::shared_ptr<SpikeData>>>>> SpikeVec4D;
typedef std::vector<std::vector<std::vector<std::shared_ptr<SpikeData>>>> SpikeVec3D;
typedef std::vector<std::vector<std::shared_ptr<SpikeData>>> SpikeVec2D;
typedef std::vector<std::shared_ptr<SpikeData>> SpikeVec1D;

namespace cereal
{
    template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols> inline
    typename std::enable_if<traits::is_output_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    save(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> const & m)
    {
        int32_t rows = m.rows();
        int32_t cols = m.cols();
        ar(rows);
        ar(cols);
        ar(binary_data(m.data(), rows * cols * sizeof(_Scalar)));
    }
    
    template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols> inline
    typename std::enable_if<traits::is_input_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    load(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & m)
    {
        int32_t rows;
        int32_t cols;
        ar(rows);
        ar(cols);
        
        m.resize(rows, cols);
        
        ar(binary_data(m.data(), static_cast<std::size_t>(rows * cols * sizeof(_Scalar))));
    }
}


#endif /* common_h */
