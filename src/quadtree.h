/*
 *
 * Copyright (c) 2013, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */


#ifndef QUADTREE_H
#define QUADTREE_H

using namespace std;

//static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max_tsne(double x, double y) { return (x <= y ? y : x); }
//static inline double abs(double x) { return (x < .0 ? -x : x); }

class Cell {
    
public:
    double x;
    double y;
    double hw;
    double hh;
    bool   containsPoint(double point[]);
};


class QuadTree
{
    
    // Fixed constants    
    static const int QT_NO_DIMS = 2;
    static const int QT_NODE_CAPACITY = 1;

    // A buffer we use when doing force computations
    double buff[QT_NO_DIMS];
    
    // Properties of this node in the tree
    QuadTree* parent;
    bool is_leaf;
    int size;
    int cum_size;
        
    // Axis-aligned bounding box stored as a center with half-dimensions to represent the boundaries of this quad tree
    Cell boundary;
    
    // Indices in this quad tree node, corresponding center-of-mass, and list of all children
    double* data;
    double center_of_mass[QT_NO_DIMS];
    int index[QT_NODE_CAPACITY];
    
    // Children
    QuadTree* northWest;
    QuadTree* northEast;
    QuadTree* southWest;
    QuadTree* southEast;
    
public:
    QuadTree(double* inp_data, int N);
    QuadTree(double* inp_data, double inp_x, double inp_y, double inp_hw, double inp_hh);
    QuadTree(double* inp_data, int N, double inp_x, double inp_y, double inp_hw, double inp_hh);
    QuadTree(QuadTree* inp_parent, double* inp_data, int N, double inp_x, double inp_y, double inp_hw, double inp_hh);
    QuadTree(QuadTree* inp_parent, double* inp_data, double inp_x, double inp_y, double inp_hw, double inp_hh);
    ~QuadTree();
    void setData(double* inp_data);
    QuadTree* getParent();
    void construct(Cell boundary);
    bool insert(int new_index);
    void subdivide();
    bool isCorrect();
    void rebuildTree();
    void getAllIndices(int* indices);
    int getDepth();
    void computeNonEdgeForces(int point_index, double theta, double neg_f[], double* sum_Q);
    void computeEdgeForces(int* row_P, int* col_P, double* val_P, int N, double* pos_f);    
    void print();
    
private:
    void init(QuadTree* inp_parent, double* inp_data, double inp_x, double inp_y, double inp_hw, double inp_hh);
    void fill(int N);
    int getAllIndices(int* indices, int loc);
    bool isChild(int test_index, int start, int end);
};

#endif
