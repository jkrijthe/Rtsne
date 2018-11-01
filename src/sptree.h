/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
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

#ifndef SPTREE_H
#define SPTREE_H

using namespace std;

static inline double max_tsne(double x, double y) { return (x <= y ? y : x); }

template <int NDims=2>
class Cell {
  double corner[NDims];
  double width[NDims];
  
  
public:
  Cell();
  Cell(double* inp_corner, double* inp_width);
  ~Cell();
  
  double getCorner(unsigned int d) const;
  double getWidth(unsigned int d) const;
  void setCorner(unsigned int d, double val);
  void setWidth(unsigned int d, double val);
  bool containsPoint(double point[]) const;
};

template <int NDims=2>
class SPTree
{
public:
  enum { no_children = 2 * SPTree<NDims-1>::no_children };
  
private:
  // Fixed constants
  static const unsigned int QT_NODE_CAPACITY = 1;
  
  // Properties of this node in the tree
  SPTree<NDims>* parent;
  unsigned int dimension;
  bool is_leaf;
  unsigned int size;
  unsigned int cum_size;
  int num_threads;
  
  // Axis-aligned bounding box stored as a center with half-dimensions to represent the boundaries of this quad tree
  Cell<NDims> boundary;
  
  // Indices in this space-partitioning tree node, corresponding center-of-mass, and list of all children
  double* data;
  double center_of_mass[NDims];
  unsigned int index[QT_NODE_CAPACITY];
  
  // Children
  SPTree<NDims>* children[no_children];
  
public:
  SPTree(double* inp_data, unsigned int N);
  SPTree(double* inp_data, double* inp_corner, double* inp_width);
  SPTree(double* inp_data, unsigned int N, double* inp_corner, double* inp_width);
  SPTree(SPTree* inp_parent, double* inp_data, unsigned int N, double* inp_corner, double* inp_width);
  SPTree(SPTree* inp_parent, double* inp_data, double* inp_corner, double* inp_width);
  ~SPTree();
  void setData(double* inp_data);
  SPTree* getParent();
  void construct(Cell<NDims> boundary);
  bool insert(unsigned int new_index);
  void subdivide();
  bool isCorrect();
  void rebuildTree();
  void getAllIndices(unsigned int* indices);
  unsigned int getDepth();
  double computeNonEdgeForces(unsigned int point_index, double theta, double neg_f[]) const;
  void computeEdgeForces(unsigned int* row_P, unsigned int* col_P, double* val_P, unsigned int N, double* pos_f, int num_threads) const;
  void print();
  
private:
  void init(SPTree* inp_parent, double* inp_data, double* inp_corner, double* inp_width);
  void fill(unsigned int N);
  unsigned int getAllIndices(unsigned int* indices, unsigned int loc);
  bool isChild(unsigned int test_index, unsigned int start, unsigned int end);
};

template <>
struct SPTree<0>
{
  enum { 	no_children = 1 };
};

#endif