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

#include <math.h>
#include <float.h>
#include <R_ext/BLAS.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <Rcpp.h>
#include "sptree.h"



// Constructs cell
template<int NDims>
Cell<NDims>::Cell() {
}

template<int NDims>
Cell<NDims>::Cell(double* inp_corner, double* inp_width) {
  for(int d = 0; d < NDims; d++) setCorner(d, inp_corner[d]);
  for(int d = 0; d < NDims; d++) setWidth( d,  inp_width[d]);
}

// Destructs cell
template<int NDims>
Cell<NDims>::~Cell() {
}

template<int NDims>
double Cell<NDims>::getCorner(unsigned int d) const {
  return corner[d];
}

template<int NDims>
double Cell<NDims>::getWidth(unsigned int d) const {
  return width[d];
}

template<int NDims>
void Cell<NDims>::setCorner(unsigned int d, double val) {
  corner[d] = val;
}

template<int NDims>
void Cell<NDims>::setWidth(unsigned int d, double val) {
  width[d] = val;
}

// Checks whether a point lies in a cell
template<int NDims>
bool Cell<NDims>::containsPoint(double point[]) const
{
  for(int d = 0; d < NDims; d++) {
    if(corner[d] - width[d] > point[d]) return false;
    if(corner[d] + width[d] < point[d]) return false;
  }
  return true;
}


// Default constructor for SPTree -- build tree, too!
template<int NDims>
SPTree<NDims>::SPTree(double* inp_data, unsigned int N)
{
  
  // Compute mean, width, and height of current map (boundaries of SPTree)
  int nD = 0;
  double* mean_Y = (double*) calloc(NDims,  sizeof(double));
  double*  min_Y = (double*) malloc(NDims * sizeof(double));
  double*  max_Y = (double*) malloc(NDims * sizeof(double));
  
  for(unsigned int d = 0; d < NDims; d++)  {
    min_Y[d] =  DBL_MAX;
    max_Y[d] = -DBL_MAX;
  }
  
  for(unsigned int n = 0; n < N; n++) {
    for(unsigned int d = 0; d < NDims; d++) {
      mean_Y[d] += inp_data[n * NDims + d];
      if(inp_data[nD + d] < min_Y[d]) min_Y[d] = inp_data[nD + d];
      if(inp_data[nD + d] > max_Y[d]) max_Y[d] = inp_data[nD + d];
    }
    nD += NDims;
  }
  
  for(int d = 0; d < NDims; d++) mean_Y[d] /= (double) N;
  // double sum_m = 0.0;
  // for(int d = 0; d < NDims; d++) sum_m+=mean_Y[d];
  // Rprintf("mean: %4.25f\n",sum_m);
  
  // Construct SPTree
  double* width = (double*) malloc(NDims * sizeof(double));
  for(int d = 0; d < NDims; d++) width[d] = max_tsne(max_Y[d] - mean_Y[d], mean_Y[d] - min_Y[d]) + 1e-5;
  init(NULL, inp_data, mean_Y, width);
  fill(N);
  
  // Clean up memory
  free(mean_Y);
  free(max_Y);
  free(min_Y);
  free(width);
}


// Constructor for SPTree with particular size and parent -- build the tree, too!
template<int NDims>
SPTree<NDims>::SPTree(double* inp_data, unsigned int N, double* inp_corner, double* inp_width)
{
  init(NULL, inp_data, inp_corner, inp_width);
  fill(N);
}


// Constructor for SPTree with particular size (do not fill the tree)
template<int NDims>
SPTree<NDims>::SPTree(double* inp_data, double* inp_corner, double* inp_width)
{
  init(NULL, inp_data, inp_corner, inp_width);
}


// Constructor for SPTree with particular size and parent (do not fill tree)
template<int NDims>
SPTree<NDims>::SPTree(SPTree* inp_parent, double* inp_data, double* inp_corner, double* inp_width) {
  init(inp_parent, inp_data, inp_corner, inp_width);
}


// Constructor for SPTree with particular size and parent -- build the tree, too!
template<int NDims>
SPTree<NDims>::SPTree(SPTree* inp_parent, double* inp_data, unsigned int N, double* inp_corner, double* inp_width)
{
  init(inp_parent, inp_data, inp_corner, inp_width);
  fill(N);
}


// Main initialization function
template<int NDims>
void SPTree<NDims>::init(SPTree* inp_parent, double* inp_data, double* inp_corner, double* inp_width)
{
  parent = inp_parent;
  data = inp_data;
  is_leaf = true;
  size = 0;
  cum_size = 0;
  
  for(unsigned int d = 0; d < NDims; d++) boundary.setCorner(d, inp_corner[d]);
  for(unsigned int d = 0; d < NDims; d++) boundary.setWidth( d, inp_width[d]);
  
  for(unsigned int i = 0; i < no_children; i++) children[i] = NULL;
  for(unsigned int d = 0; d < NDims; d++) center_of_mass[d] = .0;
}


// Destructor for SPTree
template<int NDims>
SPTree<NDims>::~SPTree()
{
  for(unsigned int i = 0; i < no_children; i++) {
    if(children[i] != NULL) delete children[i];
  }
}


// Update the data underlying this tree
template<int NDims>
void SPTree<NDims>::setData(double* inp_data)
{
  data = inp_data;
}


// Get the parent of the current tree
template<int NDims>
SPTree<NDims>* SPTree<NDims>::getParent()
{
  return parent;
}


// Insert a point into the SPTree
template<int NDims>
bool SPTree<NDims>::insert(unsigned int new_index)
{
  // Ignore objects which do not belong in this quad tree
  double* point = data + new_index * NDims;
  if(!boundary.containsPoint(point))
    return false;

  // Online update of cumulative size and center-of-mass
  cum_size++;
  double mult1 = (double) (cum_size - 1) / (double) cum_size;
  double mult2 = 1.0 / (double) cum_size;
  
  for(unsigned int d = 0; d < NDims; d++) {
    center_of_mass[d] = center_of_mass[d] * mult1 + mult2 * point[d];
  }
  
  // If there is space in this quad tree and it is a leaf, add the object here
  if(is_leaf && size < QT_NODE_CAPACITY) {
    index[size] = new_index;
    size++;
    return true;
  }
  
  // Don't add duplicates for now (this is not very nice)
  bool any_duplicate = false;
  for(unsigned int n = 0; n < size; n++) {
    bool duplicate = true;
    for(unsigned int d = 0; d < NDims; d++) {
      if(point[d] != data[index[n] * NDims + d]) { duplicate = false; break; }
    }
    any_duplicate = any_duplicate | duplicate;
  }
  if(any_duplicate) return true;
  
  // Otherwise, we need to subdivide the current cell
  if(is_leaf) subdivide();
  
  // Find out where the point can be inserted
  for(unsigned int i = 0; i < no_children; i++) {
    if(children[i]->insert(new_index)) return true;
  }
  
  // Otherwise, the point cannot be inserted (this should never happen)
  return false;
}


// Create four children which fully divide this cell into four quads of equal area
template<int NDims>
void SPTree<NDims>::subdivide() {
  
  // Create new children
  double new_corner[NDims];
  double new_width[NDims];
  for(unsigned int i = 0; i < no_children; i++) {
    unsigned int div = 1;
    for(unsigned int d = 0; d < NDims; d++) {
      new_width[d] = .5 * boundary.getWidth(d);
      if((i / div) % 2 == 1) new_corner[d] = boundary.getCorner(d) - .5 * boundary.getWidth(d);
      else                   new_corner[d] = boundary.getCorner(d) + .5 * boundary.getWidth(d);
      div *= 2;
    }
    children[i] = new SPTree(this, data, new_corner, new_width);
  }
  
  // Move existing points to correct children
  for(unsigned int i = 0; i < size; i++) {
    bool success = false;
    for(unsigned int j = 0; j < no_children; j++) {
      if(!success) success = children[j]->insert(index[i]);
    }
    index[i] = -1;
  }
  
  // Empty parent node
  size = 0;
  is_leaf = false;
}


// Build SPTree on dataset
template<int NDims>
void SPTree<NDims>::fill(unsigned int N)
{
  for(unsigned int i = 0; i < N; i++) insert(i);
}


// Checks whether the specified tree is correct
template<int NDims>
bool SPTree<NDims>::isCorrect()
{
  for(unsigned int n = 0; n < size; n++) {
    double* point = data + index[n] * NDims;
    if(!boundary.containsPoint(point)) return false;
  }
  if(!is_leaf) {
    bool correct = true;
    for(int i = 0; i < no_children; i++) correct = correct && children[i]->isCorrect();
    return correct;
  }
  else return true;
}



// Build a list of all indices in SPTree
template<int NDims>
void SPTree<NDims>::getAllIndices(unsigned int* indices)
{
  getAllIndices(indices, 0);
}


// Build a list of all indices in SPTree
template<int NDims>
unsigned int SPTree<NDims>::getAllIndices(unsigned int* indices, unsigned int loc)
{
  
  // Gather indices in current quadrant
  for(unsigned int i = 0; i < size; i++) indices[loc + i] = index[i];
  loc += size;
  
  // Gather indices in children
  if(!is_leaf) {
    for(int i = 0; i < no_children; i++) loc = children[i]->getAllIndices(indices, loc);
  }
  return loc;
}

template<int NDims>
unsigned int SPTree<NDims>::getDepth() {
  if(is_leaf) return 1;
  int depth = 0;
  for(unsigned int i = 0; i < no_children; i++) depth = max_tsne(depth, children[i]->getDepth());
  return 1 + depth;
}


// Compute non-edge forces using Barnes-Hut algorithm
template<int NDims>
double SPTree<NDims>::computeNonEdgeForces(unsigned int point_index, double theta, double neg_f[]) const
{
  double resultSum = 0;
  double buff[NDims];  // make buff local for parallelization
  
  // Make sure that we spend no time on empty nodes or self-interactions
  if(cum_size == 0 || (is_leaf && size == 1 && index[0] == point_index)) return resultSum;
  
  // Compute distance between point and center-of-mass
  double sqdist = .0;
  unsigned int ind = point_index * NDims;
  
  for(unsigned int d = 0; d < NDims; d++) {
    buff[d] = data[ind + d] - center_of_mass[d];
    sqdist += buff[d] * buff[d];
  }
  
  // Check whether we can use this node as a "summary"
  double max_width = 0.0;
  double cur_width;
  for(unsigned int d = 0; d < NDims; d++) {
    cur_width = boundary.getWidth(d);
    max_width = (max_width > cur_width) ? max_width : cur_width;
  }
  if(is_leaf || max_width / sqrt(sqdist) < theta) {
    
    // Compute and add t-SNE force between point and current node
    sqdist = 1.0 / (1.0 + sqdist);
    double mult = cum_size * sqdist;
    resultSum += mult;
    mult *= sqdist;
    for(unsigned int d = 0; d < NDims; d++) neg_f[d] += mult * buff[d];
  }
  else {
    
    // Recursively apply Barnes-Hut to children
    for(unsigned int i = 0; i < no_children; i++){
      resultSum += children[i]->computeNonEdgeForces(point_index, theta, neg_f);
    }
  }
  return resultSum;
}


// Computes edge forces
template<int NDims>
void SPTree<NDims>::computeEdgeForces(unsigned int* row_P, unsigned int* col_P, double* val_P, int N, double* pos_f) const
{
  
  // Loop over all edges in the graph
  #pragma omp parallel for schedule(static)
  for(unsigned int n = 0; n < N; n++) {
    unsigned int ind1 = n * NDims;
    for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {
      
      double buff[NDims]; // make buff local for parallelization
      
      // Compute pairwise distance and Q-value
      double sqdist = 1.0;
      unsigned int ind2 = col_P[i] * NDims;
      
      for(unsigned int d = 0; d < NDims; d++) {
        buff[d] = data[ind1 + d] - data[ind2 + d];
        sqdist += buff[d] * buff[d];
      }
      
      sqdist = val_P[i] / sqdist;
      //Rprintf("it%d: %4.25f\n",n,sqdist);
      
      // Sum positive force
      for(unsigned int d = 0; d < NDims; d++) pos_f[ind1 + d] += sqdist * buff[d];
    }
  }
}


// Print out tree
template<int NDims>
void SPTree<NDims>::print()
{
  if(cum_size == 0) {
    Rprintf("Empty node\n");
    return;
  }

  if(is_leaf) {
    Rprintf("Leaf node; data = [");
    for(int i = 0; i < size; i++) {
      double* point = data + index[i] * NDims;
      for(int d = 0; d < NDims; d++) Rprintf("%f, ", point[d]);
      Rprintf(" (index = %d)", index[i]);
      if(i < size - 1) Rprintf("\n");
      else Rprintf("]\n");
    }
  }
  else {
    Rprintf("Intersection node with center-of-mass = [");
    for(int d = 0; d < NDims; d++) Rprintf("%f, ", center_of_mass[d]);
    Rprintf("]; children are:\n");
    for(int i = 0; i < no_children; i++) children[i]->print();
  }
}

// declare templates explicitly
template class SPTree<1>;
template class SPTree<2>;
template class SPTree<3>;