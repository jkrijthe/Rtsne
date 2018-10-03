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

#include "datapoint.h"

#ifndef TSNE_H
#define TSNE_H


static inline double sign_tsne(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

template <int NDims>
class TSNE
{    
public:
    void run(double* X, int N, int D, double* Y, int no_dims, double perplexity, 
             double theta, bool verbose, int max_iter, double* costs, 
             bool distance_precomputed, double* itercost, bool init, int stop_lying_iter, 
             int mom_switch_iter, double momentum, double final_momentum, double eta, double exaggeration_factor,int num_threads);
    void symmetrizeMatrix(unsigned int** row_P, unsigned int** col_P, double** val_P, int N); // should be static?!

    
private:
    void computeGradient(double* P, unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta);
    void computeExactGradient(double* P, double* Y, int N, int D, double* dC);
    double evaluateError(double* P, double* Y, int N, int D);
    double evaluateError(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta);
    void getCost(double* P, double* Y, int N, int D, double* costs);
    void getCost(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta, double* costs);
    void zeroMean(double* X, int N, int D);
    void computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity, bool distance_precomputed);
    template<double (*distance)( const DataPoint&, const DataPoint& )>
    void computeGaussianPerplexity(double* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double perplexity, int K, bool verbose);
    void computeGaussianPerplexity(const int* nn_dex, const double* nn_dist, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double perplexity, int K, bool verbose);
    void allocateMemory(unsigned int** _row_P, unsigned int** _col_P, double** _val_P, int N, int K);
    void computeProbabilities(const double perplexity, const int K, const double* distances, double* cur_P);

    void computeSquaredEuclideanDistance(double* X, int N, int D, double* DD);
    double randn();
};

#endif

