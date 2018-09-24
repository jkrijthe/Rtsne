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
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <time.h>
#include <Rcpp.h>
#include "sptree.h"
#include "vptree.h"
#include "tsne.h"


extern "C" {
    #include <R_ext/BLAS.h>
}


using namespace std;

// Perform t-SNE
void TSNE::run(double* X, int N, int D, double* Y, int no_dims, 
               double perplexity, double theta, bool verbose, int max_iter, double* cost, 
               bool distance_precomputed, bool neighbors_precomputed, int precomputed_K, const int* nndex, const double* nndist,
               double* itercost, bool init, 
               int stop_lying_iter, int mom_switch_iter, double momentum, double final_momentum, 
               double eta, double exaggeration_factor) {
    
    // Determine whether we are using an exact algorithm
    if(N - 1 < 3 * perplexity) { Rcpp::stop("Perplexity too large for the number of data points!\n"); }
    if (verbose) Rprintf("Using no_dims = %d, perplexity = %f, and theta = %f\n", no_dims, perplexity, theta);
    bool exact = (theta == .0) ? true : false;
    
    // Set learning parameters
    float total_time = .0;
    clock_t start, end;
    
    // Allocate some memory
    double* dY    = (double*) malloc(N * no_dims * sizeof(double));
    double* uY    = (double*) malloc(N * no_dims * sizeof(double));
    double* gains = (double*) malloc(N * no_dims * sizeof(double));
    if(dY == NULL || uY == NULL || gains == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    for(int i = 0; i < N * no_dims; i++)    uY[i] =  .0;
    for(int i = 0; i < N * no_dims; i++) gains[i] = 1.0;
    
    // Normalize input data (to prevent numerical problems)
    if (verbose) Rprintf("Computing input similarities...\n");
    start = clock();
    if (!distance_precomputed) {
      if (verbose) Rprintf("Normalizing input...\n");
      zeroMean(X, N, D);
      double max_X = .0;
      for(int i = 0; i < N * D; i++) {
        if(fabs(X[i]) > max_X) max_X = fabs(X[i]);
      }
      for(int i = 0; i < N * D; i++) X[i] /= max_X;
    }
    
    // Compute input similarities for exact t-SNE
    double* P; int* row_P; int* col_P; double* val_P;
    if(exact) {
        
        // Compute similarities
        P = (double*) malloc((long)N * N * sizeof(double));
        if(P == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
        computeGaussianPerplexity(X, N, D, P, perplexity, distance_precomputed);
    
        // Symmetrize input similarities
        if (verbose) Rprintf("Symmetrizing...\n");
        for(unsigned long n = 0; n < N; n++) {
            for(unsigned long m = n + 1; m < N; m++) {
                P[n * N + m] += P[m * N + n];
                P[m * N + n]  = P[n * N + m];
            }
        }
        double sum_P = .0;
        for(unsigned long i = 0; i < (long)N * N; i++) sum_P += P[i];
        for(unsigned long i = 0; i < (long)N * N; i++) P[i] /= sum_P;
    }
    
    // Compute input similarities for approximate t-SNE
    else {

        // Compute asymmetric pairwise input similarities
        computeGaussianPerplexity(X, N, D, &row_P, &col_P, &val_P, perplexity, (neighbors_precomputed ? precomputed_K : (int) (3 * perplexity)), 
                verbose, distance_precomputed, neighbors_precomputed, nndex, nndist);

        // Symmetrize input similarities
        symmetrizeMatrix(&row_P, &col_P, &val_P, N);
        double sum_P = .0;
        for(int i = 0; i < row_P[N]; i++) sum_P += val_P[i];
        for(int i = 0; i < row_P[N]; i++) val_P[i] /= sum_P;
    }
    end = clock();
    
    // Lie about the P-values
    if(exact) { for(unsigned long i = 0; i < (long)N * N; i++)        P[i] *= exaggeration_factor; }
    else {      for(unsigned long i = 0; i < row_P[N]; i++) val_P[i] *= exaggeration_factor; }

	// Initialize solution (randomly), if not already done
	if (!init) { for(int i = 0; i < N * no_dims; i++) Y[i] = randn() * .0001; }

	
	// Perform main training loop
  if (verbose) {
    if(exact) Rprintf("Done in %4.2f seconds!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC);
    else Rprintf("Done in %4.2f seconds (sparsity = %f)!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC, (double) row_P[N] / ((double) N * (double) N));
  }
  start = clock();
  int costi = 0; //iterator for saving the total costs for the iterations
  
	for(int iter = 0; iter < max_iter; iter++) {
        
        // Stop lying about the P-values after a while, and switch momentum
        if(iter == stop_lying_iter) {
          if(exact) { for(unsigned long i = 0; i < (long)N * N; i++)        P[i] /= exaggeration_factor; }
          else      { for(int i = 0; i < row_P[N]; i++) val_P[i] /= exaggeration_factor; }
        }
        if(iter == mom_switch_iter) momentum = final_momentum;
        
        // Compute (approximate) gradient
        if(exact) computeExactGradient(P, Y, N, no_dims, dY);
        else computeGradient(P, row_P, col_P, val_P, Y, N, no_dims, dY, theta);
        
        // Update gains
        for(int i = 0; i < N * no_dims; i++) gains[i] = (sign_tsne(dY[i]) != sign_tsne(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
        for(int i = 0; i < N * no_dims; i++) if(gains[i] < .01) gains[i] = .01;
            
        // Perform gradient update (with momentum and gains)
        for(int i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
		    for(int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];
        
        // Make solution zero-mean
		    zeroMean(Y, N, no_dims);
        
        // Print out progress
        if((iter > 0 && (iter+1) % 50 == 0) || iter == max_iter - 1) {
            end = clock();
            double C = .0;
            if(exact) C = evaluateError(P, Y, N, no_dims);
            else      C = evaluateError(row_P, col_P, val_P, Y, N, no_dims, theta);  // doing approximate computation here!
            if(iter == 0) {
                if (verbose) Rprintf("Iteration %d: error is %f\n", iter + 1, C);
            } 
            else {
                total_time += (float) (end - start) / CLOCKS_PER_SEC;
                if (verbose) Rprintf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter+1, C, (float) (end - start) / CLOCKS_PER_SEC);
            }
            itercost[costi] = C;
            itercost++;
			  start = clock();
        }
    }
    end = clock(); total_time += (float) (end - start) / CLOCKS_PER_SEC;
    
    if(exact) getCost(P, Y, N, no_dims, cost);
    else      getCost(row_P, col_P, val_P, Y, N, no_dims, theta, cost);  // doing approximate computation here!
    
    
    // Clean up memory
    free(dY);
    free(uY);
    free(gains);
    if(exact) free(P);
    else {
        free(row_P); row_P = NULL;
        free(col_P); col_P = NULL;
        free(val_P); val_P = NULL;
    }
    if (verbose) Rprintf("Fitting performed in %4.2f seconds.\n", total_time);
}


// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
void TSNE::computeGradient(double* P, int* inp_row_P, int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta)
{
    
    // Construct space-partitioning tree on current map
    SPTree* tree = new SPTree(D, Y, N);
    
    // Compute all terms required for t-SNE gradient
    double sum_Q = .0;
    double* pos_f = (double*) calloc(N * D, sizeof(double));
    double* neg_f = (double*) calloc(N * D, sizeof(double));
    if(pos_f == NULL || neg_f == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f);
    for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q);
    
    // Compute final t-SNE gradient
    for(int i = 0; i < N * D; i++) {
        dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
    }
    free(pos_f);
    free(neg_f);
    delete tree;
}

// Compute gradient of the t-SNE cost function (exact)
void TSNE::computeExactGradient(double* P, double* Y, int N, int D, double* dC) {
	
	// Make sure the current gradient contains zeros
	for(int i = 0; i < N * D; i++) dC[i] = 0.0;
    
    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc((long)N * N * sizeof(double));
    if(DD == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    computeSquaredEuclideanDistance(Y, N, D, DD);
    
    // Compute Q-matrix and normalization sum
    double* Q    = (double*) malloc((long)N * N * sizeof(double));
    if(Q == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    double sum_Q = .0;
    for(unsigned long n = 0; n < N; n++) {
    	for(unsigned long m = 0; m < N; m++) {
            if(n != m) {
                Q[n * N + m] = 1 / (1 + DD[n * N + m]);
                sum_Q += Q[n * N + m];
            }
        }
    }
    
	// Perform the computation of the gradient
	for(unsigned long n = 0; n < N; n++) {
    	for(unsigned long m = 0; m < N; m++) {
            if(n != m) {
                double mult = (P[n * N + m] - (Q[n * N + m] / sum_Q)) * Q[n * N + m];
                for(int d = 0; d < D; d++) {
                    dC[n * D + d] += (Y[n * D + d] - Y[m * D + d]) * mult;
                }
            }
		}
	}
    
    // Free memory
    free(DD); DD = NULL;
    free(Q);  Q  = NULL;
}


// Evaluate t-SNE cost function (exactly)
double TSNE::evaluateError(double* P, double* Y, int N, int D) {
    
    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc((long)N * N * sizeof(double));
    double* Q = (double*) malloc((long)N * N * sizeof(double));
    if(DD == NULL || Q == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    computeSquaredEuclideanDistance(Y, N, D, DD);
    
    // Compute Q-matrix and normalization sum
    double sum_Q = DBL_MIN;
    for(unsigned long n = 0; n < N; n++) {
    	for(unsigned long m = 0; m < N; m++) {
            if(n != m) {
                Q[n * N + m] = 1 / (1 + DD[n * N + m]);
                sum_Q += Q[n * N + m];
            }
            else Q[n * N + m] = DBL_MIN;
        }
    }
    for(unsigned long i = 0; i < (long)N * N; i++) Q[i] /= sum_Q;
    
    // Sum t-SNE error
    double C = .0;
  	for(unsigned long n = 0; n < N; n++) {
  		for(unsigned long m = 0; m < N; m++) {
  			C += P[n * N + m] * log((P[n * N + m] + 1e-9) / (Q[n * N + m] + 1e-9));
  		}
  	}
    
    // Clean up memory
    free(DD);
    free(Q);
	return C;
}

// Evaluate t-SNE cost function (approximately)
double TSNE::evaluateError(int* row_P, int* col_P, double* val_P, double* Y, int N, int D, double theta)
{
    
    // Get estimate of normalization term
    SPTree* tree = new SPTree(D, Y, N);
    double* buff = (double*) calloc(D, sizeof(double));
    double sum_Q = .0;
    for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, buff, &sum_Q);
    
    // Loop over all edges to compute t-SNE error
    int ind1, ind2;
    double C = .0, Q;
    for(int n = 0; n < N; n++) {
        ind1 = n * D;
        for(int i = row_P[n]; i < row_P[n + 1]; i++) {
            Q = .0;
            ind2 = col_P[i] * D;
            for(int d = 0; d < D; d++) buff[d]  = Y[ind1 + d];
            for(int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
            for(int d = 0; d < D; d++) Q += buff[d] * buff[d];
            Q = (1.0 / (1.0 + Q)) / sum_Q;
            C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
        }
    }
    
    // Clean up memory
    free(buff);
    delete tree;
    return C;
}

// Evaluate t-SNE cost function (exactly)
void TSNE::getCost(double* P, double* Y, int N, int D, double* costs) {
  
  // Compute the squared Euclidean distance matrix
  double* DD = (double*) malloc((long)N * N * sizeof(double));
  double* Q = (double*) malloc((long)N * N * sizeof(double));
  if(DD == NULL || Q == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
  computeSquaredEuclideanDistance(Y, N, D, DD);
  
  // Compute Q-matrix and normalization sum
  double sum_Q = DBL_MIN;
  for(unsigned long n = 0; n < N; n++) {
    for(unsigned long m = 0; m < N; m++) {
      if(n != m) {
        Q[n * N + m] = 1 / (1 + DD[n * N + m]);
        sum_Q += Q[n * N + m];
      }
      else Q[n * N + m] = DBL_MIN;
    }
  }
  for(unsigned long i = 0; i < (long)N * N; i++) Q[i] /= sum_Q;
  
  // Sum t-SNE error
  for(unsigned long n = 0; n < N; n++) {
    costs[n] = 0.0;
    for(unsigned long m = 0; m < N; m++) {
      costs[n] += P[n * N + m] * log((P[n * N + m] + 1e-9) / (Q[n * N + m] + 1e-9));
    }
  }
  
  // Clean up memory
  free(DD);
  free(Q);
}

// Evaluate t-SNE cost function (approximately)
void TSNE::getCost(int* row_P, int* col_P, double* val_P, double* Y, int N, int D, double theta, double* costs)
{
  
  // Get estimate of normalization term
  SPTree* tree = new SPTree(D, Y, N);
  double* buff = (double*) calloc(D, sizeof(double));
  double sum_Q = .0;
  for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, buff, &sum_Q);
  
  // Loop over all edges to compute t-SNE error
  int ind1, ind2;
  double  Q;
  for(int n = 0; n < N; n++) {
    ind1 = n * D;
    costs[n] = 0.0;
    for(int i = row_P[n]; i < row_P[n + 1]; i++) {
      Q = .0;
      ind2 = col_P[i] * D;
      for(int d = 0; d < D; d++) buff[d]  = Y[ind1 + d];
      for(int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
      for(int d = 0; d < D; d++) Q += buff[d] * buff[d];
      Q = (1.0 / (1.0 + Q)) / sum_Q;
      costs[n] += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
    }
  }
  
  // Clean up memory
  free(buff);
  delete tree;
}


// Compute input similarities with a fixed perplexity
void TSNE::computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity, bool distance_precomputed) {
	
	// Compute the squared Euclidean distance matrix
	double* DD = (double*) malloc((long)N * N * sizeof(double));
    if(DD == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
	
	if (distance_precomputed) {
	  DD = X;
	} else {
	  computeSquaredEuclideanDistance(X, N, D, DD);
	}
	
	// For debugging purposes:
// 	for (int n=0; n<N*N; n++) {
//     Rprintf(" %4.4f \n", DD[n]);
// 	}

	// Compute the Gaussian kernel row by row
	for(unsigned long n = 0; n < N; n++) {
        
		// Initialize some variables
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta =  DBL_MAX;
		double tol = 1e-5;
        double sum_P;
		
		// Iterate until we found a good perplexity
		int iter = 0;
		while(!found && iter < 200) {
			
			// Compute Gaussian kernel row
			for(unsigned long m = 0; m < N; m++) P[n * N + m] = exp(-beta * DD[n * N + m]);
			P[n * N + n] = DBL_MIN;
			
			// Compute entropy of current row
			sum_P = DBL_MIN;
			for(unsigned long m = 0; m < N; m++) sum_P += P[n * N + m];
			double H = 0.0;
			for(unsigned long m = 0; m < N; m++) H += beta * (DD[n * N + m] * P[n * N + m]);
			H = (H / sum_P) + log(sum_P);
			
			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if(Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if(Hdiff > 0) {
					min_beta = beta;
					if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}
			
			// Update iteration counter
			iter++;
		}
		
		// Row normalize P
		for(unsigned long m = 0; m < N; m++) P[n * N + m] /= sum_P;
	}
	
	// Clean up memory
	if (!distance_precomputed) { free(DD); }
	DD = NULL;
}


// Compute input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
void TSNE::computeGaussianPerplexity(double* X, int N, int D, int** _row_P, int** _col_P, double** _val_P, double perplexity, int K, bool verbose, bool distance_precomputed, 
        bool neighbors_precomputed, const int* nndex, const double* nndist) {
    
    if(perplexity > K) Rprintf("Perplexity should be lower than K!\n");
    
    // Allocate the memory we need
    *_row_P = (int*)    malloc((N + 1) * sizeof(int));
    *_col_P = (int*)    calloc(N * K, sizeof(int));
    *_val_P = (double*) calloc(N * K, sizeof(double));
    if(*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    int* row_P = *_row_P;
    int* col_P = *_col_P;
    double* val_P = *_val_P;
    double* cur_P = (double*) malloc((N - 1) * sizeof(double));
    if(cur_P == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    row_P[0] = 0;
    for(int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + K;    
    
    // Build ball tree on data set
    if (neighbors_precomputed) {
      const int* indices=nndex;
      const double * distances=nndist;
      for (int n=0; n<N; n++, indices+=K, distances+=K) {

        if(n % 10000 == 0 && verbose) Rprintf(" - point %d of %d\n", n, N);

        // Initialize some variables for binary search
        bool found = false;
        double beta = 1.0;
        double min_beta = -DBL_MAX;
        double max_beta =  DBL_MAX;
        double tol = 1e-5;
        
        // Iterate until we found a good perplexity
        int iter = 0; double sum_P;
        while(!found && iter < 200) {
          
          // Compute Gaussian kernel row (no +1 here, as 'indices' does not have self as a neighbor).
          for(int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m] * distances[m]);
          
          // Compute entropy of current row
          sum_P = DBL_MIN;
          for(int m = 0; m < K; m++) sum_P += cur_P[m];
          double H = .0;
          for(int m = 0; m < K; m++) H += beta * (distances[m] * distances[m] * cur_P[m]);
          H = (H / sum_P) + log(sum_P);
          
          // Evaluate whether the entropy is within the tolerance level
          double Hdiff = H - log(perplexity);
          if(Hdiff < tol && -Hdiff < tol) {
            found = true;
          }
          else {
            if(Hdiff > 0) {
              min_beta = beta;
              if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
                beta *= 2.0;
              else
                beta = (beta + max_beta) / 2.0;
            }
            else {
              max_beta = beta;
              if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
                beta /= 2.0;
              else
                beta = (beta + min_beta) / 2.0;
            }
          }
          
          // Update iteration counter
          iter++;
        }
        
        // Row-normalize current row of P and store in matrix
        for(int m = 0; m < K; m++) cur_P[m] /= sum_P;
        for(int m = 0; m < K; m++) {
          col_P[row_P[n] + m] = indices[m];
          val_P[row_P[n] + m] = cur_P[m];
        }
      }

    } else if (distance_precomputed) {
      VpTree<DataPoint, precomputed_distance>* tree = new VpTree<DataPoint, precomputed_distance>();
      vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
      for(int n = 0; n < N; n++) obj_X[n] = DataPoint(D, n, X + n * D);
      tree->create(obj_X);
      
      // Loop over all points to find nearest neighbors
      if (verbose) Rprintf("Building tree...\n");
      vector<DataPoint> indices;
      vector<double> distances;
      for(int n = 0; n < N; n++) {
        
        if(n % 10000 == 0 && verbose) Rprintf(" - point %d of %d\n", n, N);
        
        // Find nearest neighbors
        indices.clear();
        distances.clear();
        tree->search(obj_X[n], K + 1, &indices, &distances);
        
        // Initialize some variables for binary search
        bool found = false;
        double beta = 1.0;
        double min_beta = -DBL_MAX;
        double max_beta =  DBL_MAX;
        double tol = 1e-5;
        
        // Iterate until we found a good perplexity
        int iter = 0; double sum_P;
        while(!found && iter < 200) {
          
          // Compute Gaussian kernel row
          for(int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m + 1] * distances[m + 1]);
          
          // Compute entropy of current row
          sum_P = DBL_MIN;
          for(int m = 0; m < K; m++) sum_P += cur_P[m];
          double H = .0;
          for(int m = 0; m < K; m++) H += beta * (distances[m + 1] * distances[m + 1] * cur_P[m]);
          H = (H / sum_P) + log(sum_P);
          
          // Evaluate whether the entropy is within the tolerance level
          double Hdiff = H - log(perplexity);
          if(Hdiff < tol && -Hdiff < tol) {
            found = true;
          }
          else {
            if(Hdiff > 0) {
              min_beta = beta;
              if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
                beta *= 2.0;
              else
                beta = (beta + max_beta) / 2.0;
            }
            else {
              max_beta = beta;
              if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
                beta /= 2.0;
              else
                beta = (beta + min_beta) / 2.0;
            }
          }
          
          // Update iteration counter
          iter++;
        }
        
        // Row-normalize current row of P and store in matrix
        for(int m = 0; m < K; m++) cur_P[m] /= sum_P;
        for(int m = 0; m < K; m++) {
          col_P[row_P[n] + m] = indices[m + 1].index();
          val_P[row_P[n] + m] = cur_P[m];
        }
      }
      
      // Clean up memory
      obj_X.clear();
      free(cur_P);
      delete tree;
    } else {
      VpTree<DataPoint, euclidean_distance>* tree =  new VpTree<DataPoint, euclidean_distance>();
      vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
      for(int n = 0; n < N; n++) obj_X[n] = DataPoint(D, n, X + n * D);
      tree->create(obj_X);
      
      // Loop over all points to find nearest neighbors
      if (verbose) Rprintf("Building tree...\n");
      vector<DataPoint> indices;
      vector<double> distances;
      for(int n = 0; n < N; n++) {
        
        if(n % 10000 == 0 && verbose) Rprintf(" - point %d of %d\n", n, N);
        
        // Find nearest neighbors
        indices.clear();
        distances.clear();
        tree->search(obj_X[n], K + 1, &indices, &distances);
        
        // Initialize some variables for binary search
        bool found = false;
        double beta = 1.0;
        double min_beta = -DBL_MAX;
        double max_beta =  DBL_MAX;
        double tol = 1e-5;
        
        // Iterate until we found a good perplexity
        int iter = 0; double sum_P;
        while(!found && iter < 200) {
          
          // Compute Gaussian kernel row
          for(int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m + 1] *distances[m + 1]);
          
          // Compute entropy of current row
          sum_P = DBL_MIN;
          for(int m = 0; m < K; m++) sum_P += cur_P[m];
          double H = .0;
          for(int m = 0; m < K; m++) H += beta * (distances[m + 1] * distances[m + 1] * cur_P[m]);
          H = (H / sum_P) + log(sum_P);
          
          // Evaluate whether the entropy is within the tolerance level
          double Hdiff = H - log(perplexity);
          if(Hdiff < tol && -Hdiff < tol) {
            found = true;
          }
          else {
            if(Hdiff > 0) {
              min_beta = beta;
              if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
                beta *= 2.0;
              else
                beta = (beta + max_beta) / 2.0;
            }
            else {
              max_beta = beta;
              if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
                beta /= 2.0;
              else
                beta = (beta + min_beta) / 2.0;
            }
          }
          
          // Update iteration counter
          iter++;
        }
        
        // Row-normalize current row of P and store in matrix
        for(int m = 0; m < K; m++) cur_P[m] /= sum_P;
        for(int m = 0; m < K; m++) {
          col_P[row_P[n] + m] = indices[m + 1].index();
          val_P[row_P[n] + m] = cur_P[m];
        }
      }
      
      // Clean up memory
      obj_X.clear();
      free(cur_P);
      delete tree;
    }
}


// Compute input similarities with a fixed perplexity (this function allocates memory another function should free)
void TSNE::computeGaussianPerplexity(double* X, int N, int D, int** _row_P, int** _col_P, double** _val_P, double perplexity, double threshold) {
    
    // Allocate some memory we need for computations
    double* buff  = (double*) malloc(D * sizeof(double));
    double* DD    = (double*) malloc(N * sizeof(double));
    double* cur_P = (double*) malloc(N * sizeof(double));
    if(buff == NULL || DD == NULL || cur_P == NULL) { Rcpp::stop("Memory allocation failed!\n"); }

    // Compute the Gaussian kernel row by row (to find number of elements in sparse P)
    int total_count = 0;
	for(int n = 0; n < N; n++) {
    
        // Compute the squared Euclidean distance matrix
        for(int m = 0; m < N; m++) {
            for(int d = 0; d < D; d++) buff[d]  = X[n * D + d];
            for(int d = 0; d < D; d++) buff[d] -= X[m * D + d];
            DD[m] = .0;
            for(int d = 0; d < D; d++) DD[m] += buff[d] * buff[d];
        }
	   
		// Initialize some variables
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta =  DBL_MAX;
		double tol = 1e-5;
		
		// Iterate until we found a good perplexity
		int iter = 0; double sum_P;
		while(!found && iter < 200) {
			
			// Compute Gaussian kernel row
			for(int m = 0; m < N; m++) cur_P[m] = exp(-beta * DD[m]);
			cur_P[n] = DBL_MIN;
			
			// Compute entropy of current row
			sum_P = DBL_MIN;
			for(int m = 0; m < N; m++) sum_P += cur_P[m];
			double H = 0.0;
			for(int m = 0; m < N; m++) H += beta * (DD[m] * cur_P[m]);
			H = (H / sum_P) + log(sum_P);
			
			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if(Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if(Hdiff > 0) {
					min_beta = beta;
					if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}
			
			// Update iteration counter
			iter++;
		}
		
		// Row-normalize and threshold current row of P
        for(int m = 0; m < N; m++) cur_P[m] /= sum_P;
        for(int m = 0; m < N; m++) {
            if(cur_P[m] > threshold / (double) N) total_count++;
        }
    }
    
    // Allocate the memory we need
    *_row_P = (int*)    malloc((N + 1)     * sizeof(int));
    *_col_P = (int*)    malloc(total_count * sizeof(int));
    *_val_P = (double*) malloc(total_count * sizeof(double));
    int* row_P = *_row_P;
    int* col_P = *_col_P;
    double* val_P = *_val_P;
    if(row_P == NULL || col_P == NULL || val_P == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    row_P[0] = 0;
    
    // Compute the Gaussian kernel row by row (this time, store the results)
    int count = 0;
	for(int n = 0; n < N; n++) {
        
        // Compute the squared Euclidean distance matrix
        for(int m = 0; m < N; m++) {
            for(int d = 0; d < D; d++) buff[d]  = X[n * D + d];
            for(int d = 0; d < D; d++) buff[d] -= X[m * D + d];
            DD[m] = .0;
            for(int d = 0; d < D; d++) DD[m] += buff[d] * buff[d];
        }
        
		// Initialize some variables
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta =  DBL_MAX;
		double tol = 1e-5;
        
		// Iterate until we found a good perplexity
		int iter = 0; double sum_P;
		while(!found && iter < 200) {
			
			// Compute Gaussian kernel row
			for(int m = 0; m < N; m++) cur_P[m] = exp(-beta * DD[m]);
			cur_P[n] = DBL_MIN;
			
			// Compute entropy of current row
			sum_P = DBL_MIN;
			for(int m = 0; m < N; m++) sum_P += cur_P[m];
			double H = 0.0;
			for(int m = 0; m < N; m++) H += beta * (DD[m] * cur_P[m]);
			H = (H / sum_P) + log(sum_P);
			
			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if(Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if(Hdiff > 0) {
					min_beta = beta;
					if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}
			
			// Update iteration counter
			iter++;
		}
		
		// Row-normalize and threshold current row of P
		for(int m = 0; m < N; m++) cur_P[m] /= sum_P;
        for(int m = 0; m < N; m++) {
            if(cur_P[m] > threshold / (double) N) {
                col_P[count] = m;
                val_P[count] = cur_P[m];
                count++;
            }
        }
        row_P[n + 1] = count;
	}
    
    // Clean up memory
    free(DD);    DD = NULL;
    free(buff);  buff = NULL;
    free(cur_P); cur_P = NULL;
}


void TSNE::symmetrizeMatrix(int** _row_P, int** _col_P, double** _val_P, int N) {
    
    // Get sparse matrix
    int* row_P = *_row_P;
    int* col_P = *_col_P;
    double* val_P = *_val_P;

    // Count number of elements and row counts of symmetric matrix
    int* row_counts = (int*) calloc(N, sizeof(int));
    if(row_counts == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    for(int n = 0; n < N; n++) {
        for(int i = row_P[n]; i < row_P[n + 1]; i++) {
            
            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if(col_P[m] == n) present = true;
            }
            if(present) row_counts[n]++;
            else {
                row_counts[n]++;
                row_counts[col_P[i]]++;
            }
        }
    }
    int no_elem = 0;
    for(int n = 0; n < N; n++) no_elem += row_counts[n];
    
    // Allocate memory for symmetrized matrix
    int*    sym_row_P = (int*)    malloc((N + 1) * sizeof(int));
    int*    sym_col_P = (int*)    malloc(no_elem * sizeof(int));
    double* sym_val_P = (double*) malloc(no_elem * sizeof(double));
    if(sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    
    // Construct new row indices for symmetric matrix
    sym_row_P[0] = 0;
    for(int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + row_counts[n];
    
    // Fill the result matrix
    int* offset = (int*) calloc(N, sizeof(int));
    if(offset == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    for(int n = 0; n < N; n++) {
        for(int i = row_P[n]; i < row_P[n + 1]; i++) {                                  // considering element(n, col_P[i])
            
            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if(col_P[m] == n) {
                    present = true;
                    if(n <= col_P[i]) {                                                 // make sure we do not add elements twice
                        sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                        sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                        sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i] + val_P[m];
                        sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
                    }
                }
            }
            
            // If (col_P[i], n) is not present, there is no addition involved
            if(!present) {
                sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i];
                sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
            }
            
            // Update offsets
            if(!present || (present && n <= col_P[i])) {
                offset[n]++;
                if(col_P[i] != n) offset[col_P[i]]++;               
            }
        }
    }
    
    // Divide the result by two
    for(int i = 0; i < no_elem; i++) sym_val_P[i] /= 2.0;
    
    // Return symmetrized matrices
    free(*_row_P); *_row_P = sym_row_P;
    free(*_col_P); *_col_P = sym_col_P;
    free(*_val_P); *_val_P = sym_val_P;
    
    // Free up some memery
    free(offset); offset = NULL;
    free(row_counts); row_counts  = NULL;
}

// Compute squared Euclidean distance matrix (using BLAS)
void TSNE::computeSquaredEuclideanDistance(double* X, int N, int D, double* DD) {
    double* dataSums = (double*) calloc(N, sizeof(double));
    if(dataSums == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    for(int n = 0; n < N; n++) {
        for(int d = 0; d < D; d++) {
            dataSums[n] += (X[n * D + d] * X[n * D + d]);
        }
    }
    for(unsigned long n = 0; n < N; n++) {
        for(unsigned long m = 0; m < N; m++) {
            DD[n * N + m] = dataSums[n] + dataSums[m];
        }
    }
    double a1 = -2.0;
    double a2 = 1.0;
    dgemm_("T", "N", &N, &N, &D, &a1, X, &D, X, &D, &a2, DD, &N);
    free(dataSums); dataSums = NULL;
}


// Makes data zero-mean
void TSNE::zeroMean(double* X, int N, int D) {
	
	// Compute data mean
	double* mean = (double*) calloc(D, sizeof(double));
    if(mean == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			mean[d] += X[n * D + d];
		}
	}
	for(int d = 0; d < D; d++) {
		mean[d] /= (double) N;
	}
	
	// Subtract data mean
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			X[n * D + d] -= mean[d];
		}
	}
    free(mean); mean = NULL;
}


// Generates a Gaussian random number
double TSNE::randn() {
  Rcpp::RNGScope scope;
	double x, y, radius;
	do {
		x = 2 * (double)R::runif(0,1) - 1;
		y = 2 * (double)R::runif(0,1) - 1;
		radius = (x * x) + (y * y);
	} while((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
}

// Function that loads data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
bool TSNE::load_data(double** data, int* n, int* d, int* no_dims, double* theta, double* perplexity, int* rand_seed) {
	
	// Open file, read first 2 integers, allocate memory, and read the data
    FILE *h;
	if((h = fopen("data.dat", "r+b")) == NULL) {
		Rprintf("Error: could not open data file.\n");
		return false;
	}
	fread(n, sizeof(int), 1, h);											// number of datapoints
	fread(d, sizeof(int), 1, h);											// original dimensionality
    fread(theta, sizeof(double), 1, h);										// gradient accuracy
	fread(perplexity, sizeof(double), 1, h);								// perplexity
  fread(no_dims, sizeof(int), 1, h);                                      // output dimensionality
	*data = (double*) calloc(*d * *n, sizeof(double));
    if(*data == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    fread(*data, sizeof(double), *n * *d, h);                               // the data
	if(!feof(h)) fread(rand_seed, sizeof(int), 1, h);                       // random seed
  fclose(h);
	Rprintf("Read the %i x %i data matrix successfully!\n", *n, *d);
	return true;
}

// Function that saves map to a t-SNE file
void TSNE::save_data(double* data, int* landmarks, double* costs, int n, int d) {
    
	// Open file, write first 2 integers and then the data
	FILE *h;
	if((h = fopen("result.dat", "w+b")) == NULL) {
		Rprintf("Error: could not open data file.\n");
		return;
	}
	fwrite(&n, sizeof(int), 1, h);
	fwrite(&d, sizeof(int), 1, h);
    fwrite(data, sizeof(double), n * d, h);
	fwrite(landmarks, sizeof(int), n, h);
    fwrite(costs, sizeof(double), n, h);
    fclose(h);
	Rprintf("Wrote the %i x %i data matrix successfully!\n", n, d);
}
