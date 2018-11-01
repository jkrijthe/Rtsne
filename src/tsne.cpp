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


#ifdef _OPENMP
  #include <omp.h>
#endif

extern "C" {
    #include <R_ext/BLAS.h>
}


using namespace std;

template <int NDims>
TSNE<NDims>::TSNE(double Perplexity, double Theta, bool Verbose, int Max_iter, bool Init, int Stop_lying_iter,
        int Mom_switch_iter, double Momentum, double Final_momentum, double Eta, double Exaggeration_factor, int Num_threads) :
    perplexity(Perplexity), theta(Theta), momentum(Momentum), final_momentum(Final_momentum), eta(Eta), exaggeration_factor(Exaggeration_factor),
    max_iter(Max_iter), stop_lying_iter(Stop_lying_iter), mom_switch_iter(Mom_switch_iter), num_threads(Num_threads),
    verbose(Verbose), init(Init), exact(theta==.0) {

    #ifdef _OPENMP
      int threads = num_threads;
      if (num_threads==0) {
        threads = omp_get_max_threads();
      }
      
      // Print notice whether OpenMP is used
      if (verbose) Rprintf("OpenMP is working. %d threads.\n", threads);
    #endif

    return;
}

// Perform t-SNE
template <int NDims>
void TSNE<NDims>::run(double* X, unsigned int N, int D, double* Y, bool distance_precomputed, double* cost, double* itercost) {
    if(N - 1 < 3 * perplexity) { Rcpp::stop("Perplexity too large for the number of data points!\n"); }
    if (verbose) Rprintf("Using no_dims = %d, perplexity = %f, and theta = %f\n", NDims, perplexity, theta);
    if (verbose) Rprintf("Computing input similarities...\n");
    clock_t start = clock();

    // Compute input similarities for exact t-SNE
    if(exact) {
        // Compute similarities
        computeGaussianPerplexity(X, N, D, distance_precomputed);

        // Symmetrize input similarities
        if (verbose) Rprintf("Symmetrizing...\n");
        for(unsigned long n = 0; n < N; n++) {
            for(unsigned long m = n + 1; m < N; m++) {
                P[n * N + m] += P[m * N + n];
                P[m * N + n]  = P[n * N + m];
            }
        }

        double sum_P = 0;
        for(size_t i = 0; i < P.size(); i++) sum_P += P[i];
        for(size_t i = 0; i < P.size(); i++) P[i] /= sum_P;
    }

    // Compute input similarities for approximate t-SNE
    else {
        int K=3*perplexity;

        // Compute asymmetric pairwise input similarities
        if (distance_precomputed) {
          computeGaussianPerplexity<precomputed_distance>(X, N, D, K);
        } else {
          computeGaussianPerplexity<euclidean_distance>(X, N, D, K);
        }

        // Symmetrize input similarities
        symmetrizeMatrix(N);
        double sum_P = .0;
        for(unsigned int i = 0; i < row_P[N]; i++) sum_P += val_P[i];
        for(unsigned int i = 0; i < row_P[N]; i++) val_P[i] /= sum_P;
    }

    if (verbose) {
        clock_t end = clock();
        if(exact) Rprintf("Done in %4.2f seconds!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC);
        else Rprintf("Done in %4.2f seconds (sparsity = %f)!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC, (double) row_P[N] / ((double) N * (double) N));
    }

    trainIterations(N, Y, cost, itercost);
    return;
}

// Perform t-SNE with nearest neighbor results.
template<int NDims>
void TSNE<NDims>::run(const int* nn_index, const double* nn_dist, unsigned int N, int K, double* Y, double* cost, double* itercost) {
    if(N - 1 < 3 * perplexity) { Rcpp::stop("Perplexity too large for the number of data points!\n"); }
    if (verbose) Rprintf("Using no_dims = %d, perplexity = %f, and theta = %f\n", NDims, perplexity, theta);
    if (verbose) Rprintf("Computing input similarities...\n");
    clock_t start = clock();

    // Compute asymmetric pairwise input similarities
    computeGaussianPerplexity(nn_index, nn_dist, N, K);

    // Symmetrize input similarities
    symmetrizeMatrix(N);
    double sum_P = .0;
    for(unsigned int i = 0; i < row_P[N]; i++) sum_P += val_P[i];
    for(unsigned int i = 0; i < row_P[N]; i++) val_P[i] /= sum_P;

    if (verbose) {
        clock_t end = clock();
        if(exact) Rprintf("Done in %4.2f seconds!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC);
        else Rprintf("Done in %4.2f seconds (sparsity = %f)!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC, (double) row_P[N] / ((double) N * (double) N));
    }

    trainIterations(N, Y, cost, itercost);
    return;
}

// Perform main training loop
template<int NDims>
void TSNE<NDims>::trainIterations(unsigned int N, double* Y, double* cost, double* itercost) {
    // Allocate some memory
    double* dY    = (double*) malloc(N * NDims * sizeof(double));
    double* uY    = (double*) malloc(N * NDims * sizeof(double));
    double* gains = (double*) malloc(N * NDims * sizeof(double));
    if(dY == NULL || uY == NULL || gains == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    for(unsigned int i = 0; i < N * NDims; i++)    uY[i] =  .0;
    for(unsigned int i = 0; i < N * NDims; i++) gains[i] = 1.0;

    // Lie about the P-values
    if(exact) { for(unsigned long i = 0; i < (unsigned long)N * N; i++) P[i] *= exaggeration_factor; }
    else {      for(unsigned long i = 0; i < row_P[N]; i++)    val_P[i] *= exaggeration_factor; }

	// Initialize solution (randomly), if not already done
	if (!init) { for(unsigned int i = 0; i < N * NDims; i++) Y[i] = randn() * .0001; }

  clock_t start = clock(), end;
  float total_time=0;
  int costi = 0; //iterator for saving the total costs for the iterations

	for(int iter = 0; iter < max_iter; iter++) {

        // Stop lying about the P-values after a while, and switch momentum
        if(iter == stop_lying_iter) {
          if(exact) { for(unsigned long i = 0; i < (unsigned long)N * N; i++)        P[i] /= exaggeration_factor; }
          else      { for(unsigned int i = 0; i < row_P[N]; i++) val_P[i] /= exaggeration_factor; }
        }
        if(iter == mom_switch_iter) momentum = final_momentum;

        // Compute (approximate) gradient
        if(exact) computeExactGradient(P.data(), Y, N, NDims, dY);
        else computeGradient(P.data(), row_P.data(), col_P.data(), val_P.data(), Y, N, NDims, dY, theta);

        // Update gains
        for(unsigned int i = 0; i < N * NDims; i++) gains[i] = (sign_tsne(dY[i]) != sign_tsne(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
        for(unsigned int i = 0; i < N * NDims; i++) if(gains[i] < .01) gains[i] = .01;

        // Perform gradient update (with momentum and gains)
        for(unsigned int i = 0; i < N * NDims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
        for(unsigned int i = 0; i < N * NDims; i++)  Y[i] = Y[i] + uY[i];

        // Make solution zero-mean
        zeroMean(Y, N, NDims);

        // Print out progress
        if((iter > 0 && (iter+1) % 50 == 0) || iter == max_iter - 1) {
            end = clock();
            double C = .0;
            if(exact) C = evaluateError(P.data(), Y, N, NDims);
            else      C = evaluateError(row_P.data(), col_P.data(), val_P.data(), Y, N, NDims, theta);  // doing approximate computation here!
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

    if(exact) getCost(P.data(), Y, N, NDims, cost);
    else      getCost(row_P.data(), col_P.data(), val_P.data(), Y, N, NDims, theta, cost);  // doing approximate computation here!
    
    // Clean up memory
    free(dY);
    free(uY);
    free(gains);
    if (verbose) Rprintf("Fitting performed in %4.2f seconds.\n", total_time);
    return;
}

// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
template <int NDims>
void TSNE<NDims>::computeGradient(double* P, unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, unsigned int N, int D, double* dC, double theta)
{
    // Construct space-partitioning tree on current map
    SPTree<NDims>* tree = new SPTree<NDims>(Y, N);

    // Compute all terms required for t-SNE gradient
    double* pos_f = (double*) calloc(N * D, sizeof(double));
    double* neg_f = (double*) calloc(N * D, sizeof(double));
    if(pos_f == NULL || neg_f == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f, num_threads);

    // Storing the output to sum in single-threaded mode; avoid randomness in rounding errors.
    std::vector<double> output(N);

    #pragma omp parallel for schedule(guided) num_threads(num_threads)
    for (unsigned int n = 0; n < N; n++) {
      output[n]=tree->computeNonEdgeForces(n, theta, neg_f + n * D);
    }

    double sum_Q = .0;
    for (unsigned int n=0; n<N; ++n) {
        sum_Q += output[n];
    }

    // Compute final t-SNE gradient
    for(unsigned int i = 0; i < N * D; i++) {
        dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
    }

    free(pos_f);
    free(neg_f);
    delete tree;
}

// Compute gradient of the t-SNE cost function (exact)
template <int NDims>
void TSNE<NDims>::computeExactGradient(double* P, double* Y, unsigned int N, int D, double* dC) {
	
	// Make sure the current gradient contains zeros
	for(unsigned int i = 0; i < N * D; i++) dC[i] = 0.0;

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc((unsigned long)N * N * sizeof(double));
    if(DD == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    computeSquaredEuclideanDistance(Y, N, D, DD);

    // Compute Q-matrix and normalization sum
    double* Q    = (double*) malloc((unsigned long)N * N * sizeof(double));
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
template <int NDims>
double TSNE<NDims>::evaluateError(double* P, double* Y, unsigned int N, int D) {

    // Compute the squared Euclidean distance matrix
    double* DD = (double*) malloc((unsigned long)N * N * sizeof(double));
    double* Q = (double*) malloc((unsigned long)N * N * sizeof(double));
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
    for(unsigned long i = 0; i < (unsigned long)N * N; i++) Q[i] /= sum_Q;

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
template <int NDims>
double TSNE<NDims>::evaluateError(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, unsigned int N, int D, double theta)
{

    // Get estimate of normalization term
    SPTree<NDims>* tree = new SPTree<NDims>(Y, N);
    double* buff = (double*) calloc(D, sizeof(double));
    double sum_Q = .0;
    for(unsigned int n = 0; n < N; n++) sum_Q += tree->computeNonEdgeForces(n, theta, buff);

    // Loop over all edges to compute t-SNE error
    int ind1, ind2;
    double C = .0, Q;
    for(unsigned int n = 0; n < N; n++) {
        ind1 = n * D;
        for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {
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
template <int NDims>
void TSNE<NDims>::getCost(double* P, double* Y, unsigned int N, int D, double* costs) {

  // Compute the squared Euclidean distance matrix
  double* DD = (double*) malloc((unsigned long)N * N * sizeof(double));
  double* Q = (double*) malloc((unsigned long)N * N * sizeof(double));
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
  for(unsigned long i = 0; i < (unsigned long)N * N; i++) Q[i] /= sum_Q;

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
template <int NDims>
void TSNE<NDims>::getCost(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, unsigned int N, int D, double theta, double* costs)
{

  // Get estimate of normalization term
  SPTree<NDims>* tree = new SPTree<NDims>(Y, N);
  double* buff = (double*) calloc(D, sizeof(double));
  double sum_Q = .0;
  for(unsigned int n = 0; n < N; n++) sum_Q += tree->computeNonEdgeForces(n, theta, buff);

  // Loop over all edges to compute t-SNE error
  int ind1, ind2;
  double  Q;
  for(unsigned int n = 0; n < N; n++) {
    ind1 = n * D;
    costs[n] = 0.0;
    for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {
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
template <int NDims>
void TSNE<NDims>::computeGaussianPerplexity(double* X, unsigned int N, int D, bool distance_precomputed) {
    size_t N2=N;
    N2*=N;
    P.resize(N2);

	// Compute the squared Euclidean distance matrix
	double* DD = (double*) malloc(N2 * sizeof(double));
  if(DD == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
	
	if (distance_precomputed) {
	  DD = X;
	} else {
	  computeSquaredEuclideanDistanceDirect(X, N, D, DD);
	  
	  // Needed to make sure the results are exactly equal to distance calculation in R
	  for (size_t n=0; n<N*N; n++) {
	    DD[n] = sqrt(DD[n])*sqrt(DD[n]);
	  }
	}

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
template <int NDims>
template<double (*T)( const DataPoint&, const DataPoint& )>
void TSNE<NDims>::computeGaussianPerplexity(double* X, unsigned int N, int D, int K) {

    if(perplexity > K) Rprintf("Perplexity should be lower than K!\n");

    // Allocate the memory we need
    setupApproximateMemory(N, K);

    // Build ball tree on data set
      VpTree<DataPoint, T>* tree = new VpTree<DataPoint, T>();
      vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
      for(unsigned int n = 0; n < N; n++) obj_X[n] = DataPoint(D, n, X + n * D);
      tree->create(obj_X);

      // Loop over all points to find nearest neighbors
      if (verbose) Rprintf("Building tree...\n");

      int steps_completed = 0;
      #pragma omp parallel for schedule(guided) num_threads(num_threads)
      for(unsigned int n = 0; n < N; n++) {

        vector<DataPoint> indices;
        vector<double> distances;
        indices.reserve(K+1);
        distances.reserve(K+1);

        // Find nearest neighbors
        tree->search(obj_X[n], K + 1, &indices, &distances);

        double * cur_P = val_P.data() + row_P[n];
        computeProbabilities(perplexity, K, distances.data()+1, cur_P); // +1 to avoid self.

        unsigned int * cur_col_P = col_P.data() + row_P[n];
        for (int m=0; m<K; ++m) {
            cur_col_P[m] = indices[m+1].index(); // +1 to avoid self.
        }

        #pragma omp atomic
        ++steps_completed;

        if (verbose) { 
          if(steps_completed % 10000 == 0) Rprintf(" - point %d of %d\n", steps_completed, N);
        }
      }

      // Clean up memory
      obj_X.clear();
      delete tree;
}

// Compute input similarities with a fixed perplexity from nearest-neighbour results.
template<int NDims>
void TSNE<NDims>::computeGaussianPerplexity(const int* nn_idx, const double* nn_dist, unsigned int N, int K) {

    if(perplexity > K) Rprintf("Perplexity should be lower than K!\n");

    // Allocate the memory we need
    setupApproximateMemory(N, K);

    // Loop over all points to find nearest neighbors
    int steps_completed = 0;
    #pragma omp parallel for schedule(guided) num_threads(num_threads)
    for(unsigned int n = 0; n < N; n++) {
      double * cur_P = val_P.data() + row_P[n];
      computeProbabilities(perplexity, K, nn_dist + row_P[n], cur_P);

      const int * cur_idx = nn_idx + row_P[n];
      unsigned int * cur_col_P = col_P.data() + row_P[n];
      for (int m=0; m<K; ++m) {
          cur_col_P[m] = cur_idx[m];
      }

      #pragma omp atomic
      ++steps_completed;

      if (verbose) { 
        if(steps_completed % 10000 == 0) Rprintf(" - point %d of %d\n", steps_completed, N);
      }
    }
}

template<int NDims>
void TSNE<NDims>::setupApproximateMemory(unsigned int N, int K) {
    row_P.resize(N+1);
    col_P.resize(N*K);
    val_P.resize(N*K);
    row_P[0] = 0;
    for(unsigned int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + K;
    return;
}

template<int NDims>
void TSNE<NDims>::computeProbabilities (const double perplexity, const int K, const double* distances, double* cur_P) {

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

    // Row-normalize current row of P.
    for(int m = 0; m < K; m++) {
      cur_P[m] /= sum_P;
    }
    return;
}

template <int NDims>
void TSNE<NDims>::symmetrizeMatrix(unsigned int N) {
    // Count number of elements and row counts of symmetric matrix
    int* row_counts = (int*) calloc(N, sizeof(int));
    if(row_counts == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    for(unsigned int n = 0; n < N; n++) {
        for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
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
    for(unsigned int n = 0; n < N; n++) no_elem += row_counts[n];

    // Allocate memory for symmetrized matrix
    std::vector<unsigned int> sym_row_P(N+1), sym_col_P(no_elem);
    std::vector<double> sym_val_P(no_elem);

    // Construct new row indices for symmetric matrix
    sym_row_P[0] = 0;
    for(unsigned int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + row_counts[n];

    // Fill the result matrix
    int* offset = (int*) calloc(N, sizeof(int));
    if(offset == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    for(unsigned int n = 0; n < N; n++) {
        for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {                                  // considering element(n, col_P[i])

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
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
    row_P.swap(sym_row_P);
    col_P.swap(sym_col_P);
    val_P.swap(sym_val_P);

    // Free up some memery
    free(offset); offset = NULL;
    free(row_counts); row_counts  = NULL;
}

// Compute squared Euclidean distance matrix (using BLAS)
template <int NDims>
void TSNE<NDims>::computeSquaredEuclideanDistance(double* X, unsigned int N, int D, double* DD) {
    double* dataSums = (double*) calloc(N, sizeof(double));
    if(dataSums == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    for(unsigned int n = 0; n < N; n++) {
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
    int Nsigned = N;
    dgemm_("T", "N", &Nsigned, &Nsigned, &D, &a1, X, &D, X, &D, &a2, DD, &Nsigned);
    free(dataSums); dataSums = NULL;
}

// Compute squared Euclidean distance matrix (No BLAS)
template <int NDims>
void TSNE<NDims>::computeSquaredEuclideanDistanceDirect(double* X, unsigned int N, int D, double* DD) {
  const double* XnD = X;
  for(unsigned int n = 0; n < N; ++n, XnD += D) {
    const double* XmD = XnD + D;
    double* curr_elem = &DD[n*N + n];
    *curr_elem = 0.0;
    double* curr_elem_sym = curr_elem + N;
    for(unsigned int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
      *(++curr_elem) = 0.0;
      for(int d = 0; d < D; ++d) {
        *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
      }
      *curr_elem_sym = *curr_elem;
    }
  }
}


// Makes data zero-mean
template <int NDims>
void TSNE<NDims>::zeroMean(double* X, unsigned int N, int D) {
	
	// Compute data mean
	double* mean = (double*) calloc(D, sizeof(double));
  if(mean == NULL) {  Rcpp::stop("Memory allocation failed!\n"); }
  int nD = 0;
  for(unsigned int n = 0; n < N; n++) {
    for(int d = 0; d < D; d++) {
      mean[d] += X[nD + d];
    }
    nD += D;
  }
  for(int d = 0; d < D; d++) {
    mean[d] /= (double) N;
  }

  // Subtract data mean
  nD = 0;
  for(unsigned int n = 0; n < N; n++) {
    for(int d = 0; d < D; d++) {
      X[nD + d] -= mean[d];
    }
    nD += D;
  }
  free(mean); mean = NULL;
}

// Generates a Gaussian random number
template <int NDims>
double TSNE<NDims>::randn() {
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

// declare templates explicitly
template class TSNE<1>;
template class TSNE<2>;
template class TSNE<3>;
