#include <Rcpp.h>
#include "tsne.h"
using namespace Rcpp;

// Function that runs the Barnes-Hut implementation of t-SNE
// [[Rcpp::export]]
Rcpp::List Rtsne_cpp(SEXP X_in, int no_dims_in, double perplexity_in, double theta_in) {

  Rcpp::NumericMatrix X(X_in); 

  int origN, N, D, no_dims = no_dims_in;

	double  *data;
  TSNE* tsne = new TSNE();
  double perplexity = perplexity_in;
  double theta = theta_in;

  origN = X.nrow();
  D = X.ncol();
    
	data = (double*) calloc(D * origN, sizeof(double));
    if(data == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    for (int i = 0; i < origN; i++){
        for (int j = 0; j < D; j++){
            data[i*D+j] = X(i,j);
        }
    }
  
    // Set random seed (now done using R's RNG)
//        if(rand_seed >= 0) {
//            Rprintf("Using random seed: %d\n", rand_seed);
//            srand((unsigned int) rand_seed);
//        }
//        else {
//            Rprintf("Using current time as random seed...\n");
//            srand(time(NULL));
//        } 
    
    // Make dummy landmarks
    N = origN;
    Rprintf("Read the %i x %i data matrix successfully!\n", N, D);
    int* landmarks = (int*) malloc(N * sizeof(int));
    if(landmarks == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    for(int n = 0; n < N; n++) landmarks[n] = n;

		double* Y = (double*) malloc(N * no_dims * sizeof(double));
		double* costs = (double*) calloc(N, sizeof(double));
    if(Y == NULL || costs == NULL) { Rcpp::stop("Memory allocation failed!\n"); }
    
    // Run tsne
		tsne->run(data, N, D, Y, no_dims, perplexity, theta);

  	// Save the results
    Rcpp::NumericMatrix Yr(N,no_dims);
    for (int i = 0; i < N; i++){
        for (int j = 0; j < no_dims; j++){
            Yr(i,j) = Y[i*no_dims+j];
        }
    }
    
    Rcpp::NumericVector costsr(N);
    for (int i = 0; i < N; i++){
      costsr(i) = costs[i];
    }
    
    free(data); data = NULL;
  	free(Y); Y = NULL;
		free(costs); costs = NULL;
		free(landmarks); landmarks = NULL;
    delete(tsne);
    
    Rcpp::List output = Rcpp::List::create(Rcpp::_["theta"]=theta, Rcpp::_["perplexity"]=perplexity, Rcpp::_["N"]=N,Rcpp::_["origD"]=D,Rcpp::_["Y"]=Yr, Rcpp::_["costs"]=costsr);
    return output; 
}
