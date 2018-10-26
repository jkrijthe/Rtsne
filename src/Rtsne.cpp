#include <Rcpp.h>
#include "tsne.h"
using namespace Rcpp;

Rcpp::List save_results(int N, int no_dims, const std::vector<double>& Y, const std::vector<double>& costs, const std::vector<double>& itercosts,
        double theta, double perplexity, int D, int stop_lying_iter, int mom_switch_iter, 
        double momentum, double final_momentum, double eta, double exaggeration_factor);

// Function that runs the Barnes-Hut implementation of t-SNE
// [[Rcpp::export]]
Rcpp::List Rtsne_cpp(NumericMatrix X, int no_dims, double perplexity, 
                     double theta, bool verbose, int max_iter, 
                     bool distance_precomputed, NumericMatrix Y_in, bool init, 
                     int stop_lying_iter, int mom_switch_iter,
                     double momentum, double final_momentum, 
                     double eta, double exaggeration_factor, unsigned int num_threads) {

    size_t N = X.ncol(), D = X.nrow();
    double * data=X.begin();
    
    if (verbose) Rprintf("Read the %i x %i data matrix successfully!\n", N, D);
    std::vector<double> Y(N * no_dims), costs(N), itercosts(static_cast<int>(std::ceil(max_iter/50.0)));
  
    // Providing user-supplied solution.
    if (init) {
        for (size_t i = 0; i < Y.size(); i++) Y[i] = Y_in[i];
        if (verbose) Rprintf("Using user supplied starting positions\n");
    }
    
    // Run tsne
    if (no_dims==1) {
      TSNE<1> tsne(perplexity, theta, verbose, max_iter, init, stop_lying_iter, mom_switch_iter, 
              momentum, final_momentum, eta, exaggeration_factor, num_threads);
      tsne.run(data, N, D, Y.data(), distance_precomputed, costs.data(), itercosts.data());
    } else if (no_dims==2) {
      TSNE<2> tsne(perplexity, theta, verbose, max_iter, init, stop_lying_iter, mom_switch_iter, 
              momentum, final_momentum, eta, exaggeration_factor, num_threads);
      tsne.run(data, N, D, Y.data(), distance_precomputed, costs.data(), itercosts.data());
    } else if (no_dims==3) {
      TSNE<3> tsne(perplexity, theta, verbose, max_iter, init, stop_lying_iter, mom_switch_iter, 
              momentum, final_momentum, eta, exaggeration_factor, num_threads);
      tsne.run(data, N, D, Y.data(), distance_precomputed, costs.data(), itercosts.data());
    } else {
      Rcpp::stop("Only 1, 2 or 3 dimensional output is suppported.\n");
    }

    return Rcpp::List::create(Rcpp::_["Y"]=Rcpp::NumericMatrix(no_dims, N, Y.data()), 
            Rcpp::_["costs"]=Rcpp::NumericVector(costs.begin(), costs.end()),
            Rcpp::_["itercosts"]=Rcpp::NumericVector(itercosts.begin(), itercosts.end()));
}

// Function that runs the Barnes-Hut implementation of t-SNE on nearest neighbor results.
// [[Rcpp::export]]
Rcpp::List Rtsne_nn_cpp(IntegerMatrix nn_dex, NumericMatrix nn_dist, 
                     int no_dims, double perplexity, 
                     double theta, bool verbose, int max_iter, 
                     NumericMatrix Y_in, bool init, 
                     int stop_lying_iter, int mom_switch_iter,
                     double momentum, double final_momentum, 
                     double eta, double exaggeration_factor, unsigned int num_threads) {

    size_t N = nn_dex.ncol(), K=nn_dex.nrow(); // transposed - columns are points, rows are neighbors.
    if (verbose) Rprintf("Read the NN results for %i points successfully!\n", N);
    std::vector<double> Y(N * no_dims), costs(N), itercosts(static_cast<int>(std::ceil(max_iter/50.0)));
  
    // Providing user-supplied solution.
    if (init) {
        for (size_t i = 0; i < Y.size(); i++) Y[i] = Y_in[i];
        if (verbose) Rprintf("Using user supplied starting positions\n");
    }
    
    // Run tsne
    if (no_dims==1) {
      TSNE<1> tsne(perplexity, theta, verbose, max_iter, init, stop_lying_iter, mom_switch_iter, 
              momentum, final_momentum, eta, exaggeration_factor, num_threads);
      tsne.run(nn_dex.begin(), nn_dist.begin(), N, K, Y.data(), costs.data(), itercosts.data());
    } else if (no_dims==2) {
      TSNE<2> tsne(perplexity, theta, verbose, max_iter, init, stop_lying_iter, mom_switch_iter, 
              momentum, final_momentum, eta, exaggeration_factor, num_threads);
      tsne.run(nn_dex.begin(), nn_dist.begin(), N, K, Y.data(), costs.data(), itercosts.data());
    } else if (no_dims==3) {
      TSNE<3> tsne(perplexity, theta, verbose, max_iter, init, stop_lying_iter, mom_switch_iter, 
              momentum, final_momentum, eta, exaggeration_factor, num_threads);
      tsne.run(nn_dex.begin(), nn_dist.begin(), N, K, Y.data(), costs.data(), itercosts.data());
    } else {
      Rcpp::stop("Only 1, 2 or 3 dimensional output is suppported.\n");
    }

    return Rcpp::List::create(Rcpp::_["Y"]=Rcpp::NumericMatrix(no_dims, N, Y.data()), 
            Rcpp::_["costs"]=Rcpp::NumericVector(costs.begin(), costs.end()),
            Rcpp::_["itercosts"]=Rcpp::NumericVector(itercosts.begin(), itercosts.end()));
}

