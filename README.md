# R wrapper for Van der Maaten's Barnes-Hut implementation of t-Distributed Stochastic Neighbor Embedding 

##Usage
To use this package use the following code in R to install the package and run a simple example.

```{R}
install.packages("Rtsne") # Install Rtsne package from CRAN
library(Rtsne) # Load package
iris_unique <- unique(iris) # Remove duplicates
tsne_out <- Rtsne(as.matrix(iris_unique[,1:4])) # Run TSNE
plot(tsne_out$Y,col=iris$Species) # Plot the result
```

To install the latest version from the github repository, use:
```{R}
install.packages("devtools") # If not already installed
library(devtools)
install_github("Rtsne","jkrijthe")
```

#Details
This R package offers a wrapper around the Barnes-Hut TSNE C++ implementation of [2] [3]. Only minor changes were made to the original code to allow it to function as an R package.

#References
[1] L.J.P. van der Maaten and G.E. Hinton. Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning Research 9(Nov):2579-2605, 2008.

[2] L.J.P. van der Maaten. Barnes-Hut-SNE. In Proceedings of the International Conference on Learning Representations, 2013.

[3] http://homepage.tudelft.nl/19j49/t-SNE.html