# BarcodeCalc

Small package for calculating zero persistent homology of sublevel set filtration of a function. 
It is based on the algorithm from article 
<a href="https://arxiv.org/search/?query=Dmitry+Oganesyan&searchtype=all&source=header">Barcodes as summary of objective function's topology</a>.

The algorithm works with function’s values on a randomly sampled or specifically chosen set of points. 
The local minima give birth to clusters of points in sublevel sets.
The algorithm works by looking at neighbors of each point with lower value of the function and deciding if this point 
belongs to the existing clusters, gives birth to a new cluster (minimum), or merges two or more clusters (index one saddle). 
Algorithm has complexity of O(n log(n)), where n is the cardinality of the set of points.

