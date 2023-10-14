# About Principle Component Reduction

[Principal component analysis, wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)  

## Intuitions of PCA

[Principal component analysis, wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)  
Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components.  
If there are $n$ observations with $p$ variables, then the number of distinct principal components is $min (n−1 , p)$. This transformation is defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components.  
The resulting vectors (each being a linear combination of the variables and containing n observations) are an uncorrelated orthogonal basis set. PCA is sensitive to the relative scaling of the original variables.  


## Steps of PCA Algorithm

1. subtract the mean of each variable from the dataset to center the data around the origin.
2. compute the covariance matrix of the data,
3. calculate the eigenvalues and corresponding eigenvectors of this covariance matrix.
4. normalize each of the orthogonal eigenvectors to become unit vectors
5. 

## Backgrounds of PCA

### Orthogonal transformation (正交變換)

[Orthogonal transformation, wikipedia](https://en.wikipedia.org/wiki/Orthogonal_transformation)  

### Orthogonal basis

[Orthogonal basis, wikipedia](https://en.wikipedia.org/wiki/Orthogonal_basis)  

### Singular value decomposition

[Singular value decomposition, wikipedia](https://en.wikipedia.org/wiki/Singular_value_decomposition)  

## Appendix: References

+ [sklearn.decomposition.PCA, sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
