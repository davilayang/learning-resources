# About Eigenvalue and Eigenvector

## Eigenvalue and Eigenvector

[Eigenvalues and eigenvectors, wikipedia](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors)  
&nbsp;&nbsp;&nbsp;&nbsp; In linear algebra, an eigenvector or characteristic vector of a linear transformation is a non-zero vector that _changes by only a scalar factor when that linear transformation is applied to it_. More formally, if $T$ is a linear transformation from a vector space $$ over a field $F$ into itself and $v$ is a vector in $V$ that is not the zero vector, then $v$ is an eigenvector of $T$ if $T(v)$ is a scalar multiple of $v$. This condition can be written as the equation  

$$ T(v) = \lambda v $$

where $\lambda$ is a scalar in the field $F$, known as the eigenvalue, characteristic value, or characteristic root associated with the eigenvector v.  

## Background Information

### Identity Matrix

&nbsp;&nbsp;&nbsp;&nbsp; The $n\times n$ identity matrix, denoted $I_n$, is a matrix with $n$ rows and $n$ columns. **The entries on the diagonal from the upper left to the bottom right are all $1$'s**, and all other entries are $0$.  
&nbsp;&nbsp;&nbsp;&nbsp; The identity matrix plays a similar role in operations with matrices as the number 11 plays in operations with real numbers.  

+ $I_3 = \begin{bmatrix}1&0&0 \\0&1&0 \\0&0&1 \end{bmatrix}$, Example Matrix $A_{3 \times 3} = \begin{bmatrix}1&2&3 \\4&5&6 \\7&8&9 \end{bmatrix}$
+ Show that $I \cdot A = A$
  + $I \cdot A = \begin{bmatrix}1&0&0 \\0&1&0 \\0&0&1 \end{bmatrix} \times \begin{bmatrix}1&2&3 \\4&5&6 \\7&8&9 \end{bmatrix}$ $= \begin{bmatrix}1×1+0×4+0×7 & 1×2+0×5+0×8 & 1×3+0×6+0×9 \\0×1+1×4+0×7 & 0×2+1×5+0×8 & 0×3+1×6+0×9 \\0×1+0×4+1×7 & 0×2+0×5+1×8 & 0×3+0×6+1×9 \end{bmatrix}$ $= \begin{bmatrix}1&2&3 \\4&5&6 \\7&8&9 \end{bmatrix}$ $= A$
  + _"Fixed row, multiply the columns; then move to next row"_
+ Show that $A \cdot I = A$
  + $A \cdot I = \begin{bmatrix}1&2&3 \\4&5&6 \\7&8&9 \end{bmatrix} \times \begin{bmatrix}1&0&0 \\0&1&0 \\0&0&1 \end{bmatrix} = A$

### Determinant

[Determinant, wikipedia](https://en.wikipedia.org/wiki/Determinant)  
&nbsp;&nbsp;&nbsp;&nbsp; In linear algebra, **the determinant is a value that can be computed from the elements of a square matrix and encodes certain properties of the linear transformation described by the matrix**. The determinant of a matrix A is denoted $\det(A)$, $\det A$, or $|A|$.  
&nbsp;&nbsp;&nbsp;&nbsp; Geometrically, it can be viewed _as the volume scaling factor of the linear transformation described by the matrix_. The determinant is positive or negative according to whether the linear mapping preserves or reverses the orientation of $n$-space.  

+ Determinant of $2 \times 2$ matrix
  + $\det(A) = \begin{vmatrix}a & b \\ c & d \end{vmatrix} = a \times d - b \times c$
+ Determinant of $3 \times 3$ matrix
  + $\det(A) = \begin{vmatrix}a & b & c \\d & e & f \\g & h & i \end{vmatrix}$ $= a \times \begin{vmatrix}e & f \\ h & i \end{vmatrix} - b \times \begin{vmatrix}d & f \\ g & i \end{vmatrix} + c \times \begin{vmatrix}d & e \\ g & h \end{vmatrix}$

&nbsp;&nbsp;&nbsp;&nbsp; In linear algebra, a matrix is **invertible if and only if its determinant is non-zero**, and correspondingly the matrix is singular if and only if its determinant is zero. This leads to the use of determinants in defining the characteristic polynomial of a matrix, whose roots are the eigenvalues.  

&nbsp;&nbsp;&nbsp;&nbsp; For a square matrix, i.e., a matrix with the same number of rows and columns, one can capture important information about the matrix in a just single number, called the determinant. The determinant is useful for solving linear equations, capturing how linear transformation change area or volume, and changing variables in integrals. The simplest square matrix is a $1×1$ matrix, which isn't very interesting since it contains just a single number. The determinant of a $1×1$ matrix is that number itself.  

### Characteristic polynomial, 特徵多項式

[Characteristic polynomial, wikipedia](https://en.wikipedia.org/wiki/Characteristic_polynomial)  

### Linear Transformation

[Matrices as transformations](https://www.khanacademy.org/math/precalculus/precalc-matrices/modal/a/matrices-as-transformations)  
The word “transformation” means the same thing as “function”: something which takes in a number and outputs a number, like $f(x) = 2x$. However, while we typically visualize functions with their graphs, people **tend to use the word “transformation” to indicate that you should instead visualize some object moving, stretching, squishing, etc**.


## References

Eigenvalues and Eigenvectors

+ [Section 5-3 : Review : Eigenvalues & Eigenvectors](http://tutorial.math.lamar.edu/Classes/DE/LA_Eigen.aspx)
+ [Linear Algebra Example Problems - Basis for an Eigenspace](https://www.youtube.com/watch?v=hIoAcfPfPoM)

Identity Matrix

+ [Intro to identity matrices](https://www.khanacademy.org/math/precalculus/precalc-matrices/properties-of-matrix-multiplication/a/intro-to-identity-matrices)

Determinant

+ [What does the determinant of a matrix mean physically? How do I visualize it?](https://www.quora.com/What-does-the-determinant-of-a-matrix-mean-physically-How-do-I-visualize-it)
+ [Determinant of a Matrix](https://www.mathsisfun.com/algebra/matrix-determinant.html)
+ [The determinant of a matrix](https://mathinsight.org/determinant_matrix)

[Principal Component Analysis in Python](https://plot.ly/ipython-notebooks/principal-component-analysis/)
