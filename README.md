# Iterative Shrinkage Thresholding Algorithm (ISTA) for LASSO problem
This is Iterative Shrinkage Thresholding Algorithm (ISTA) for solving LASSO problem. LASSO problem assumes that signal x be sparse, and this assumption is not wrong. Most natural siggnal can be represented sparse in some domain. For example, natural scenes are sparse in Fourier transform domain or DCT domain. Sometimes the scene itself can be very sparse (e.g. stars at night). <br>

ISTA is a first-order method which is gradient-based so it is simple and efficient. However, its convergence is slow - O(1/k). A fast ISTA (FISTA) is developed for faster convergence, which gives an improved complexity, O(1/(k^2)). <br>

Here we will compare the LASSO problem with ISTA and FISTA to CLS problem with CG. You will see ISTA and FISTA work well on the sparse signal while CLS doesn't. You will see the improved performance of FISTA over ISTA as well.

# Results of image deconvolution
![alt tag](https://github.com/seunghwanyoo/ista_lasso/blob/master/results/original.jpg) 
![alt tag](https://github.com/seunghwanyoo/ista_lasso/blob/master/results/degraded.jpg) <br>
![alt tag](https://github.com/seunghwanyoo/ista_lasso/blob/master/results/lasso-ista.jpg) 
![alt tag](https://github.com/seunghwanyoo/ista_lasso/blob/master/results/lasso-fista.jpg) <br>
![alt tag](https://github.com/seunghwanyoo/ista_lasso/blob/master/results/cls-cg.jpg) <br>


# Description of files/folders
- demo_ista.m: test script
- /opt: includes functions for optimization methods
- /func: includes functions for objective function and corresponding gradient function, and hessian function.
- /result: includes result images

# Image degradation and deconvolution
- Degradation model: y = Hx = HPb = Ab 
          where A = HP, P:representation matrix (P = I in my example)
- Deconvolution: <br>
   (1) LASSO: min_x ||y-Ab||2 + lambda*||b||1 <br>
   (2) CLS: min_x 0.5||y-Hx||2 + 0.5*lambda*||Cx||2 <br>

# Optimization methods
Each approach is solved by a different numerical optimization method. ISTA/FISTA are used for LASSO problem and CG method is used for CLS. <br>
  (1) ISTA: A type of proximal gradient method <br>
  (2) FISTA: Fast version of ISTA <br>
  (3) Conjugate Gradient A efficient optimization algorithm for solving Ax=b (or quadratic objective function) <br>

# Contact
Seunghwan Yoo (seunghwanyoo2013@u.northwestern.edu)
