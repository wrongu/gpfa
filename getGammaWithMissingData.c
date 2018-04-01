/*
 * getGammaWithMissingData.c - Get 'Gamma' matrix for use in gaussian process latent inference when
 * data contains missing (NaN) values
 *
 * This is a MEX file for MATLAB.
*/

#include "mex.h"
#include "matrix.h"

double getMxVal(const mxArray *matrix, int rowsize, int rowidx, int colidx){
	int subs[2];
	double *values = mxGetPr(matrix);
	return values[rowidx + colidx * rowsize];
}

void setMxVal(mxArray *matrix, int rowsize, int rowidx, int colidx, double value){
	int subs[2];
	double *values = mxGetPr(matrix);
	values[rowidx + colidx * rowsize] = value;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	// Check #input and #output arguments
 	if(nrhs != 3){
 		mexErrMsgIdAndTxt("GPFA:getGammaWithMissingData:nrhs",
 			"Wrong number of arguments. Expecting (Y, diag(D), C)");
 	}

 	if(nlhs > 1){
 		mexErrMsgIdAndTxt("GPFA:getGammaWithMissingData:nlhs", "Too many output arguments.");
 	}

 	const mxArray *Y = prhs[0];
 	const mxArray *D = prhs[1];
 	const mxArray *C = prhs[2];

 	const mwSize *dimsY = mxGetDimensions(Y);
 	const mwSize *dimsD = mxGetDimensions(D);
 	const mwSize *dimsC = mxGetDimensions(C);

 	if(dimsY[1] != dimsC[0] || dimsD[1] != 1 || dimsY[1] != dimsD[0]){
 		mexErrMsgIdAndTxt("GPFA:getGammaWithMissingData:nlhs", "Size mismatch! Expected Y [T x N], D [N x 1], and C [N x L]");
 	}

 	const int T = dimsY[0]; // time points
 	const int N = dimsY[1]; // neurons
 	const int L = dimsC[1]; // latents

 	// Allocate output matrix Gamma with size [T*L x T*L], initialized to zeros by default
 	plhs[0] = mxCreateDoubleMatrix(T*L, T*L, mxREAL);
 	mxArray *Gamma = plhs[0];

 	// Precompute outer product of (C' * inv(D) * C) terms at each N
 	mxArray *outerC = mxCreateDoubleMatrix(N, L * L, mxREAL);
 	for(int n=0; n < N; n++){
 		for(int a=0; a < L; a++){
 			for(int b=a; b<L; b++){
 				double value = getMxVal(C, N, n, a) * getMxVal(C, N, n, b) / getMxVal(D, N, n, 0);
 				// set upper + lower triangular part (symmetric)
 				setMxVal(outerC, N, n, a * L + b, value);
 				setMxVal(outerC, N, n, b * L + a, value);
 			}
 		}
 	}

 	// Compute values of output matrix.
 	// t indexes into time
 	for(int t = 0; t < T; t++){
 		// n indexes into neurons
 		for(int n = 0; n < N; n++){
 			double yval = getMxVal(Y, T, t, n);
 			
 			// Skip values wherever Y is NaN
 			if(mxIsNaN(yval)) continue;

 			// a,b index latents
 			for(int a = 0; a < L; a++){
 				for(int b = 0; b < L; b++){
 					double value = getMxVal(outerC, N, n, a * L + b);
 					setMxVal(Gamma, T*L, t+a*T, t+b*T, value + getMxVal(Gamma, T*L, t+a*T, t+b*T));
 				}
 			}
 		}
 	}

 	// Free memory created for C' * inv(D) * C product
 	mxDestroyArray(outerC);
 }