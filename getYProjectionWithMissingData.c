/*
 * getYProjectionWithMissingData.c - Get "Y * inv(D) * C" term for use in gaussian process latent
 * inference when data contains missing (NaN) values
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
 		mexErrMsgIdAndTxt("GPFA:getYProjectionWithMissingData:nrhs",
 			"Wrong number of arguments. Expecting (Y, diag(D), C)");
 	}

 	if(nlhs > 1){
 		mexErrMsgIdAndTxt("GPFA:getYProjectionWithMissingData:nlhs", "Too many output arguments.");
 	}

 	const mxArray *Y = prhs[0];
 	const mxArray *D = prhs[1];
 	const mxArray *C = prhs[2];

 	const mwSize *dimsY = mxGetDimensions(Y);
 	const mwSize *dimsD = mxGetDimensions(D);
 	const mwSize *dimsC = mxGetDimensions(C);

 	if(dimsY[1] != dimsC[0] || dimsD[1] != 1 || dimsY[1] != dimsD[0]){
 		mexErrMsgIdAndTxt("GPFA:getYProjectionWithMissingData:nlhs", "Size mismatch! Expected Y [T x N], D [N x 1], and C [N x L]");
 	}

 	const int T = dimsY[0]; // time points
 	const int N = dimsY[1]; // neurons
 	const int L = dimsC[1]; // latents

 	// Allocate output array for projection with size [T*L x 1], initialized to zeros by default
 	plhs[0] = mxCreateDoubleMatrix(T*L, 1, mxREAL);
 	mxArray *Proj = plhs[0];

 	// Compute values of output matrix.
 	// t indexes into time
 	for(int t = 0; t < T; t++){
 		// n indexes into neurons
 		for(int n = 0; n < N; n++){
 			double yval = getMxVal(Y, T, t, n);
 			
 			// Skip values wherever Y is NaN
 			if(mxIsNaN(yval)) continue;

 			// l indexes latents
 			for(int l = 0; l < L; l++){
				double value = yval * getMxVal(C, N, n, l) / getMxVal(D, N, n, 0);
				setMxVal(Proj, T*L, t+T*l, 0, value + getMxVal(Proj, T*L, t+T*l, 0));
 			}
 		}
 	}
 }