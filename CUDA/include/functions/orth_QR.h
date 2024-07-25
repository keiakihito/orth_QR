#ifndef ORTH_QR_H
#define ORTH_QR_H


#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <magma_v2.h>
#include <magma_lapack.h>

// helper function CUDA error checking and initialization
#include "helper.h"
#include "cuBLAS_util.h"
#include "cuSOLVER_util.h"

//Input: double* mtxQ_trnc_d, double* mtxZ_d, int number of Row, int number Of column, int & currentRank
//Process: the function extracts orthonormal set from the matrix Z
//Output: double* mtxQ_trnc_d, the orthonormal set of matrix Z with significant column vectors
void orth_QR(double** mtxQ_trnc_d, double* mtxZ_d, int numOfRow, int numOfClm, int &currentRank);

//Input: singluar values, int currnet rank, double threashould
//Process: Check eigenvalues are greater than threashould, then set new rank
//Output: int newRank
int setRank(double* sngVals_d, int currentRank, double threashold);





void orth_QR(double** mtxQ_trnc_d, double* mtxZ_d, int numOfRow, int numOfClm, int &currentRank)
{
    magma_init();

    bool debug = true;
    const double THREASHOLD = 1e-6;

    magmaDouble_ptr mtxZ_cpy_d = NULL;
    magmaDouble_ptr mtxQ_d = NULL;
    magma_int_t lda = numOfRow; // leading dimenstion
    magma_int_t lwork = 0;
    magma_int_t nb = magma_get_dgeqp3_nb(numOfRow, numOfClm); // Block size
    magma_int_t info = 0;
    magma_int_t* ipiv = NULL; //pivotig information
    double *tau_d = NULL;
    magmaDouble_ptr dT = NULL;
    magmaDouble_ptr work_d = NULL;

    

    //(1) Allocate memory
    CHECK(cudaMalloc((void**)&mtxZ_cpy_d, numOfRow * numOfClm * sizeof(magmaDouble_ptr)));
    CHECK(cudaMalloc((void**)&mtxQ_d, numOfRow * numOfClm * sizeof(magmaDouble_ptr)));
    // CHECK(cudaMalloc((void**)&ipiv, numOfClm * sizeof(magma_int_t)));
    ipiv = (magma_int_t*)calloc(numOfClm, sizeof(magma_int_t));
    CHECK(cudaMalloc((void**)&tau_d, numOfClm * sizeof(double)));
    CHECK(cudaMalloc((void**)&dT, (2 * nb * numOfRow + (2*nb+1) * nb) * sizeof(magmaDouble_ptr)));

    //(2) Copy value to device
    CHECK(cudaMemcpy(mtxZ_cpy_d, mtxZ_d, numOfRow * numOfClm * sizeof(magmaDouble_ptr), cudaMemcpyDeviceToDevice));

    if(debug){
        printf("\n\n~~mtxZ_cpy_d~~\n\n");
        print_mtx_clm_d(mtxZ_cpy_d, numOfRow, numOfClm);
    }
    

    //(3) Calculate lwork based on the documentation
    lwork = (numOfClm + 1) * nb + 2 * numOfClm;
    lwork = max( lwork, numOfRow*numOfClm + numOfClm );
    if(debug){
        printf("\n\nlwork: %d\n\n", lwork);
    }

    //(4) Allocate workspace 
    CHECK(cudaMalloc((void**)&work_d, lwork * sizeof(magmaDouble_ptr)));

    if (debug) {
        if (cudaMalloc((void**)&work_d, lwork * sizeof(double)) != cudaSuccess) {
            fprintf(stderr, "Failed to allocate workspace\n");
            exit(EXIT_FAILURE);
        }
        printf("\n\nwork_d allocated successfully.\n\n");
    }
   
    if(debug){
        printf("\n\nipiv: \n");
        for(int i = 0; i < numOfClm; i++){
            printf("%d\n", ipiv[i]);
        }
    }
       
    if(debug){
        printf("\n\ntau_d: \n");
        print_vector(tau_d, numOfClm);
    }



    printf("\nBefore magma_dgeqp3_gpu\n\n");
    //(5) Perform QR decompostion with column pivoting
    //mtxZ_cpy_d contains the R matrix in its upper triangular part,
    //the orthogonal matrix Q is represented implicitly by the Householder vectors stored in the lower triangular part of mtxZ_cpy_d and in tau_d, scalor.
    CHECK_MAGMA(magma_dgeqp3_gpu((magma_int_t)numOfRow, (magma_int_t)numOfClm, mtxZ_cpy_d, lda, ipiv, tau_d, work_d, lwork, &info));

    if (info != 0) {
        fprintf(stderr, "magma_dgeqp3_gpu returned error %d\n", info);
        exit(EXIT_FAILURE);
    }


    if(debug){
        printf("\nAfter magma_dgeqp3_gpu\n\n");    
        printf("\n\n~~mtxZ_cpy_d~~\n-the R matrix in its upper triangular part\n-the orthogonal matrix Q is represented implicitly by the Householder vectors stored in the lower triangular\n\n");
        print_mtx_clm_d(mtxZ_cpy_d, numOfRow, numOfClm);
    }
    

    //(6) Iterate diagonal elements matrix R that is upper trianguar part of mtxZ_cpy_d to find rank
    currentRank = setRank(mtxZ_cpy_d, currentRank, THREASHOLD);


    //(7) Create an identity matrix on the device to extract matrix Q
    // CHECK_MAGMA(magma_dorgqr_gpu((magma_int_t)numOfRow, (magma_int_t)numOfRow, (magma_int_t)numOfClm, mtxZ_cpy_d, lda, tau_d, dT, nb, &info));


    //(8) Copy matrix Q to the mtxQ_trnc_d with significant column vectors
    CHECK(cudaMemcpy(mtxQ_d, mtxZ_cpy_d, numOfRow * currentRank * sizeof(double), cudaMemcpyDeviceToDevice));

    //(9)Pass the address to the provided pointer, updating orhtonomal set
    CHECK(cudaFree(*mtxQ_trnc_d));
    *mtxQ_trnc_d = NULL;
    *mtxQ_trnc_d = mtxQ_d;

    //(10) Clean up
    CHECK(cudaFree(mtxZ_cpy_d));
    CHECK(cudaFree(mtxQ_d));
    free(ipiv);
    CHECK(cudaFree(tau_d));
    CHECK(cudaFree(work_d));
    CHECK(cudaFree(dT));

    magma_finalize();

} // end of orth_QR


//Input: singluar values, int currnet rank, double threashould
//Process: Check eigenvalues are greater than threashould, then set new rank
//Output: int newRank
int setRank(double* sngVals_d, int currentRank, double threashold)
{   
    int newRank = 0;

    //Allcoate in heap to copy value from device
    double* sngVals_h = (double*)malloc(currentRank * sizeof(double));
    // Copy singular values from Device to count eigen values
    CHECK(cudaMemcpy(sngVals_h, sngVals_d, currentRank * sizeof(double), cudaMemcpyDeviceToHost));

    for(int wkr = 0; wkr < currentRank; wkr++){
        if(sngVals_h[wkr] > threashold){
            newRank++;
        } // end of if
    } // end of for

    free(sngVals_h);

    return newRank;
}




#endif // ORTH_QR_H