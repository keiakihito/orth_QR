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




/*
Personal Note: How to Run magma in NCSA
1. Srun
2. Set path
export LD_LIBRARY_PATH=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/magma-2.8.0-mvkrj4y/lib:/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc/lib64:$LD_LIBRARY_PATH

3. Run orth_QRTest.cu
nvcc orth_QRTest.cu -o orth_QRTest -I/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc/include -L/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc/lib64 -lcudart -lcublas -lcusolver -I/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/magma-2.8.0-mvkrj4y/include -L/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/magma-2.8.0-mvkrj4y/lib -L/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/openblas-0.3.25-5yvxjnl/lib -lmagma -lopenblas

4. ./orth_QRTest
 */
void orth_QR(double** mtxQ_trnc_d, double* mtxZ_d, int numOfRow, int numOfClm, int &currentRank)
{
    magma_init();

    bool debug = true;
    const double THREASHOLD = 1e-6;

    magmaDouble_ptr mtxZ_cpy_d = NULL;
    magmaDouble_ptr mtxQ_d = NULL;
    magmaDouble_ptr tau_d = NULL;

    magma_int_t lda = numOfRow; // leading dimenstion
    magma_int_t mn = numOfClm; // numOfClm < numOfRow
    magma_int_t lwork = 0;
    magma_int_t nb = magma_get_dgeqp3_nb(numOfRow, numOfClm); // Block size
    magma_int_t info = 0;
    magma_int_t* ipiv = NULL; //pivotig information

    magmaDouble_ptr tau_h = NULL;
    magmaDouble_ptr dT = NULL;
    magmaDouble_ptr work_d = NULL;



    //(1) Allocate memory
    // CHECK(cudaMalloc((void**)&mtxZ_cpy_d, numOfRow * numOfClm * sizeof(magmaDouble_ptr)));
    CHECK_MAGMA(magma_dmalloc(&mtxZ_cpy_d, lda * mn));

    // CHECK(cudaMalloc((void**)&mtxQ_d, numOfRow * numOfRow * sizeof(magmaDouble_ptr)));
    CHECK_MAGMA(magma_dmalloc(&mtxQ_d, lda * lda));

    // ipiv = (magma_int_t*)calloc(numOfClm, sizeof(magma_int_t));
    CHECK_MAGMA(magma_imalloc_cpu(&ipiv, mn));

    // CHECK(cudaMalloc((void**)&tau_d, numOfClm * sizeof(double)));
    CHECK_MAGMA(magma_dmalloc(&tau_d, mn));

    // tau_h = (double*)calloc(numOfClm, sizeof(double));
    CHECK_MAGMA(magma_dmalloc_cpu(&tau_h, mn));

    // CHECK(cudaMalloc((void**)&dT, (2 * nb * numOfRow + (2*nb+1) * nb) * sizeof(magmaDouble_ptr)));
    CHECK_MAGMA(magma_dmalloc(&dT, (2 * nb * numOfRow + (2*nb+1) * nb)));

    //(2) Copy value to device
    CHECK(cudaMemcpy(mtxZ_cpy_d, mtxZ_d, numOfRow * numOfClm * sizeof(magmaDouble_ptr), cudaMemcpyDeviceToDevice));

    if(debug){
        printf("\n\n~~mtxZ_cpy_d~~\n\n");
        print_mtx_clm_d(mtxZ_cpy_d, numOfRow, numOfClm);
    }


    //(3) Calculate lwork based on the documentation
    // lwork = (numOfClm + 1) * nb + 2 * numOfClm;
    lwork = (mn + 1) * nb + 2 * mn;

    // lwork = max( lwork, numOfRow*numOfClm + numOfClm );
    if(debug){
        printf("\n\nlwork: %d\n\n", lwork);
    }

    //(4) Allocate workspace 
    // CHECK(cudaMalloc((void**)&work_d, lwork * sizeof(magmaDouble_ptr)));
    CHECK_MAGMA(magma_dmalloc(&work_d, lwork));


    printf("\nBefore magma_dgeqp3_gpu\n\n");

    if(debug){
        printf("\n\nipiv: \n");
        for(int i = 0; i < numOfClm; i++){
            printf("%d\n", ipiv[i]);
        }
    }

    if(debug){
        printf("\n\ntau_h: \n");
        for(int i = 0; i < numOfClm; i++){
            printf("%f\n", tau_h[i]);
        }
    }
    
    if(debug){
        printf("\n\ntau_d: \n");
        print_vector(tau_d, numOfClm);
    }
    
    //FIXME the expected and ectual tau_d do not match.
    //(5) Perform QR decompostion with column pivoting
    //mtxZ_cpy_d contains the R matrix in its upper triangular part,
    //the orthogonal matrix Q is represented implicitly by the Householder vectors stored in the lower triangular part of mtxZ_cpy_d and in tau_d, scalor.
    CHECK_MAGMA(magma_dgeqp3_gpu((magma_int_t)numOfRow, (magma_int_t)numOfClm, mtxZ_cpy_d, lda, ipiv, tau_d, work_d, lwork, &info));

    /*
    Expected

    Permutation indices P:
    [2 0 1]

    Tau values:
    [-0.42616235 -0.27529031 -0.0221448   0.47085005]

    Matrix R (upper triangular):
    [[-2.11187121e+01 -5.20865096e+00 -1.31636815e+01]
    [ 0.00000000e+00  1.69409420e+00  8.47047100e-01]
    [ 0.00000000e+00  0.00000000e+00  7.10889596e-16]
    [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]]

    Householder vectors (lower triangular part of Q):
    [[ 0.          0.          0.          0.        ]
    [-0.47351372  0.          0.          0.        ]
    [-0.5208651   0.16940942  0.          0.        ]
    [-0.56821647  0.61410915  0.27982178  0.        ]]
    
    */

    if(debug){
        printf("\nAfter magma_dgeqp3_gpu\n\n");
        printf("\n\nipiv: \n");
        for(int i = 0; i < numOfClm; i++){
            printf("%d\n", ipiv[i]);
        }
        printf("\n\ntau_d: \n");
        print_vector(tau_d, numOfClm);

        printf("\n\n~~mtxZ_cpy_d~~\n-the R matrix in its upper triangular part\n-the orthogonal matrix Q is represented implicitly by the Householder vectors stored in the lower triangular\n\n");
        print_mtx_clm_d(mtxZ_cpy_d, numOfRow, numOfClm);
    }

    if(debug){
        
    }

    CHECK(cudaMemcpy(tau_h, tau_d, numOfClm * sizeof(double), cudaMemcpyDeviceToHost));

    if(debug){
        printf("\n\ntau_h: \n");
        for(int i = 0; i < numOfClm; i++){
            printf("%f\n", tau_h[i]);
        }
    }
    
    
    //(6) Iterate diagonal elements matrix R that is upper trianguar part of mtxZ_cpy_d to find rank
    currentRank = setRank(mtxZ_cpy_d, currentRank, THREASHOLD);


    //TODO Moving on after fixing magma_dgeqp3_gpu issue
    // //(7) Create an identity matrix on the device to extract matrix Q
    // CHECK_MAGMA(magma_dorgqr_gpu((magma_int_t)numOfRow, (magma_int_t)numOfRow, (magma_int_t)numOfClm, mtxZ_cpy_d, lda, tau_h, dT, nb, &info));
    
    // //(8) Copy matrix Q to the mtxQ_trnc_d with significant column vectors
    // CHECK(cudaMemcpy(mtxQ_d, mtxZ_cpy_d, numOfRow * currentRank * sizeof(magmaDouble_ptr), cudaMemcpyDeviceToDevice));


    // if(debug){
    //     printf("\n\nAfter magma_dorgqr_gpu\n");  
    //     printf("\n~~mtxZ_cpy_d will be mtxQ~~\n\n");
    //     print_mtx_clm_d(mtxZ_cpy_d, numOfRow, numOfClm);
    //     printf("\n\n~~mtxQ_d_truncated~~\n\n");
    //     print_mtx_clm_d(mtxZ_cpy_d, numOfRow, currentRank);
    // }



    // // //(9)Pass the address to the provided pointer, updating orhtonomal set
    // CHECK(cudaFree(*mtxQ_trnc_d));
    // *mtxQ_trnc_d = NULL;
    // *mtxQ_trnc_d = mtxQ_d;

    //(10) Clean up
    // CHECK(cudaFree(mtxZ_cpy_d));
    CHECK_MAGMA(magma_free(mtxZ_cpy_d));

    // CHECK(cudaFree(mtxQ_d));
    CHECK_MAGMA(magma_free(mtxQ_d));

    // free(ipiv);
    CHECK_MAGMA(magma_free_cpu(ipiv));

    // free(tau_h);
    CHECK_MAGMA(magma_free_cpu(tau_h));

    // CHECK(cudaFree(tau_d));
    CHECK_MAGMA(magma_free(tau_d));

    // CHECK(cudaFree(work_d));
    CHECK_MAGMA(magma_free(work_d));

    // CHECK(cudaFree(dT));
    CHECK_MAGMA(magma_free(dT));

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