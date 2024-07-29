// includes, system
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

/*Using updated (v2) interfaces to cublas*/
#include<cublas_v2.h>
#include<cuda_runtime.h>
#include<cusparse.h>
#include <cusolverDn.h>
#include<sys/time.h>


//Utilities
// helper function CUDA error checking and initialization 
#include "../include/functions/orth_QR.h"



void orth_QRtest1();
void orth_QRtest2();
void orth_QRtest3();
void orth_QRtest4();
void orth_QRtest5();



int main(int argc, char** argv)
{
    printf("\n\n~~orth_QR_Test()~~\n\n");

    printf("\n\n🔍🔍🔍Test case 1🔍🔍🔍\n");
    orth_QRtest1();
    
    printf("\n\n🔍🔍🔍Test case 2🔍🔍🔍\n");
    orth_QRtest2();

    printf("\n\n🔍🔍🔍Test case 3🔍🔍🔍\n");
    orth_QRtest3();

    printf("\n\n🔍🔍🔍Test case 4🔍🔍🔍\n");
    orth_QRtest4();

    printf("\n\n🔍🔍🔍Test case 5🔍🔍🔍\n");
    orth_QRtest5();

    printf("\n= = = End of orth_test  = = = \n\n");


}// end of main


void orth_QRtest1()
{
       /*
    Z = | 1.0  5.0  9.0 |
        | 2.0  6.0  10.0|
        | 3.0  7.0  11.0|
        | 4.0  8.0  12.0| 
    */

    // Define the dense matrixB column major
    double mtxZ[] = {
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
    };

    int numOfRow = 4;
    int numOfClm = 3;
    int crntRank = 3;

    double* mtxZ_d = NULL;
    double* mtxQ_trnc_d = NULL;
    double* mtxI_d = NULL;

    bool debug = true;

    cublasHandle_t cublasHandler = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandler));


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(double)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    orth_QR(&mtxQ_trnc_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    printf("\n\n~~Outside function: mtxQ_trunc_d~~\n\n");
    print_mtx_clm_d(mtxQ_trnc_d, numOfRow, crntRank);
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);

    printf("\n\n= = 🔍Check Orthogonality🔍 = = \n\n");
    CHECK(cudaMalloc((void**)&mtxI_d, crntRank * crntRank * sizeof(double)));
    multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxQ_trnc_d, mtxI_d, numOfRow, crntRank);
    print_mtx_clm_d(mtxI_d, crntRank, crntRank);



} // end of orth_QRtest1()

void orth_QRtest2()
{
 // Define the dense matrixB column major
    double mtxZ[] = {
    1.1, 0.8, 3.0, 2.2, 0.2, 0.7,
    2.2, 1.6, 4.1, 3.3, 0.3, 0.8,
    3.3, 2.4, 5.2, 4.4, 0.4, 1.1,
    4.4, 3.2, 6.3, 5.5, 0.5, 1.5,
    5.5, 2.3, 0.7, 1.7, 0.6, 3.2
    };

    int numOfRow = 6;
    int numOfClm = 5;
    int crntRank = 5;

    double* mtxZ_d = NULL;
    double* mtxQ_trnc_d = NULL;
    double* mtxI_d = NULL;

    bool debug = true;

    cublasHandle_t cublasHandler = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandler));


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(double)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    orth_QR(&mtxQ_trnc_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    printf("\n\n~~Outside function: mtxQ_trunc_d~~\n\n");
    print_mtx_clm_d(mtxQ_trnc_d, numOfRow, crntRank);
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);

    printf("\n\n= = 🔍Check Orthogonality🔍 = = \n\n");
    CHECK(cudaMalloc((void**)&mtxI_d, crntRank * crntRank * sizeof(double)));
    multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxQ_trnc_d, mtxI_d, numOfRow, crntRank);
    print_mtx_clm_d(mtxI_d, crntRank, crntRank);

} // end of orth_QRtest1()

void orth_QRtest3()
{
// Define the dense matrixB column major
    double mtxZ[] = {
        1.1, 0.8, 3.0, 2.2, 1.4, 0.6, 7.7,
        2.2, 1.6, 4.1, 3.3, 2.5, 1.5, 5.6,
        3.3, 2.4, 5.2, 4.4, 3.6, 2.4, 9.6,
        4.4, 3.2, 6.3, 5.5, 4.7, 3.3, 8.8,
        5.5, 4.0, 7.4, 6.6, 5.8, 4.2, 7.0
    };

    int numOfRow = 7;
    int numOfClm = 5;
    int crntRank = 5;

        double* mtxZ_d = NULL;
    double* mtxQ_trnc_d = NULL;
    double* mtxI_d = NULL;

    bool debug = true;

    cublasHandle_t cublasHandler = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandler));


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(double)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    orth_QR(&mtxQ_trnc_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    printf("\n\n~~Outside function: mtxQ_trunc_d~~\n\n");
    print_mtx_clm_d(mtxQ_trnc_d, numOfRow, crntRank);
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);

    printf("\n\n= = 🔍Check Orthogonality🔍 = = \n\n");
    CHECK(cudaMalloc((void**)&mtxI_d, crntRank * crntRank * sizeof(double)));
    multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxQ_trnc_d, mtxI_d, numOfRow, crntRank);
    print_mtx_clm_d(mtxI_d, crntRank, crntRank);


} // end of orth_QRtest1()

void orth_QRtest4()
{
 // Define the dense matrixB column major
    double mtxZ[] = {
        1.2, 0.9, 3.1, 2.3, 1.5, 0.7, 2.1, 3.3, 0.1, 0.2, 0.3, 
        2.3, 1.8, 4.2, 3.4, 2.6, 1.6, 3.0, 4.4, 0.5, 0.7, 0.2, 
        3.4, 2.7, 5.3, 4.5, 3.7, 2.5, 4.9, 5.5, 0.3, 0.4, 0.5, 
        4.5, 3.6, 6.4, 5.6, 4.8, 3.4, 5.8, 6.6, 1.1, 1.2, 1.3, 
        5.6, 4.5, 7.5, 6.7, 5.9, 4.3, 6.7, 7.7, 1.9, 1.5, 1.8, 
        6.7, 5.4, 8.6, 7.8, 6.0, 5.2, 7.6, 8.8, 2.2, 2.3, 2.5, 
        7.8, 6.3, 9.7, 8.9, 7.1, 6.1, 8.5, 9.9, 2.9, 3.1, 3.2
    };


    int numOfRow = 11;
    int numOfClm = 7;
    int crntRank = 7;

    double* mtxZ_d = NULL;
    double* mtxQ_trnc_d = NULL;
    double* mtxI_d = NULL;

    bool debug = true;

    cublasHandle_t cublasHandler = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandler));


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(double)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    orth_QR(&mtxQ_trnc_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    printf("\n\n~~Outside function: mtxQ_trunc_d~~\n\n");
    print_mtx_clm_d(mtxQ_trnc_d, numOfRow, crntRank);
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);

    printf("\n\n= = 🔍Check Orthogonality🔍 = = \n\n");
    CHECK(cudaMalloc((void**)&mtxI_d, crntRank * crntRank * sizeof(double)));
    multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxQ_trnc_d, mtxI_d, numOfRow, crntRank);
    print_mtx_clm_d(mtxI_d, crntRank, crntRank);



} // end of orth_QRtest1()

void orth_QRtest5()
{
    // Define the dense matrixB column major
    // Define the dense matrixB column major
    double mtxZ[] = {
        1.2, 0.9, 3.1, 2.3, 1.5, 0.7, 2.1, 3.3, 2.5, 1.9, 0.9, 1.0, 1.1, 1.2, 1.3, 7.7, 6.4, 8.6, 8.8, 6.0,
        2.3, 1.8, 4.2, 3.4, 2.6, 1.6, 3.0, 4.4, 3.6, 2.8, 1.8, 2.1, 2.4, 2.6, 2.8, 6.2, 7.6, 8.8, 7.0, 6.4,
        3.4, 2.7, 5.3, 4.5, 3.7, 2.5, 4.9, 5.5, 4.7, 3.7, 2.7, 3.2, 3.7, 4.0, 4.3, 5.4, 6.5, 7.6, 8.2, 8.8,
        4.5, 3.6, 6.4, 5.6, 4.8, 3.4, 5.8, 6.6, 5.8, 4.6, 3.6, 4.3, 5.0, 5.4, 5.8, 8.8, 7.3, 9.7, 9.9, 7.1,
        5.6, 4.5, 7.5, 6.7, 5.9, 4.3, 6.7, 7.7, 6.9, 5.5, 4.5, 5.4, 6.3, 6.8, 7.3, 7.1, 8.5, 9.9, 8.1, 7.3,
        6.7, 5.4, 8.6, 7.8, 6.0, 5.2, 7.6, 8.8, 7.0, 6.4, 5.4, 6.5, 7.6, 8.2, 8.8, 6.3, 7.6, 8.9, 9.6, 10.3,
        7.8, 6.3, 9.7, 8.9, 7.1, 6.1, 8.5, 9.9, 8.1, 7.3, 6.3, 7.6, 8.9, 9.6, 10.3, 9.9, 8.2, 0.8, 9.0, 8.2,
        8.9, 7.2, 0.8, 9.0, 8.2, 7.0, 9.4, 0.1, 9.2, 8.2, 7.2, 8.7, 0.2, 0.5, 1.8, 8.0, 9.4, 0.3, 9.2, 8.2,
        9.0, 8.1, 1.9, 1.1, 9.3, 8.9, 0.3, 1.1, 0.3, 9.1, 8.1, 9.8, 1.5, 1.4, 2.4, 7.2, 8.7, 0.2, 0.4, 1.8,
        1.1, 9.0, 2.0, 2.2, 0.4, 9.8, 1.2, 2.2, 1.4, 0.2, 9.0, 0.9, 2.8, 2.8, 3.0, 1.0, 9.1, 1.9, 2.2, 0.4,
        2.2, 1.9, 3.1, 3.3, 1.5, 1.7, 2.1, 3.3, 2.5, 1.9, 0.9, 1.0, 1.1, 1.2, 1.3, 9.9, 1.2, 2.2, 1.4, 0.7,
        3.3, 2.8, 4.2, 4.4, 2.6, 2.6, 3.0, 4.4, 3.6, 2.8, 1.8, 2.1, 2.4, 2.6, 2.8, 9.0, 0.9, 2.8, 2.8, 3.0,
        4.4, 3.7, 5.3, 5.5, 3.7, 3.5, 4.9, 5.5, 4.7, 3.7, 2.7, 3.2, 3.7, 4.0, 4.3, 2.1, 0.7, 2.0, 3.2, 1.5,
        5.5, 4.6, 6.4, 6.6, 4.8, 4.4, 5.8, 6.6, 5.8, 4.6, 3.6, 4.3, 5.0, 5.4, 5.8, 0.8, 2.1, 3.2, 2.3, 1.0,
        6.6, 5.5, 7.5, 7.7, 5.9, 5.3, 6.7, 7.7, 6.9, 5.5, 4.5, 5.4, 6.3, 6.8, 7.3, 0.9, 1.0, 1.1, 1.2, 1.3
    };
    int numOfRow = 20;
    int numOfClm = 15;
    int crntRank = 15;

        double* mtxZ_d = NULL;
    double* mtxQ_trnc_d = NULL;
    double* mtxI_d = NULL;

    bool debug = true;

    cublasHandle_t cublasHandler = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandler));


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(double)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    orth_QR(&mtxQ_trnc_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    printf("\n\n~~Outside function: mtxQ_trunc_d~~\n\n");
    print_mtx_clm_d(mtxQ_trnc_d, numOfRow, crntRank);
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);

    printf("\n\n= = 🔍Check Orthogonality🔍 = = \n\n");
    CHECK(cudaMalloc((void**)&mtxI_d, crntRank * crntRank * sizeof(double)));
    multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxQ_trnc_d, mtxI_d, numOfRow, crntRank);
    print_mtx_clm_d(mtxI_d, crntRank, crntRank);


} // end of orth_QRtest1()

/*
Sample Run

~~orth_QR_Test()~~



🔍🔍🔍Test case 1🔍🔍🔍


~~mtxZ~~

1.000000 5.000000 9.000000 
2.000000 6.000000 10.000000 
3.000000 7.000000 11.000000 
4.000000 8.000000 12.000000 


~~Outside function: mtxQ_trunc_d~~

-0.426162 -0.719990 
-0.473514 -0.275290 
-0.520865 0.169409 
-0.568216 0.614109 


~~Current Rarnk = 2~~



= = 🔍Check Orthogonality🔍 = = 

1.000000 0.000000 
0.000000 1.000000 


🔍🔍🔍Test case 2🔍🔍🔍


~~mtxZ~~

1.100000 2.200000 3.300000 4.400000 5.500000 
0.800000 1.600000 2.400000 3.200000 2.300000 
3.000000 4.100000 5.200000 6.300000 0.700000 
2.200000 3.300000 4.400000 5.500000 1.700000 
0.200000 0.300000 0.400000 0.500000 0.600000 
0.700000 0.800000 1.100000 1.500000 3.200000 


~~Outside function: mtxQ_trunc_d~~

-0.435580 0.670472 0.217707 0.529401 
-0.316785 0.144908 0.532534 -0.595437 
-0.623671 -0.486435 -0.344535 0.226417 
-0.544474 -0.205153 0.116951 -0.234891 
-0.049498 0.071150 -0.087421 0.235557 
-0.148493 0.495681 -0.727314 -0.450848 


~~Current Rarnk = 4~~



= = 🔍Check Orthogonality🔍 = = 

1.000000 0.000000 0.000000 0.000000 
0.000000 1.000000 -0.000000 -0.000000 
0.000000 -0.000000 1.000000 -0.000000 
0.000000 -0.000000 -0.000000 1.000000 


🔍🔍🔍Test case 3🔍🔍🔍


~~mtxZ~~

1.100000 2.200000 3.300000 4.400000 5.500000 
0.800000 1.600000 2.400000 3.200000 4.000000 
3.000000 4.100000 5.200000 6.300000 7.400000 
2.200000 3.300000 4.400000 5.500000 6.600000 
1.400000 2.500000 3.600000 4.700000 5.800000 
0.600000 1.500000 2.400000 3.300000 4.200000 
7.700000 5.600000 9.600000 8.800000 7.000000 


~~Outside function: mtxQ_trunc_d~~

-0.351490 -0.263976 0.326902 
-0.255629 -0.191983 0.237747 
-0.472913 -0.064907 -0.708444 
-0.421788 -0.148726 -0.272509 
-0.370662 -0.232544 0.163426 
-0.268410 -0.247413 0.430941 
-0.447351 0.867085 0.219182 


~~Current Rarnk = 3~~



= = 🔍Check Orthogonality🔍 = = 

1.000000 -0.000000 -0.000000 
-0.000000 1.000000 -0.000000 
-0.000000 -0.000000 1.000000 


🔍🔍🔍Test case 4🔍🔍🔍


~~mtxZ~~

1.200000 2.300000 3.400000 4.500000 5.600000 6.700000 7.800000 
0.900000 1.800000 2.700000 3.600000 4.500000 5.400000 6.300000 
3.100000 4.200000 5.300000 6.400000 7.500000 8.600000 9.700000 
2.300000 3.400000 4.500000 5.600000 6.700000 7.800000 8.900000 
1.500000 2.600000 3.700000 4.800000 5.900000 6.000000 7.100000 
0.700000 1.600000 2.500000 3.400000 4.300000 5.200000 6.100000 
2.100000 3.000000 4.900000 5.800000 6.700000 7.600000 8.500000 
3.300000 4.400000 5.500000 6.600000 7.700000 8.800000 9.900000 
0.100000 0.500000 0.300000 1.100000 1.900000 2.200000 2.900000 
0.200000 0.700000 0.400000 1.200000 1.500000 2.300000 3.100000 
0.300000 0.200000 0.500000 1.300000 1.800000 2.500000 3.200000 


~~Outside function: mtxQ_trunc_d~~

-0.329619 -0.190766 0.306499 -0.147336 -0.105426 -0.210807 0.181717 
-0.266231 -0.173692 0.281095 -0.126330 -0.087860 -0.178344 0.153961 
-0.409911 0.218189 -0.397669 -0.013060 -0.068218 -0.074600 0.059033 
-0.376104 0.045997 -0.101177 -0.069597 -0.083884 -0.131950 0.110690 
-0.300038 0.083483 0.277866 0.684617 -0.101722 0.550401 0.209307 
-0.257779 -0.216739 0.355218 -0.140464 -0.091776 -0.192681 0.166875 
-0.359200 0.299835 0.342931 -0.197235 0.425152 0.155807 -0.647720 
-0.418363 0.261237 -0.471792 0.001075 -0.064302 -0.060262 0.046119 
-0.122551 -0.480589 -0.150380 0.542684 -0.041537 -0.401700 -0.523267 
-0.131002 -0.480032 -0.195697 -0.367172 -0.407910 0.583697 -0.268705 
-0.135228 -0.458509 -0.232758 -0.025766 0.773111 0.175327 0.296989 


~~Current Rarnk = 7~~



= = 🔍Check Orthogonality🔍 = = 

1.000000 -0.000000 0.000000 -0.000000 -0.000000 -0.000000 0.000000 
-0.000000 1.000000 -0.000000 -0.000000 -0.000000 -0.000000 0.000000 
0.000000 -0.000000 1.000000 0.000000 -0.000000 -0.000000 0.000000 
-0.000000 -0.000000 0.000000 1.000000 0.000000 -0.000000 0.000000 
-0.000000 -0.000000 -0.000000 0.000000 1.000000 0.000000 0.000000 
-0.000000 -0.000000 -0.000000 -0.000000 0.000000 1.000000 -0.000000 
0.000000 0.000000 0.000000 0.000000 0.000000 -0.000000 1.000000 


 */