#ifndef GENERAL_HEADER
    #define GENERAL_HEADER 0
    #include <stdio.h>
    #include <stdlib.h>
    #include <arrayfire.h>

    typedef enum activation_fun{
        RELU
    }activation_fun;

    typedef enum loss_fun{
        CROSS_ENTROPY
    }loss_fun;

    static inline void stampa_matrice(af_array matrice){
        af_print_array(matrice);
    }
    
#endif
