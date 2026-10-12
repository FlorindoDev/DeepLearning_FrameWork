#ifndef OPTIMIZER_HEADER
    #define OPTIMIZER_HEADER 0
    #include <arrayfire.h>

    void sgd(af_array* weight, af_array gradiente, double learning_rate);

#endif
