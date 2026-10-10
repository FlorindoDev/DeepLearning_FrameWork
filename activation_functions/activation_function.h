#ifndef ACTIVATION_FUNCTION_HEADER
    #define ACTIVATION_FUNCTION_HEADER 0
    #include "../general.h"


    float relu(float);

    /*
        dim = 0: normalizza ogni colonna; dim = 1: ogni riga.
        Restituisce un nuovo array, oppure NULL in caso di errore.
    */
    af_array softmax(af_array input, int dim);
    


#endif
