#ifndef ACTIVATION_FUNCTION_HEADER
    #define ACTIVATION_FUNCTION_HEADER 0
    #include "../general.h"

    // L'input resta valido; il chiamante rilascia il risultato.
    af_array relu(af_array input);
    // In zero si usa la convenzione derivata = 0.
    af_array relu_derivative(af_array input);

    /*
        dim = 0: normalizza ogni colonna; dim = 1: ogni riga.
        Restituisce un nuovo array, oppure NULL in caso di errore.
    */
    af_array softmax(af_array input, int dim);
    // Predizioni softmax f64: un campione per riga, classi sulle colonne (dim=1).
    // Restituisce [campioni, classi, classi]: risultato[n, i, j] = d s[n,i] / d x[n,j].
    // L'input resta valido; il chiamante rilascia il risultato. NULL in caso di errore.
    af_array softmax_derivative(af_array predizioni);

#endif
