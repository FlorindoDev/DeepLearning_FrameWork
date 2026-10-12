#ifndef LOSS_HEADER
    #define LOSS_HEADER 0
    #include "../general.h"


    af_array cross_entropy(af_array predizioni, af_array target);

    // -target / (N * predizioni) elemento per elemento, con predizioni > 0.
    // Input della stessa forma; N = campioni nella dimensione 0, risultato f64 senza clamp.
    // Gli input restano validi; il chiamante rilascia il risultato. NULL su errore.
    af_array cross_entropy_derivative(af_array predizioni, af_array target);


#endif
