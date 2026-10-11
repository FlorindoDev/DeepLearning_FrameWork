#ifndef LOSS_HEADER
    #define LOSS_HEADER 0
    #include "../general.h"


    af_array cross_entropy(af_array predizioni, af_array target);

    // d(-target * log(predizione))/d predizione, con predizione > 0.
    // Singolo termine: non include il clamp o la media sul batch della loss.
    float cross_entropy_derivative(float predizione, float target);


#endif
