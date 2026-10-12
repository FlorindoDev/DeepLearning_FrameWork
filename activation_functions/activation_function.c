#include "activation_function.h"

/*
################################################################
# RELU                                                         #
################################################################
*/

double relu(double val){
    return val > 0 ? val : 0;

}

double relu_derivative(double val){
    return val > 0 ? 1.0 : 0.0;
}

/*
################################################################
# SOFTMAX                                                      #
################################################################
*/

af_array softmax(af_array input, int dim){

    af_array massimi = 0, centrata = 0, esponenziali = 0;
    af_array somme = 0, risultato = 0;

    // centrata = input - massimo di ogni gruppo lungo dim (riga se dim=0, colonna se dim=1).
    // La softmax resta invariata, ma exp riceve valori <= 0 ed evita overflow.
    af_max(&massimi, input, dim);
    af_sub(&centrata, input, massimi, true); 

    af_exp(&esponenziali, centrata);
    af_sum(&somme, esponenziali, dim);
    af_div(&risultato, esponenziali, somme, true);

    af_release_array(input);
    af_release_array(massimi);
    af_release_array(centrata);
    af_release_array(esponenziali);
    af_release_array(somme);

    return risultato;

}

af_array softmax_derivative(af_array predizioni){
    if (predizioni == NULL) return NULL;

    dim_t dims[4];
    af_dtype tipo;
    af_get_dims(&dims[0], &dims[1], &dims[2], &dims[3], predizioni);
    af_get_type(&tipo, predizioni); 



    af_array identita = 0, delta = 0, predizioni_j = 0;
    af_array differenze = 0, risultato = 0;
    dim_t forma_identita[] = {dims[1], dims[1]};
    dim_t forma_delta[] = {1, dims[1], dims[1]};

    // risultato[n, i, j] = predizioni[n, i] * (delta[i, j] - predizioni[n, j]).
    af_identity(&identita, 2, forma_identita, tipo);
    af_moddims(&delta, identita, 3, forma_delta);
    af_reorder(&predizioni_j, predizioni, 0, 2, 1, 3);
    af_sub(&differenze, delta, predizioni_j, true);
    af_mul(&risultato, predizioni, differenze, true);
    
    af_release_array(identita);
    af_release_array(delta);
    af_release_array(predizioni_j);
    af_release_array(differenze);

    return risultato;
}
