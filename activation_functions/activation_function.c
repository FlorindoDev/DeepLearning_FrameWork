#include "activation_function.h"

float relu(float val){
    return val > 0 ? val : 0;

}

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
