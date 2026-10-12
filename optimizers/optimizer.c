#include "optimizer.h"

/*
################################################################
# SGD CLASSICO                                                 #
################################################################
*/

void sgd(af_array* weight, af_array gradiente, double learning_rate){
    af_dtype tipo;
    af_get_type(&tipo, *weight);
    dim_t forma_scalare[] = {1};
    af_array step = NULL, aggiornamento = NULL, nuovi_weight = NULL;

    af_constant(&step, learning_rate, 1, forma_scalare, tipo);
    af_mul(&aggiornamento, gradiente, step, true);
    af_sub(&nuovi_weight, *weight, aggiornamento, false);

    af_release_array(*weight);
    *weight = nuovi_weight;

    af_release_array(step);
    af_release_array(aggiornamento);
}
