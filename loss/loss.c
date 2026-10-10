#include "loss.h"

af_array cross_entropy(af_array predizioni, af_array target)
{
    af_array y = 0, t = 0, minimo = 0, massimo = 0, meno_uno = 0;
    af_array limitate = 0, logaritmi = 0, pesate = 0;
    af_array per_riga = 0, media = 0, risultato = 0;
    dim_t scalare[] = {1};

    af_cast(&y, predizioni, f64);
    af_cast(&t, target, f64);

    //serve per non avere valori sopra 1 o negativi
    af_constant(&minimo, 1e-12, 1, scalare, f64);
    af_constant(&massimo, 1.0, 1, scalare, f64);
    af_clamp(&limitate, y, minimo, massimo, true);

    
    af_log(&logaritmi, limitate);
    af_mul(&pesate, t, logaritmi, false);
    af_sum(&per_riga, pesate, 1);
    af_mean(&media, per_riga, 0);

    //serve per mettere il meno davanti alla sommaotria come da formula
    af_constant(&meno_uno, -1.0, 1, scalare, f64);
    af_mul(&risultato, media, meno_uno, false);

    
    af_array temporanei[] = {
        y, t, minimo, massimo, meno_uno, limitate,
        logaritmi, pesate, per_riga, media
    };

    const size_t numero_temporanei = sizeof temporanei / sizeof temporanei[0]; //numero di elementi totali size totale/size di un elmento = numero di elementi
    for (size_t i = 0; i < numero_temporanei; i++) {
        if (temporanei[i] != NULL) {
            af_release_array(temporanei[i]);
        }
    }


    return risultato;
}
