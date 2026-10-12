#include "autograd/graph.h"
#include "autograd/backpropagation.h"
#include <assert.h>
#include <math.h>

enum { MAX_NODI = 12, LARGHEZZA = 200, PASSO = 4 };

// Disegna i collegamenti gia' presenti: non costruisce nodi e non cerca foglie.
static af_err stampa_grafo(int numero_grafo, node* const* nodi, int numero_nodi,
                           const int figli[][2], int* entranti, char etichette[][40])
{
    // Calcoliamo ordine topologico e livelli dai collegamenti scoperti.
    int rimanenti[MAX_NODI], ordine[MAX_NODI];
    af_err err = AF_SUCCESS;
    for (int i = 0; i < numero_nodi; ++i) entranti[i] = 0;
    int livello[MAX_NODI] = {0}, ordinati = 0, larghezza = 95;
    for (int i = 0; i < numero_nodi; ++i)
        for (int j = 0; j < 2; ++j)
            if (figli[i][j] >= 0) ++entranti[figli[i][j]];
    for (int i = 0; i < numero_nodi; ++i) {
        rimanenti[i] = entranti[i];
        if (entranti[i] == 0) {
            ordine[ordinati++] = i;
            if (18 + i * 24 + 20 > larghezza) larghezza = LARGHEZZA;
        }
    }
    for (int p = 0; p < ordinati; ++p) {
        int i = ordine[p];
        for (int j = 0; j < 2; ++j) {
            int dest = figli[i][j];
            if (dest < 0) continue;
            if (livello[dest] <= livello[i]) livello[dest] = livello[i] + 1;
            if (--rimanenti[dest] == 0) ordine[ordinati++] = dest;
        }
    }
    if (ordinati != numero_nodi) {
        fprintf(stderr, "Test fallito: il grafo contiene un ciclo.\n");
        return AF_ERR_SIZE;
    }

    // Ogni operazione viene posizionata sotto i suoi operandi.
    int colonne[MAX_NODI], righe[MAX_NODI], lunghezze[MAX_NODI], ultima_riga = 0;
    for (int p = 0; p < ordinati; ++p) {
        int i = ordine[p];
        if (entranti[i] == 0) {
            colonne[i] = 18 + i * 24;
        } else {
            int somma = 0;
            for (int k = 0; k < numero_nodi; ++k)
                for (int j = 0; j < 2; ++j)
                    if (figli[k][j] == i) somma += colonne[k];
            colonne[i] = somma / entranti[i];
        }
        righe[i] = livello[i] * PASSO;
        if (righe[i] > ultima_riga) ultima_riga = righe[i];

        dim_t dims[4];
        err = af_get_dims(&dims[0], &dims[1], &dims[2], &dims[3], nodi[i]->valore);
        if (err != AF_SUCCESS) return err;
        lunghezze[i] = snprintf(etichette[i], sizeof etichette[i],
                               "[N%d %s %lldx%lld]", i,
                               nodi[i]->tipo == TYPE_INPUT ? "INPUT" :
                               nodi[i]->tipo == TYPE_WEIGHT ? "WEIGHT" :
                               nodi[i]->tipo == TYPE_BIAS ? "BIAS" :
                               nodi[i]->tipo == TYPE_VAL ? "VAL" :
                               nodi[i]->operazione == OP_MATMUL ? "MATMUL" :
                               nodi[i]->operazione == OP_RELU ? "RELU" :
                               nodi[i]->operazione == OP_SOFTMAX ? "SOFTMAX" :
                               nodi[i]->operazione == OP_CROSS_ENTROPY ? "CROSS_ENTROPY" :
                               nodi[i]->operazione == OP_SUM ? "SUM" : "OP",
                               (long long)dims[0], (long long)dims[1]);
        // Manteniamo separate le etichette dei nodi sullo stesso livello.
        for (int q = 0; q < p; ++q) {
            int precedente = ordine[q];
            if (righe[precedente] != righe[i]) continue;
            int fine_precedente = colonne[precedente] - lunghezze[precedente] / 2
                                + lunghezze[precedente];
            if (colonne[i] - lunghezze[i] / 2 < fine_precedente + 2)
                colonne[i] = fine_precedente + 2 + lunghezze[i] / 2;
        }
        if (lunghezze[i] < 0 || lunghezze[i] >= (int)sizeof etichette[i] ||
            colonne[i] - lunghezze[i] / 2 < 0 ||
            colonne[i] + lunghezze[i] - lunghezze[i] / 2 > larghezza) {
            fprintf(stderr, "Grafo troppo largo per questo test.\n");
            return AF_ERR_SIZE;
        }
    }

    // La griglia ASCII contiene solo i nodi e gli archi trovati dalla visita.
    char disegno[MAX_NODI * PASSO][LARGHEZZA + 1];
    for (int y = 0; y <= ultima_riga; ++y) {
        for (int x = 0; x < larghezza; ++x) disegno[y][x] = ' ';
        disegno[y][larghezza] = '\0';
    }
    int deviazioni = 0;
    for (int i = 0; i < numero_nodi; ++i) {
        for (int j = 0; j < 2; ++j) {
            int dest = figli[i][j];
            if (dest < 0) continue;

            // Percorso a segmenti: gli archi lunghi aggirano i livelli intermedi.
            int punti[6][2], numero_punti;
            punti[0][0] = colonne[i];
            punti[0][1] = righe[i] + 1;
            if (livello[dest] == livello[i] + 1) {
                int mezzo = (righe[i] + righe[dest]) / 2;
                punti[1][0] = colonne[i];    punti[1][1] = mezzo;
                punti[2][0] = colonne[dest]; punti[2][1] = mezzo;
                numero_punti = 4;
            } else {
                int corsia = colonne[i] < colonne[dest]
                           ? 2 + deviazioni * 3 : larghezza - 3 - deviazioni * 3;
                ++deviazioni;
                if (corsia < 0 || corsia >= larghezza) return AF_ERR_SIZE;
                punti[1][0] = colonne[i];    punti[1][1] = righe[i] + 2;
                punti[2][0] = corsia;        punti[2][1] = righe[i] + 2;
                punti[3][0] = corsia;        punti[3][1] = righe[dest] - 2;
                punti[4][0] = colonne[dest]; punti[4][1] = righe[dest] - 2;
                numero_punti = 6;
            }
            punti[numero_punti - 1][0] = colonne[dest];
            punti[numero_punti - 1][1] = righe[dest] - 1;

            for (int s = 0; s < numero_punti - 1; ++s) {
                int x = punti[s][0], y = punti[s][1];
                int fine_x = punti[s + 1][0], fine_y = punti[s + 1][1];
                int dx = (fine_x > x) - (fine_x < x);
                int dy = (fine_y > y) - (fine_y < y);
                if (dx == 0 && dy == 0) continue;
                char tratto = dx != 0 ? '-' : '|';
                for (;;) {
                    if (disegno[y][x] == ' ') disegno[y][x] = tratto;
                    else if (disegno[y][x] != tratto) disegno[y][x] = '+';
                    if (x == fine_x && y == fine_y) break;
                    x += dx;
                    y += dy;
                }
            }
        }
    }
    for (int i = 0; i < numero_nodi; ++i) {
        if (entranti[i] > 0) disegno[righe[i] - 1][colonne[i]] = 'v';
        int inizio = colonne[i] - lunghezze[i] / 2;
        for (int c = 0; c < lunghezze[i]; ++c)
            disegno[righe[i]][inizio + c] = etichette[i][c];
    }

    printf("\nGrafo %d (entry:", numero_grafo);
    for (int i = 0; i < numero_nodi; ++i)
        if (entranti[i] == 0) printf(" N%d", i);
    printf("):\n\n");
    for (int y = 0; y <= ultima_riga; ++y) {
        int fine = larghezza;
        while (fine > 0 && disegno[y][fine - 1] == ' ') --fine;
        disegno[y][fine] = '\0';
        puts(disegno[y]);
    }

    return AF_SUCCESS;
}

// Legge soltanto il risultato della libreria: non calcola o assegna gradienti.
static int verifica_valori(const char* nome, af_array gradiente,
                             dim_t righe, dim_t colonne, const double* atteso)
{
    if (gradiente == NULL) {
        fprintf(stderr, "%s: array assente.\n", nome);
        return 0;
    }

    dim_t dims[4];
    af_err err = af_get_dims(&dims[0], &dims[1], &dims[2], &dims[3], gradiente);
    if (err != AF_SUCCESS || dims[0] != righe || dims[1] != colonne ||
        dims[2] != 1 || dims[3] != 1) {
        fprintf(stderr, "%s: dimensioni errate, attese %lldx%lld.\n",
                nome, (long long)righe, (long long)colonne);
        return 0;
    }

    af_array convertito = 0;
    double valori[4];
    assert(righe * colonne <= 4);
    err = af_cast(&convertito, gradiente, f64);
    if (err == AF_SUCCESS) err = af_get_data_ptr(valori, convertito);
    if (convertito) af_release_array(convertito);
    if (err != AF_SUCCESS) {
        fprintf(stderr, "%s: lettura dei valori fallita (%d).\n", nome, (int)err);
        return 0;
    }

    int corretto = 1;
    for (dim_t i = 0; i < righe * colonne; ++i) {
        int coincide = isfinite(valori[i]) && fabs(valori[i] - atteso[i]) < 1e-5;
        printf("%s[%lld]: ottenuto %.9f, atteso %.9f [%s]\n",
               nome, (long long)i, valori[i], atteso[i], coincide ? "OK" : "ERRORE");
        if (!coincide) corretto = 0;
    }
    return corretto;
}

static af_err test_rete_due_livelli(void)
{
    enum { X, W1, B1, W2, B2, M1, Z1, H, M2, Z2, P, LOSS, NUM_NODI };
    node* nodi[NUM_NODI] = {NULL};
    node* foglie[NUM_NODI] = {NULL};
    array_node ingressi = {.node = nodi, .size = 5, .length = 5};
    array_node leaf = {.node = foglie, .size = NUM_NODI, .length = 0};
    af_array target = 0;
    af_err err = AF_SUCCESS;
    int gradienti_corretti = 1;

    // Due campioni, due layer con bias: ReLU nel primo, softmax nel secondo.
    // H = ReLU(W1 * X + B1); P = softmax(H * W2 + B2).
    // Buffer per colonne. Il primo matmul segue la convenzione TYPE_INPUT.
    const float dati[5][4] = {
        {2, 1, 1, 2}, {1, -1, 1, 1}, {0.5f, -0.5f},
        {0.5f, 1, -0.5f, -1}, {0.75f, 0.25f}
    };
    const dim_t forme[5][2] = {{2, 2}, {2, 2}, {1, 2}, {2, 2}, {1, 2}};
    const node_type tipi[5] = {TYPE_INPUT, TYPE_WEIGHT, TYPE_BIAS, TYPE_WEIGHT, TYPE_BIAS};
    const float dati_target[4] = {0, 1, 1, 0};
    for (int i = 0; i < ingressi.length; ++i) {
        af_array valore = 0;
        err = af_create_array(&valore, dati[i], 2, forme[i], f32);
        if (err != AF_SUCCESS) goto cleanup;
        nodi[i] = create_val(valore, tipi[i]);
        if (nodi[i] == NULL) {
            af_release_array(valore);
            err = AF_ERR_NO_MEM;
            goto cleanup;
        }
    }
    err = af_create_array(&target, dati_target, 2, forme[W2], f32);
    if (err != AF_SUCCESS) goto cleanup;

    puts("Forward livello 1: W1 * X + B1 -> ReLU");
    nodi[M1] = add_op(nodi[W1], nodi[X], NULL, OP_MATMUL);
    if (nodi[M1] == NULL) { err = AF_ERR_NO_MEM; goto cleanup; }
    nodi[Z1] = add_op(nodi[M1], nodi[B1], NULL, OP_SUM);
    if (nodi[Z1] == NULL) { err = AF_ERR_NO_MEM; goto cleanup; }
    nodi[H] = add_op(nodi[Z1], NULL, NULL, OP_RELU);
    if (nodi[H] == NULL) { err = AF_ERR_NO_MEM; goto cleanup; }
    puts("Forward livello 2: H * W2 + B2 -> softmax");
    nodi[M2] = add_op(nodi[H], nodi[W2], NULL, OP_MATMUL);
    if (nodi[M2] == NULL) { err = AF_ERR_NO_MEM; goto cleanup; }
    nodi[Z2] = add_op(nodi[M2], nodi[B2], NULL, OP_SUM);
    if (nodi[Z2] == NULL) { err = AF_ERR_NO_MEM; goto cleanup; }
    puts("Forward uscita: softmax -> cross-entropy");
    nodi[P] = add_op(nodi[Z2], NULL, NULL, OP_SOFTMAX);
    if (nodi[P] == NULL) { err = AF_ERR_NO_MEM; goto cleanup; }
    nodi[LOSS] = add_op(nodi[P], NULL, target, OP_CROSS_ENTROPY);
    if (nodi[LOSS] == NULL) { err = AF_ERR_NO_MEM; goto cleanup; }

    // Il disegno usa i collegamenti effettivi creati da add_op.
    int figli[NUM_NODI][2], entranti[NUM_NODI];
    char etichette[NUM_NODI][40];
    for (int i = 0; i < NUM_NODI; ++i) {
        node* collegati[] = {nodi[i]->left, nodi[i]->right};
        for (int j = 0; j < 2; ++j) {
            figli[i][j] = -1;
            if (collegati[j] == NULL) continue;
            int indice = 0;
            while (indice < NUM_NODI && nodi[indice] != collegati[j]) ++indice;
            assert(indice < NUM_NODI);
            if (indice == NUM_NODI) { err = AF_ERR_SIZE; goto cleanup; }
            figli[i][j] = indice;
        }
    }
    err = stampa_grafo(1, nodi, NUM_NODI, figli, entranti, etichette);
    if (err != AF_SUCCESS) goto cleanup;

    // Risultati indipendenti, calcolati a mano:
    // M1 = [[3,3],[-1,1]], Z1 = [[3.5,2.5],[-0.5,0.5]].
    // H = [[3.5,2.5],[0,0.5]], Z2 = [[5,-4],[1.25,-0.25]].
    // p = sigmoid(9), q = sigmoid(1.5); P = [[p,1-p],[q,1-q]].
    // Loss media: (log(1+exp(9)) + log(1+exp(-1.5))) / 2.
    // Tutti i valori attesi sono per colonne, come i buffer ArrayFire.
    const double atteso_m1[] = {3, -1, 3, 1};
    const double atteso_z1[] = {3.5, -0.5, 2.5, 0.5};
    const double atteso_h[] = {3.5, 0, 2.5, 0.5};
    const double atteso_m2[] = {4.25, 0.5, -4.25, -0.5};
    const double atteso_z2[] = {5, 1.25, -4, -0.25};
    const double atteso_p[] = {0.9998766054240137, 0.8175744761936437,
                              0.0001233945759863, 0.1824255238063563};
    const double atteso_loss[] = {4.600768340086238};
    printf("\nRete 2 -> 2 -> 2, due campioni, bias a entrambi i livelli, ReLU -> softmax\n");
    int forward_corretto = verifica_valori("W1 * X", nodi[M1]->valore, 2, 2, atteso_m1);
    forward_corretto &= verifica_valori("Z1 = W1 * X + B1", nodi[Z1]->valore, 2, 2, atteso_z1);
    forward_corretto &= verifica_valori("H = ReLU(Z1)", nodi[H]->valore, 2, 2, atteso_h);
    forward_corretto &= verifica_valori("H * W2", nodi[M2]->valore, 2, 2, atteso_m2);
    forward_corretto &= verifica_valori("Z2 = H * W2 + B2", nodi[Z2]->valore, 2, 2, atteso_z2);
    forward_corretto &= verifica_valori("P (forward)", nodi[P]->valore, 2, 2, atteso_p);
    forward_corretto &= verifica_valori("Loss (forward)", nodi[LOSS]->valore, 1, 1, atteso_loss);
    assert(forward_corretto && "Forward della rete a due livelli errato");
    if (!forward_corretto) { err = AF_ERR_RUNTIME; goto cleanup; }

    find_leaf(&ingressi, &leaf);
    assert(leaf.length == 1 && leaf.node[0] == nodi[LOSS]);
    if (leaf.length != 1 || leaf.node[0] != nodi[LOSS]) {
        err = AF_ERR_RUNTIME;
        goto cleanup;
    }
    assert(leaf.node[0]->left == NULL && leaf.node[0]->right == NULL);
    for (int i = 0; i < ingressi.length; ++i)
        assert(!is_node_in_array(&leaf, ingressi.node[i]));

    // La loss e' raggiunta da tutti e cinque gli ingressi: nessun duplicato,
    // anche ripetendo la ricerca sullo stesso array.
    find_leaf(&ingressi, &leaf);
    assert(leaf.length == 1 && leaf.node[0] == nodi[LOSS]);
    if (leaf.length != 1 || leaf.node[0] != nodi[LOSS]) {
        err = AF_ERR_RUNTIME;
        goto cleanup;
    }
    puts("Assert foglie superati: unica foglia CROSS_ENTROPY, nessun duplicato.");

    // Unica chiamata che produce i gradienti: implementazione dell'utente.
    fflush(stdout);
    backwards(&leaf, target);

    // a = p/2, b = (q-1)/2: il fattore 1/2 viene dalla loss media.
    // dZ2 = [[a,-a],[b,-b]].
    // dW2 = H^T * dZ2; dH = dZ2 * W2^T; dZ1 = dH * maschera_ReLU(Z1).
    // dW1 = dZ1 * X^T; dB1 e dB2 sono le somme sui campioni.
    const double atteso_dw1[] = {1.9997532108480274, -0.1824255238063563,
                                2.4996915135600344, -0.3648510476127127};
    const double atteso_db1[] = {0.4999383027120068, 0.8174510816176573};
    const double atteso_dw2[] = {1.7497840594920240, 1.2042393758284282,
                                -1.7497840594920240, -1.2042393758284282};
    const double atteso_db2[] = {0.4087255408088287, -0.4087255408088287};
    const double atteso_dh[] = {0.4999383027120068, -0.0912127619031782,
                               0.9998766054240137, -0.1824255238063563};
    const double atteso_dz1[] = {0.4999383027120068, 0, 0.9998766054240137, -0.1824255238063563};
    const double atteso_dz2[] = {0.4999383027120068, -0.0912127619031782,
                                -0.4999383027120068, 0.0912127619031782};
    gradienti_corretti &= verifica_valori("dW1", nodi[W1]->gradiente, 2, 2, atteso_dw1);
    gradienti_corretti &= verifica_valori("dB1", nodi[B1]->gradiente, 1, 2, atteso_db1);
    gradienti_corretti &= verifica_valori("dW2", nodi[W2]->gradiente, 2, 2, atteso_dw2);
    gradienti_corretti &= verifica_valori("dB2", nodi[B2]->gradiente, 1, 2, atteso_db2);
    gradienti_corretti &= verifica_valori("dH", nodi[H]->gradiente, 2, 2, atteso_dh);
    gradienti_corretti &= verifica_valori("dZ1", nodi[Z1]->gradiente, 2, 2, atteso_dz1);
    gradienti_corretti &= verifica_valori("dZ2", nodi[Z2]->gradiente, 2, 2, atteso_dz2);
    fflush(stdout);

cleanup:
    if (target) af_release_array(target);
    // matmul_gradient puo' condividere lo stesso handle tra nodi:
    // il cleanup rilascia ogni handle distinto una volta, dopo i confronti.
    for (int i = 0; i < NUM_NODI; ++i) {
        if (nodi[i] == NULL || nodi[i]->gradiente == NULL) continue;
        int gia_visto = 0;
        for (int j = 0; j < i; ++j)
            if (nodi[j] && nodi[j]->gradiente == nodi[i]->gradiente) gia_visto = 1;
        if (!gia_visto) af_release_array(nodi[i]->gradiente);
    }
    for (int i = 0; i < NUM_NODI; ++i) {
        if (nodi[i] == NULL) continue;
        if (nodi[i]->valore) af_release_array(nodi[i]->valore);
        free(nodi[i]);
    }
    assert(gradienti_corretti && "backwards: gradienti diversi dai valori attesi");
    if (!gradienti_corretti) return AF_ERR_RUNTIME;
    if (err == AF_SUCCESS) puts("Assert backwards a due livelli superati.");
    return err;
}

int main(void)
{
    // Mantiene visibile l'ultimo passaggio anche se la libreria si interrompe.
    setvbuf(stdout, NULL, _IONBF, 0);
    af_err err = test_rete_due_livelli();
    if (err != AF_SUCCESS) {
        fprintf(stderr, "Test fallito (errore %d).\n", (int)err);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
