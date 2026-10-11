#include "autograd/graph.h"
#include "autograd/backpropagation.h"
#include <assert.h>
#include <math.h>

enum { NUM_VAL = 3, NUM_MISTI = 8, MAX_NODI = 16, LARGHEZZA = 95, PASSO = 8 };

// Disegna i collegamenti gia' presenti: non costruisce nodi e non cerca foglie.
static af_err stampa_grafo(int numero_grafo, node* const* nodi, int numero_nodi,
                           const int figli[][2], int* entranti, char etichette[][40])
{
    // Calcoliamo ordine topologico e livelli dai collegamenti scoperti.
    int rimanenti[MAX_NODI], ordine[MAX_NODI];
    af_err err = AF_SUCCESS;
    for (int i = 0; i < numero_nodi; ++i) entranti[i] = 0;
    int livello[MAX_NODI] = {0}, ordinati = 0;
    for (int i = 0; i < numero_nodi; ++i)
        for (int j = 0; j < 2; ++j)
            if (figli[i][j] >= 0) ++entranti[figli[i][j]];
    for (int i = 0; i < numero_nodi; ++i) {
        rimanenti[i] = entranti[i];
        if (entranti[i] == 0) ordine[ordinati++] = i;
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
            colonne[i] + lunghezze[i] - lunghezze[i] / 2 > LARGHEZZA) {
            fprintf(stderr, "Grafo troppo largo per questo test.\n");
            return AF_ERR_SIZE;
        }
    }

    // La griglia ASCII contiene solo i nodi e gli archi trovati dalla visita.
    char disegno[MAX_NODI * PASSO][LARGHEZZA + 1];
    for (int y = 0; y <= ultima_riga; ++y) {
        for (int x = 0; x < LARGHEZZA; ++x) disegno[y][x] = ' ';
        disegno[y][LARGHEZZA] = '\0';
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
                           ? 2 + deviazioni * 3 : LARGHEZZA - 3 - deviazioni * 3;
                ++deviazioni;
                if (corsia < 0 || corsia >= LARGHEZZA) return AF_ERR_SIZE;
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

    printf("\nGrafo %d (N0, N1, N2 sono gli ingressi):\n\n", numero_grafo);
    for (int y = 0; y <= ultima_riga; ++y) {
        int fine = LARGHEZZA;
        while (fine > 0 && disegno[y][fine - 1] == ' ') --fine;
        disegno[y][fine] = '\0';
        puts(disegno[y]);
    }

    return AF_SUCCESS;
}

int main(void)
{
    node* val[NUM_VAL] = {NULL};
    node* nodi[MAX_NODI] = {NULL};
    array_node nodes = {.node = val, .size = NUM_VAL, .length = NUM_VAL};
    array_node leaf = {.size = MAX_NODI, .length = 0};
    node* foglia_attesa = NULL;
    node* val_misti[NUM_VAL] = {NULL};
    node* nodi_misti[NUM_MISTI] = {NULL};
    array_node nodes_misti = {.node = val_misti, .size = NUM_VAL, .length = NUM_VAL};
    array_node leaf_miste = {.size = NUM_MISTI, .length = 0};
    int figli[MAX_NODI][2];
    int numero_nodi = 0, costruito = 0, stato = EXIT_FAILURE;
    af_err err = AF_SUCCESS;
    af_array target = 0;

    // A contiene un campione per riga; B e C sono gli altri ingressi del grafo.
    // I buffer sono in ordine per colonne, come richiesto da ArrayFire.
    float dati[NUM_VAL][6] = {
        {1, -4, -2, 5, 3, -6}, {1, 0, 1, 0, 1, 1}, {1, 0, 0, -2}
    };
    dim_t dimensioni[NUM_VAL][2] = {{2, 3}, {3, 2}, {2, 2}};

    for (int i = 0; i < NUM_VAL; ++i) {
        af_array ingresso = 0;
        err = af_create_array(&ingresso, dati[i], 2, dimensioni[i], f32);
        if (err != AF_SUCCESS) goto visita;
        val[i] = create_val(ingresso);
        if (val[i] == NULL) {
            af_release_array(ingresso);
            goto visita;
        }
    }

    // Target one-hot per riga: entrambi i campioni appartengono alla seconda classe.
    // Matrice [[0, 1], [0, 1]], memorizzata per colonne; parametro esterno della loss.
    float dati_target[] = {0, 0, 1, 1};
    err = af_create_array(&target, dati_target, 2, dimensioni[2], f32);
    if (err != AF_SUCCESS) goto visita;

    // Solo la libreria costruisce il grafo. La foglia attesa viene conservata
    // per gli assert; la ricerca riceve esclusivamente le entry val.
    {
        node* primo = add_op(val[0], val[1], NULL, OP_MATMUL);
        if (primo == NULL) goto visita;
        node* attivato = add_op(primo, NULL, NULL, OP_RELU);
        if (attivato == NULL) goto visita;
        node* secondo = add_op(attivato, val[2], NULL, OP_MATMUL);
        if (secondo == NULL) goto visita;
        node* terzo = add_op(attivato, secondo, NULL, OP_MATMUL);
        if (terzo == NULL) goto visita;
        node* somma = add_op(terzo, val[2], NULL, OP_SUM);
        if (somma == NULL) goto visita;
        node* finale = add_op(somma, NULL, NULL, OP_RELU);
        if (finale == NULL) goto visita;
        node* probabilita = add_op(finale, NULL, NULL, OP_SOFTMAX);
        if (probabilita == NULL) goto visita;
        foglia_attesa = add_op(probabilita, NULL, target, OP_CROSS_ENTROPY);
        if (foglia_attesa == NULL) goto visita;
        costruito = 1;
    }

visita:
    // Visita in ampiezza: l'unico punto di partenza e' il vettore val.
    for (int i = 0; i < MAX_NODI; ++i)
        figli[i][0] = figli[i][1] = -1;
    for (int i = 0; i < NUM_VAL; ++i)
        if (val[i] != NULL) nodi[numero_nodi++] = val[i];

    for (int i = 0; i < numero_nodi; ++i) {
        node* collegati[] = {nodi[i]->left, nodi[i]->right};
        for (int j = 0; j < 2; ++j) {
            if (collegati[j] == NULL) continue;
            int indice = 0;
            while (indice < numero_nodi && nodi[indice] != collegati[j]) ++indice;
            // Salviamo una sola volta anche i nodi condivisi.
            if (indice == numero_nodi) {
                if (numero_nodi == MAX_NODI) {
                    fprintf(stderr, "Troppi nodi per questo test.\n");
                    goto cleanup;
                }
                nodi[numero_nodi++] = collegati[j];
            }
            figli[i][j] = indice;
        }
    }
    if (!costruito) {
        fprintf(stderr, "Costruzione del grafo non completata.\n");
        goto cleanup;
    }

    // La ricerca parte solo dalle entry val; il risultato lo riempie find_leaf.
    leaf.node = calloc((size_t)leaf.size, sizeof *leaf.node);
    assert(leaf.node != NULL);
    if (leaf.node == NULL) goto cleanup;

    find_leaf(&nodes, &leaf);
    printf("\nGrafo 1 - Foglie trovate da find_leaf: %d\n", leaf.length);
    for (int i = 0; i < leaf.length; ++i) {
        int indice = 0;
        while (indice < numero_nodi && nodi[indice] != leaf.node[i]) ++indice;
        printf("Foglia %d: nodo N%d\n", i, indice);
        err = af_print_array(leaf.node[i]->valore);
        if (err != AF_SUCCESS) goto cleanup;
    }
    assert(leaf.length == 1);
    assert(leaf.node[0] == foglia_attesa);
    assert(leaf.node[0]->left == NULL && leaf.node[0]->right == NULL);
    for (int i = 0; i < NUM_VAL; ++i)
        assert(!is_node_in_array(&leaf, val[i]));

    // La stessa foglia e' raggiungibile da piu' entry: non deve essere duplicata.
    find_leaf(&nodes, &leaf);
    assert(leaf.length == 1);
    assert(leaf.node[0] == foglia_attesa);

    leaf.length = 0;
    find_leaf_helper(val[0], &leaf);
    assert(leaf.length == 1);
    assert(leaf.node[0] == foglia_attesa);

    // Nessuna entry: nessuna foglia.
    array_node vuoto = {0};
    leaf.length = 0;
    find_leaf(&vuoto, &leaf);
    assert(leaf.length == 0);

    printf("\nAssert grafo 1 superati: loss condivisa, nessun duplicato, helper ed entry vuote.\n");

    int entranti[MAX_NODI] = {0};
    char etichette[MAX_NODI][40];
    err = stampa_grafo(1, nodi, numero_nodi, figli, entranti, etichette);
    if (err != AF_SUCCESS) goto cleanup;

    // ReLU, softmax e loss hanno un solo collegamento entrante nel grafo.
    int uscite = 0, matmul = 0, somme = 0, relu = 0, softmax = 0, loss = 0;
    int collegamenti_ok = numero_nodi == 11;
    for (int i = 0; i < numero_nodi; ++i) {
        if (nodi[i]->tipo == TYPE_VAL) {
            if (i >= NUM_VAL || entranti[i] != 0) collegamenti_ok = 0;
        } else {
            int operandi = (nodi[i]->operazione == OP_RELU ||
                            nodi[i]->operazione == OP_SOFTMAX ||
                            nodi[i]->operazione == OP_CROSS_ENTROPY) ? 1 : 2;
            if (entranti[i] != operandi) collegamenti_ok = 0;
            if (nodi[i]->operazione == OP_MATMUL) ++matmul;
            else if (nodi[i]->operazione == OP_SUM) ++somme;
            else if (nodi[i]->operazione == OP_RELU) ++relu;
            else if (nodi[i]->operazione == OP_SOFTMAX) ++softmax;
            else if (nodi[i]->operazione == OP_CROSS_ENTROPY) ++loss;
            else collegamenti_ok = 0;
        }
        if (figli[i][0] < 0 && figli[i][1] < 0) ++uscite;
    }
    if (!collegamenti_ok || uscite != 1 || matmul != 3 || somme != 1 ||
        relu != 2 || softmax != 1 || loss != 1) {
        fprintf(stderr, "Test fallito: collegamenti diversi da quelli attesi.\n");
        goto cleanup;
    }
    printf("\nTest collegamenti superato: 3 ingressi, 3 matmul, 1 somma, 2 ReLU, 1 softmax, 1 cross-entropy, un'uscita.\n");

    // Risultati calcolati a mano, in ordine di visita e memorizzati per colonne.
    const double attesi[11][6] = {
        {1, -4, -2, 5, 3, -6}, // N0: A
        {1, 0, 1, 0, 1, 1},    // N1: B
        {1, 0, 0, -2},         // N2: C
        {4, -10, 1, -1},       // N3: A * B
        {4, 0, -2, 0},         // N4: ReLU(A * B) * C
        {17, 0, -8, -2},       // N5: N7 + C
        {4, 0, 1, 0},          // N6: ReLU(A * B)
        {16, 0, -8, 0},        // N7: N6 * N4
        {17, 0, 0, 0},         // N8: ReLU(N5)
        {1.0, 0.5, 4.1399377e-8, 0.5}, // N9: softmax(N8), classi lungo ogni riga
        {(17.0 + log(2.0)) / 2.0} // N10: cross-entropy(N9, target)
    };
    const op operazioni_attese[11] = {
        OP_NULL, OP_NULL, OP_NULL, OP_MATMUL, OP_MATMUL,
        OP_SUM, OP_RELU, OP_MATMUL, OP_RELU, OP_SOFTMAX, OP_CROSS_ENTROPY
    };

    printf("\nValori di ogni nodo:\n");
    for (int i = 0; i < numero_nodi; ++i) {
        dim_t dims[4];
        err = af_get_dims(&dims[0], &dims[1], &dims[2], &dims[3], nodi[i]->valore);
        if (err != AF_SUCCESS) goto cleanup;
        int nodo_loss = nodi[i]->operazione == OP_CROSS_ENTROPY;
        dim_t righe_attese = i < NUM_VAL ? dimensioni[i][0] : nodo_loss ? 1 : 2;
        dim_t colonne_attese = i < NUM_VAL ? dimensioni[i][1] : nodo_loss ? 1 : 2;
        assert(dims[0] == righe_attese && dims[1] == colonne_attese);
        assert(dims[2] == 1 && dims[3] == 1);
        assert(nodi[i]->operazione == operazioni_attese[i]);
        if (dims[0] != righe_attese || dims[1] != colonne_attese ||
            dims[2] != 1 || dims[3] != 1) goto cleanup;

        // La loss e' f64; convertiamo la lettura per verificare tutti i nodi.
        double ottenuti[6];
        af_array lettura = 0;
        err = af_cast(&lettura, nodi[i]->valore, f64);
        if (err != AF_SUCCESS) goto cleanup;
        err = af_get_data_ptr(ottenuti, lettura);
        af_release_array(lettura);
        if (err != AF_SUCCESS) goto cleanup;
        printf("\n%s\n", etichette[i]);
        err = af_print_array(nodi[i]->valore);
        if (err != AF_SUCCESS) goto cleanup;
        fflush(stdout);

        // Gli interi sono esatti; softmax e loss ammettono errori di arrotondamento.
        for (dim_t j = 0; j < righe_attese * colonne_attese; ++j) {
            int corretto = nodo_loss
                         ? fabs(ottenuti[j] - attesi[i][j]) <= 1e-6
                         : nodi[i]->operazione == OP_SOFTMAX
                         ? fabs(ottenuti[j] - attesi[i][j]) <=
                           1e-12 + 1e-6 * fabs(attesi[i][j])
                         : ottenuti[j] == attesi[i][j];
            if (!corretto) {
                fprintf(stderr, "N%d, elemento %lld: atteso %g, ottenuto %g.\n",
                        i, (long long)j, (double)attesi[i][j], (double)ottenuti[j]);
            }
            assert(corretto);
            if (!corretto) goto cleanup;
        }
    }
    printf("\nTutti gli assert sui valori dei nodi sono superati.\n");

    // Grafo 2: tre entry nuove. Il registro nodi_misti serve per stampa,
    // assert e cleanup; find_leaf riceve soltanto val_misti.
    float dati_misti[NUM_VAL][4] = {
        {1, -2, 3, -4}, {2, 1, 0, 2}, {1, 0, 0, 1}
    };
    dim_t dimensioni_miste[] = {2, 2};
    for (int i = 0; i < NUM_VAL; ++i) {
        af_array ingresso = 0;
        err = af_create_array(&ingresso, dati_misti[i], 2, dimensioni_miste, f32);
        if (err != AF_SUCCESS) goto cleanup;
        val_misti[i] = create_val(ingresso);
        if (val_misti[i] == NULL) {
            af_release_array(ingresso);
            goto cleanup;
        }
        nodi_misti[i] = val_misti[i];
    }

    nodi_misti[3] = add_op(val_misti[0], val_misti[1], NULL, OP_SUM);
    if (nodi_misti[3] == NULL) goto cleanup;
    nodi_misti[4] = add_op(val_misti[1], NULL, NULL, OP_RELU);
    if (nodi_misti[4] == NULL) goto cleanup;
    nodi_misti[5] = add_op(nodi_misti[3], NULL, NULL, OP_SOFTMAX);
    if (nodi_misti[5] == NULL) goto cleanup;
    nodi_misti[6] = add_op(nodi_misti[5], NULL, target, OP_CROSS_ENTROPY);
    if (nodi_misti[6] == NULL) goto cleanup;
    nodi_misti[7] = add_op(nodi_misti[3], val_misti[2], NULL, OP_MATMUL);
    if (nodi_misti[7] == NULL) goto cleanup;

    // Ricaviamo gli archi del disegno dai collegamenti reali del secondo grafo.
    int figli_misti[NUM_MISTI][2], entranti_misti[NUM_MISTI] = {0};
    char etichette_miste[NUM_MISTI][40];
    for (int i = 0; i < NUM_MISTI; ++i) {
        node* collegati[] = {nodi_misti[i]->left, nodi_misti[i]->right};
        for (int j = 0; j < 2; ++j) {
            figli_misti[i][j] = -1;
            if (collegati[j] == NULL) continue;
            int indice = 0;
            while (indice < NUM_MISTI && nodi_misti[indice] != collegati[j]) ++indice;
            if (indice == NUM_MISTI) {
                fprintf(stderr, "Nodo del grafo 2 non registrato.\n");
                goto cleanup;
            }
            figli_misti[i][j] = indice;
        }
    }
    err = stampa_grafo(2, nodi_misti, NUM_MISTI, figli_misti,
                       entranti_misti, etichette_miste);
    if (err != AF_SUCCESS) goto cleanup;

    // Solo le funzioni in prova riempiono l'array delle foglie.
    leaf_miste.node = calloc((size_t)leaf_miste.size, sizeof *leaf_miste.node);
    assert(leaf_miste.node != NULL);
    if (leaf_miste.node == NULL) goto cleanup;
    find_leaf(&nodes_misti, &leaf_miste);
    assert(leaf_miste.length == 3);

    // I puntatori attesi provengono dalla costruzione, senza una seconda ricerca.
    node* foglie_miste_attese[] = {nodi_misti[4], nodi_misti[6], nodi_misti[7]};
    for (int i = 0; i < 3; ++i) {
        int occorrenze = 0;
        for (int j = 0; j < leaf_miste.length; ++j)
            if (leaf_miste.node[j] == foglie_miste_attese[i]) ++occorrenze;
        assert(occorrenze == 1);
    }

    const char* nomi_operazioni[] = {
        "SUM", "MATMUL", "RELU", "SOFTMAX", "CROSS_ENTROPY", "NULL"
    };
    printf("\nGrafo 2 - Foglie trovate da find_leaf: %d\n", leaf_miste.length);
    for (int i = 0; i < leaf_miste.length; ++i) {
        node* trovata = leaf_miste.node[i];
        assert(trovata->left == NULL && trovata->right == NULL);
        int indice = 0;
        while (indice < NUM_MISTI && nodi_misti[indice] != trovata) ++indice;
        printf("Foglia %d: nodo N%d (%s)\n", i, indice,
               nomi_operazioni[trovata->operazione]);
        err = af_print_array(trovata->valore);
        if (err != AF_SUCCESS) goto cleanup;
    }

    find_leaf(&nodes_misti, &leaf_miste);
    assert(leaf_miste.length == 3);
    for (int i = 0; i < 3; ++i)
        assert(is_node_in_array(&leaf_miste, foglie_miste_attese[i]));

    leaf_miste.length = 0;
    for (int i = 0; i < nodes_misti.length; ++i)
        find_leaf_helper(nodes_misti.node[i], &leaf_miste);
    assert(leaf_miste.length == 3);
    for (int i = 0; i < 3; ++i) {
        int occorrenze = 0;
        for (int j = 0; j < leaf_miste.length; ++j)
            if (leaf_miste.node[j] == foglie_miste_attese[i]) ++occorrenze;
        assert(occorrenze == 1);
    }
    printf("Assert grafo 2 superati: tre foglie miste, nessun duplicato e helper.\n");
    stato = EXIT_SUCCESS;

cleanup:
    free(leaf.node);
    free(leaf_miste.node);
    for (int i = 0; i < NUM_MISTI; ++i) {
        if (nodi_misti[i] == NULL) continue;
        if (nodi_misti[i]->gradiente) af_release_array(nodi_misti[i]->gradiente);
        if (nodi_misti[i]->valore) af_release_array(nodi_misti[i]->valore);
        free(nodi_misti[i]);
    }
    if (err != AF_SUCCESS)
        fprintf(stderr, "Errore ArrayFire: %d\n", (int)err);
    if (target) af_release_array(target);
    // Ogni nodo trovato viene rilasciato una volta, anche se e' condiviso.
    for (int i = 0; i < numero_nodi; ++i) {
        if (nodi[i]->gradiente) af_release_array(nodi[i]->gradiente);
        if (nodi[i]->valore) af_release_array(nodi[i]->valore);
        free(nodi[i]);
    }
    return stato;
}
