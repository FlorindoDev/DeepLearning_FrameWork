#include "autograd/graph.h"
#include <assert.h>
#include <math.h>

int main(void)
{
    enum { NUM_VAL = 3, MAX_NODI = 16, LARGHEZZA = 95, PASSO = 8 };
    node* val[NUM_VAL] = {NULL};
    node* nodi[MAX_NODI] = {NULL};
    int figli[MAX_NODI][2];
    int numero_nodi = 0, costruito = 0, stato = EXIT_FAILURE;
    af_err err = AF_SUCCESS;

    // Dati per colonne: A = [1 -2 3; -4 5 -6], B = [1 0; 0 1; 1 1], C = [1 0; 0 -2].
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

    // Solo la libreria costruisce il grafo. Questi puntatori sono temporanei:
    // non vengono conservati per la visita o per il disegno.
    {
        node* primo = add_op(val[0], val[1], OP_MATMUL);
        if (primo == NULL) goto visita;
        node* attivato = add_op(primo, NULL, OP_RELU);
        if (attivato == NULL) goto visita;
        node* secondo = add_op(attivato, val[2], OP_MATMUL);
        if (secondo == NULL) goto visita;
        node* terzo = add_op(attivato, secondo, OP_MATMUL);
        if (terzo == NULL) goto visita;
        node* somma = add_op(terzo, val[2], OP_SUM);
        if (somma == NULL) goto visita;
        node* finale = add_op(somma, NULL, OP_RELU);
        if (finale == NULL) goto visita;
        if (add_op(finale, NULL, OP_SOFTMAX) == NULL) goto visita;
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

    // Calcoliamo ordine topologico e livelli dai collegamenti scoperti.
    int entranti[MAX_NODI] = {0}, rimanenti[MAX_NODI], ordine[MAX_NODI];
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
        goto cleanup;
    }

    // Ogni operazione viene posizionata sotto i suoi operandi.
    int colonne[MAX_NODI], righe[MAX_NODI], lunghezze[MAX_NODI], ultima_riga = 0;
    char etichette[MAX_NODI][40];
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
        if (err != AF_SUCCESS) goto cleanup;
        lunghezze[i] = snprintf(etichette[i], sizeof etichette[i],
                               "[N%d %s %lldx%lld]", i,
                               nodi[i]->tipo == TYPE_VAL ? "VAL" :
                               nodi[i]->operazione == OP_MATMUL ? "MATMUL" :
                               nodi[i]->operazione == OP_RELU ? "RELU" :
                               nodi[i]->operazione == OP_SOFTMAX ? "SOFTMAX" :
                               nodi[i]->operazione == OP_SUM ? "SUM" : "OP",
                               (long long)dims[0], (long long)dims[1]);
        if (lunghezze[i] < 0 || lunghezze[i] >= (int)sizeof etichette[i] ||
            colonne[i] - lunghezze[i] / 2 < 0 ||
            colonne[i] + lunghezze[i] - lunghezze[i] / 2 > LARGHEZZA) {
            fprintf(stderr, "Grafo troppo largo per questo test.\n");
            goto cleanup;
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
                if (corsia < 0 || corsia >= LARGHEZZA) goto cleanup;
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

    printf("Grafo trovato partendo da val (N0, N1, N2 sono gli ingressi):\n\n");
    for (int y = 0; y <= ultima_riga; ++y) {
        int fine = LARGHEZZA;
        while (fine > 0 && disegno[y][fine - 1] == ' ') --fine;
        disegno[y][fine] = '\0';
        puts(disegno[y]);
    }

    // ReLU e softmax sono unarie: hanno un solo collegamento entrante.
    int uscite = 0, matmul = 0, somme = 0, relu = 0, softmax = 0;
    int collegamenti_ok = numero_nodi == 10;
    for (int i = 0; i < numero_nodi; ++i) {
        if (nodi[i]->tipo == TYPE_VAL) {
            if (i >= NUM_VAL || entranti[i] != 0) collegamenti_ok = 0;
        } else {
            int operandi = (nodi[i]->operazione == OP_RELU ||
                            nodi[i]->operazione == OP_SOFTMAX) ? 1 : 2;
            if (entranti[i] != operandi) collegamenti_ok = 0;
            if (nodi[i]->operazione == OP_MATMUL) ++matmul;
            else if (nodi[i]->operazione == OP_SUM) ++somme;
            else if (nodi[i]->operazione == OP_RELU) ++relu;
            else if (nodi[i]->operazione == OP_SOFTMAX) ++softmax;
            else collegamenti_ok = 0;
        }
        if (figli[i][0] < 0 && figli[i][1] < 0) ++uscite;
    }
    if (!collegamenti_ok || uscite != 1 || matmul != 3 || somme != 1 ||
        relu != 2 || softmax != 1) {
        fprintf(stderr, "Test fallito: collegamenti diversi da quelli attesi.\n");
        goto cleanup;
    }
    printf("\nTest collegamenti superato: 3 ingressi, 3 matmul, 1 somma, 2 ReLU, 1 softmax, un'uscita.\n");

    // Risultati calcolati a mano, in ordine di visita e in formato per colonne.
    const float attesi[10][6] = {
        {1, -4, -2, 5, 3, -6}, // N0: A
        {1, 0, 1, 0, 1, 1},    // N1: B
        {1, 0, 0, -2},         // N2: C
        {4, -10, 1, -1},       // N3: A * B
        {4, 0, -2, 0},         // N4: ReLU(A * B) * C
        {17, 0, -8, -2},       // N5: N7 + C
        {4, 0, 1, 0},          // N6: ReLU(A * B)
        {16, 0, -8, 0},        // N7: N6 * N4
        {17, 0, 0, 0},         // N8: ReLU(N5)
        {1.0f, 4.1399377e-8f, 0.5f, 0.5f} // N9: softmax(N8), per colonna
    };
    const op operazioni_attese[10] = {
        OP_NULL, OP_NULL, OP_NULL, OP_MATMUL, OP_MATMUL,
        OP_SUM, OP_RELU, OP_MATMUL, OP_RELU, OP_SOFTMAX
    };

    printf("\nValori di ogni nodo:\n");
    for (int i = 0; i < numero_nodi; ++i) {
        dim_t dims[4];
        err = af_get_dims(&dims[0], &dims[1], &dims[2], &dims[3], nodi[i]->valore);
        if (err != AF_SUCCESS) goto cleanup;
        dim_t righe_attese = i < NUM_VAL ? dimensioni[i][0] : 2;
        dim_t colonne_attese = i < NUM_VAL ? dimensioni[i][1] : 2;
        assert(dims[0] == righe_attese && dims[1] == colonne_attese);
        assert(dims[2] == 1 && dims[3] == 1);
        assert(nodi[i]->operazione == operazioni_attese[i]);
        if (dims[0] != righe_attese || dims[1] != colonne_attese ||
            dims[2] != 1 || dims[3] != 1) goto cleanup;

        float ottenuti[6];
        err = af_get_data_ptr(ottenuti, nodi[i]->valore);
        if (err != AF_SUCCESS) goto cleanup;
        printf("\n%s\n", etichette[i]);
        err = af_print_array(nodi[i]->valore);
        if (err != AF_SUCCESS) goto cleanup;
        fflush(stdout);

        // Gli interi sono esatti; per softmax tolleriamo l'arrotondamento in f32.
        for (dim_t j = 0; j < righe_attese * colonne_attese; ++j) {
            int corretto = nodi[i]->operazione == OP_SOFTMAX
                         ? fabsf(ottenuti[j] - attesi[i][j]) <=
                           1e-12f + 1e-6f * fabsf(attesi[i][j])
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
    stato = EXIT_SUCCESS;

cleanup:
    if (err != AF_SUCCESS)
        fprintf(stderr, "Errore ArrayFire: %d\n", (int)err);
    // Ogni nodo trovato viene rilasciato una volta, anche se e' condiviso.
    for (int i = 0; i < numero_nodi; ++i) {
        if (nodi[i]->gradiente) af_release_array(nodi[i]->gradiente);
        if (nodi[i]->valore) af_release_array(nodi[i]->valore);
        free(nodi[i]);
    }
    return stato;
}
