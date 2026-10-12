#include "models/MLP/MLP.h"
#include <math.h>
#include <stdint.h>
#include <time.h>

/*
################################################################
# CONFIGURAZIONE MNIST                                         #
################################################################
*/

enum { PIXEL = 784, CLASSI = 10, NASCOSTI = 64, BATCH = 64, EPOCHE = 5 };

typedef struct {
    unsigned char* immagini;
    unsigned char* etichette;
    int numero;
} dati_mnist;

static void banner(const char* titolo)
{
    puts("\n################################################################");
    printf("# %s\n", titolo);
    puts("################################################################");
}

static void verifica(int condizione, const char* messaggio)
{
    if (!condizione) {
        fprintf(stderr, "\n[ERRORE] %s\n", messaggio);
        exit(EXIT_FAILURE);
    }
}

static af_array matrice(const double* valori, dim_t righe, dim_t colonne)
{
    af_array risultato = NULL;
    dim_t forma[] = {righe, colonne};
    verifica(af_create_array(&risultato, valori, 2, forma, f64) == AF_SUCCESS,
             "Creazione della matrice.");
    return risultato;
}

static void verifica_forma(af_array matrice, dim_t righe, dim_t colonne)
{
    dim_t d[4];
    verifica(af_get_dims(&d[0], &d[1], &d[2], &d[3], matrice) == AF_SUCCESS,
             "Matrice assente o non valida.");
    verifica(d[0] == righe && d[1] == colonne && d[2] == 1 && d[3] == 1,
             "Forma della matrice diversa da quella attesa.");
}

static double scalare(af_array matrice)
{
    double valore;
    verifica_forma(matrice, 1, 1);
    verifica(af_get_scalar(&valore, matrice) == AF_SUCCESS && isfinite(valore),
             "Loss non valida.");
    return valore;
}

/*
################################################################
# STAMPA ASCII DEI GRAFI                                       #
################################################################
*/

enum { MAX_NODI = 16, LARGHEZZA = 200, PASSO = 4 };

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
            if (18 + (ordinati - 1) * 24 + 20 > larghezza) larghezza = LARGHEZZA;
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
            colonne[i] = 18 + p * 24;
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

static void stampa_modello(int numero_grafo, array_node* graph, node* const* ramo, int numero_ramo)
{
    node* nodi[MAX_NODI];
    int numero_nodi = graph->length;
    verifica(numero_nodi <= MAX_NODI, "Troppi nodi per il disegno del grafo.");
    for (int i = 0; i < graph->length; ++i) nodi[i] = graph->node[i];
    for (node* corrente = graph->node[graph->length - 1]->left;
         corrente != NULL; corrente = corrente->left) {
        verifica(numero_nodi < MAX_NODI, "Troppi nodi per il disegno del grafo.");
        nodi[numero_nodi++] = corrente;
    }
    for (int i = 0; i < numero_ramo; ++i) {
        verifica(numero_nodi < MAX_NODI, "Troppi nodi per il disegno del grafo.");
        nodi[numero_nodi++] = ramo[i];
    }

    int figli[MAX_NODI][2], entranti[MAX_NODI];
    char etichette[MAX_NODI][40];
    for (int i = 0; i < numero_nodi; ++i) {
        node* collegati[] = {nodi[i]->left, nodi[i]->right};
        for (int j = 0; j < 2; ++j) {
            figli[i][j] = -1;
            if (collegati[j] == NULL) continue;
            int indice = 0;
            while (indice < numero_nodi && nodi[indice] != collegati[j]) ++indice;
            verifica(indice < numero_nodi, "Collegamento a un nodo assente dal disegno.");
            figli[i][j] = indice;
        }
    }
    verifica(stampa_grafo(numero_grafo, nodi, numero_nodi, figli, entranti, etichette) == AF_SUCCESS,
             "Stampa del grafo non riuscita.");
}


/*
################################################################
# TEST DEL GRAFO: DATI NOTI E COSTRUZIONE                       #
################################################################
*/

// Campioni sulle righe; i buffer ArrayFire sono memorizzati per colonne.
static const double TEST_X[] = {0.2, 0.5, -0.1, 0.7, -0.4, 0.3, -0.5, 0.6, 0.8};
static const double TEST_Y[] = {1, 0, 1, 0, 1, 0};
static const dim_t TEST_FORME[4][2] = {{3, 4}, {1, 4}, {4, 2}, {1, 2}};
static const int TEST_ELEMENTI[] = {12, 4, 8, 2, 2};
static const double TEST_PARAMETRI[5][12] = {
    {0.1, -0.2, 0.3, -0.4, 0.2, 0.1, 0.2, 0.3, -0.1, -0.2, 0.1, 0.4},
    {0.15, -0.3, 0.2, 0.1},
    {0.2, -0.1, 0.3, -0.2, -0.3, 0.2, 0.1, 0.4},
    {0.1, -0.1},
    {0.1, -0.2}  // Bias del secondo ramo, usato solo nel test con due loss.
};

static void imposta_parametro(array_node* graph, int indice, const double* valori)
{
    af_release_array(graph->node[indice]->valore);
    graph->node[indice]->valore = matrice(valori, TEST_FORME[indice][0], TEST_FORME[indice][1]);
}

static array_node* grafo_test(void)
{
    // Fixture dei soli test di graph.h: non usa le funzioni interne della MLP.
    array_node* graph = calloc(1, sizeof *graph);
    graph->size = graph->length = 5;
    graph->node = calloc((size_t)graph->size, sizeof *graph->node);
    for (int i = 0; i < 4; ++i)
        graph->node[i] = create_val(matrice(TEST_PARAMETRI[i], TEST_FORME[i][0], TEST_FORME[i][1]),
                                    i % 2 == 0 ? TYPE_WEIGHT : TYPE_BIAS);
    graph->node[4] = create_val(matrice(TEST_X, 3, 3), TYPE_INPUT);
    verifica(graph->size == 5 && graph->length == 5, "Numero di nodi iniziali.");
    for (int i = 0; i < 4; ++i) {
        verifica(graph->node[i]->tipo == (i % 2 == 0 ? TYPE_WEIGHT : TYPE_BIAS),
                 "Tipo del parametro nel grafo.");
        verifica_forma(graph->node[i]->valore, TEST_FORME[i][0], TEST_FORME[i][1]);
    }
    verifica(graph->node[4]->tipo == TYPE_INPUT, "N4 deve essere l'input.");
    verifica_forma(graph->node[4]->valore, 3, 3);
    return graph;
}

static void libera_nodo_test(node* nodo)
{
    if (nodo->valore != NULL) af_release_array(nodo->valore);
    if (nodo->gradiente != NULL) af_release_array(nodo->gradiente);
    if (nodo->temp != NULL) af_release_array(nodo->temp);
    free(nodo);
}

static void libera_operazioni_grafo(array_node* graph)
{
    node* corrente = graph->node[graph->length - 1]->left;
    while (corrente != NULL) {
        node* successivo = corrente->left;
        libera_nodo_test(corrente);
        corrente = successivo;
    }
    for (int i = 0; i < graph->length; ++i) {
        node* nodo = graph->node[i];
        if (nodo->gradiente != NULL) af_release_array(nodo->gradiente);
        if (nodo->temp != NULL) af_release_array(nodo->temp);
        nodo->gradiente = nodo->temp = NULL;
        nodo->left = nodo->right = NULL;
    }
}

static void libera_grafo(array_node* graph)
{
    libera_operazioni_grafo(graph);
    for (int i = 0; i < graph->length; ++i) libera_nodo_test(graph->node[i]);
    free(graph->node);
    free(graph);
}

static node* forward_grafo_test(array_node* graph)
{
    libera_operazioni_grafo(graph);
    node* output = add_op(graph->node[4], graph->node[0], NULL, OP_MATMUL);
    output = add_op(output, graph->node[1], NULL, OP_SUM);
    output = add_op(output, NULL, NULL, OP_RELU);
    output = add_op(output, graph->node[2], NULL, OP_MATMUL);
    output = add_op(output, graph->node[3], NULL, OP_SUM);
    return add_op(output, NULL, NULL, OP_SOFTMAX);
}

static void test_struttura_forward(void)
{
    banner("TEST 1 - NODI, COLLEGAMENTI E FORWARD");
    array_node* graph = grafo_test();
    node* output = forward_grafo_test(graph);
    af_array target = matrice(TEST_Y, 3, 2);
    node* loss = add_op(output, NULL, target, OP_CROSS_ENTROPY);

    // N0 W1, N1 B1, N2 W2, N3 B2, N4 X; N5..N11 sono le operazioni.
    node* nodi[12];
    for (int i = 0; i < 5; ++i) nodi[i] = graph->node[i];
    node* corrente = graph->node[4]->left;
    const op operazioni[] = {OP_MATMUL, OP_SUM, OP_RELU, OP_MATMUL,
                             OP_SUM, OP_SOFTMAX, OP_CROSS_ENTROPY};
    const int padri[][2] = {{4, 0}, {5, 1}, {6, -1}, {7, 2},
                            {8, 3}, {9, -1}, {10, -1}};
    for (int i = 0; i < 7; ++i) {
        verifica(corrente != NULL && corrente->operazione == operazioni[i],
                 "Ordine delle operazioni del grafo.");
        nodi[i + 5] = corrente;
        verifica(corrente->father_left == nodi[padri[i][0]], "Collegamento al padre sinistro.");
        verifica(corrente->father_right == (padri[i][1] < 0 ? NULL : nodi[padri[i][1]]),
                 "Collegamento al padre destro.");
        verifica(corrente->right == NULL, "Ramo inatteso nel grafo sequenziale.");
        corrente = corrente->left;
    }
    verifica(corrente == NULL && nodi[11] == loss, "La loss deve terminare il grafo.");
    for (int i = 0; i < 4; ++i)
        verifica(graph->node[i]->left == nodi[5 + (i / 2) * 3 + i % 2],
                 "Collegamento dal parametro alla sua operazione.");

    stampa_modello(1, graph, NULL, 0);

    const double atteso[] = {0.5744425168116590, 0.5827323189732048, 0.5289675250309517,
                             0.4255574831883410, 0.4172676810267954, 0.4710324749690482};
    double ottenuto[6];
    verifica_forma(output->valore, 3, 2);
    verifica(af_get_data_ptr(ottenuto, output->valore) == AF_SUCCESS, "Lettura del forward.");
    for (int i = 0; i < 6; ++i)
        verifica(isfinite(ottenuto[i]) && fabs(ottenuto[i] - atteso[i]) < 1e-12,
                 "Forward diverso dai valori attesi.");

    node* trovate[2];
    array_node foglie = {.node = trovate, .size = 2, .length = 0};
    find_leaf(graph, &foglie);
    verifica(foglie.length == 1 && foglie.node[0] == loss, "Ricerca della foglia loss.");
    verifica(scalare(loss->valore) > 0.0, "Loss del grafo di prova.");
    libera_grafo(graph);
    af_release_array(target);
    puts("[OK] N0..N11: forme, collegamenti, probabilita attese e ricerca foglie.");
}

/*
################################################################
# TEST DEL GRAFO: GRADIENTI E DUE LOSS CONDIVISE                 #
################################################################
*/

static double loss_grafo_test(array_node* graph, af_array target, int due_loss,
                              const double* bias_ramo, double gradienti[5][12], int mostra_grafo)
{
    node* output = forward_grafo_test(graph);
    node* condiviso = output->father_left;  // N9: logits prima della softmax.
    add_op(output, NULL, target, OP_CROSS_ENTROPY);
    node* ramo[4] = {NULL};
    if (due_loss) {
        ramo[0] = create_val(matrice(bias_ramo, 1, 2), TYPE_BIAS);        // N12
        ramo[1] = add_op(condiviso, ramo[0], NULL, OP_SUM);              // N13
        ramo[2] = add_op(ramo[1], NULL, NULL, OP_SOFTMAX);              // N14
        ramo[3] = add_op(ramo[2], NULL, target, OP_CROSS_ENTROPY);       // N15
        verifica(have_tow_child(condiviso), "N9 deve essere condiviso dai due rami.");
    }

    if (mostra_grafo) stampa_modello(2, graph, ramo, 4);

    node* trovate[2];
    array_node foglie = {.node = trovate, .size = 2, .length = 0};
    find_leaf(graph, &foglie);
    verifica(foglie.length == 1 + due_loss, "Numero di loss trovate nel grafo.");
    double totale = 0.0;
    for (int i = 0; i < foglie.length; ++i) totale += scalare(foglie.node[i]->valore);

    if (gradienti != NULL) {
        backwards(&foglie, target);
        for (int i = 0; i < 4 + due_loss; ++i) {
            node* parametro = i < 4 ? graph->node[i] : ramo[0];
            verifica_forma(parametro->gradiente, i < 4 ? TEST_FORME[i][0] : 1,
                            i < 4 ? TEST_FORME[i][1] : 2);
            verifica(af_get_data_ptr(gradienti[i], parametro->gradiente) == AF_SUCCESS,
                     "Lettura dei gradienti.");
        }
    }

    if (due_loss) {
        // Liberiamo il ramo separato prima della catena principale del test.
        condiviso->right = NULL;
        for (int i = 0; i < 4; ++i) libera_nodo_test(ramo[i]);
    }
    libera_operazioni_grafo(graph);
    return totale;
}

static void test_gradienti(int due_loss)
{
    banner(due_loss ? "TEST 3 - DUE LOSS: ACCUMULO SUL GRAFO CONDIVISO"
                    : "TEST 2 - BACKWARD: GRADIENTI NUMERICI");
    array_node* graph = grafo_test();
    af_array target = matrice(TEST_Y, 3, 2);
    double gradienti[5][12], bias_ramo[] = {0.1, -0.2};
    loss_grafo_test(graph, target, due_loss, bias_ramo, gradienti, due_loss);
    const double epsilon = 1e-6;
    double massimo_errore = 0.0;
    int verificati = 0;

    for (int i = 0; i < 4 + due_loss; ++i) {
        double valori[12];
        for (int j = 0; j < TEST_ELEMENTI[i]; ++j) valori[j] = TEST_PARAMETRI[i][j];
        for (int j = 0; j < TEST_ELEMENTI[i]; ++j) {
            double originale = valori[j];
            valori[j] = originale + epsilon;
            if (i < 4) imposta_parametro(graph, i, valori);
            double piu = loss_grafo_test(graph, target, due_loss, i == 4 ? valori : bias_ramo, NULL, 0);

            valori[j] = originale - epsilon;
            if (i < 4) imposta_parametro(graph, i, valori);
            double meno = loss_grafo_test(graph, target, due_loss, i == 4 ? valori : bias_ramo, NULL, 0);

            valori[j] = originale;
            if (i < 4) imposta_parametro(graph, i, valori);
            double numerico = (piu - meno) / (2.0 * epsilon);
            double errore = fabs(gradienti[i][j] - numerico);
            if (!isfinite(gradienti[i][j]) || errore > 1e-6)
                fprintf(stderr, "Nodo N%d, gradiente[%d]: %.12g; numerico %.12g\n",
                        i == 4 ? 12 : i, j, gradienti[i][j], numerico);
            verifica(isfinite(gradienti[i][j]) && errore < 1e-6, "Gradiente del grafo errato.");
            if (errore > massimo_errore) massimo_errore = errore;
            ++verificati;
        }
    }
    libera_grafo(graph);
    af_release_array(target);
    printf("[OK] %d gradienti; errore massimo %.3g.\n", verificati, massimo_errore);
}

static void memoria_in_uso(size_t* bytes, size_t* buffers)
{
    size_t allocati, numero_allocati;
    verifica(af_sync(-1) == AF_SUCCESS, "Sincronizzazione ArrayFire.");
    verifica(af_device_gc() == AF_SUCCESS, "Pulizia cache ArrayFire.");
    verifica(af_device_mem_info(&allocati, &numero_allocati, bytes, buffers) == AF_SUCCESS,
             "Lettura della memoria ArrayFire.");
}

static void test_riuso_grafo(void)
{
    banner("TEST 4 - RIUSO E LIBERAZIONE DEL GRAFO");
    size_t bytes_prima, buffers_prima, bytes_dopo, buffers_dopo;
    memoria_in_uso(&bytes_prima, &buffers_prima);
    array_node* graph = grafo_test();
    af_array target = matrice(TEST_Y, 3, 2);
    af_array pesi[4];
    for (int i = 0; i < 4; ++i) pesi[i] = graph->node[i]->valore;
    double gradienti[5][12];
    for (int ciclo = 0; ciclo < 20; ++ciclo) {
        forward_grafo_test(graph);
        forward_grafo_test(graph);
        loss_grafo_test(graph, target, ciclo % 2, TEST_PARAMETRI[4], gradienti, 0);
        for (int i = 0; i < 5; ++i) {
            node* nodo = graph->node[i];
            verifica(nodo->left == NULL && nodo->right == NULL, "Collegamenti residui nel grafo di test.");
            verifica(nodo->gradiente == NULL && nodo->temp == NULL, "Gradienti residui nel grafo di test.");
            if (i < 4) verifica(nodo->valore == pesi[i], "Il reset ha sostituito un parametro.");
        }
    }
    libera_grafo(graph);
    af_release_array(target);
    memoria_in_uso(&bytes_dopo, &buffers_dopo);
    verifica(bytes_prima == bytes_dopo && buffers_prima == buffers_dopo, "Memoria residua del grafo.");
    puts("[OK] 20 cicli; parametri conservati, collegamenti puliti e memoria rilasciata.");
}

/*
################################################################
# MNIST: FILE IDX E MINI-BATCH                                  #
################################################################
*/

static uint32_t leggi_intero_idx(FILE* file)
{
    unsigned char b[4];
    verifica(fread(b, 1, 4, file) == 4, "Intestazione IDX incompleta.");
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | b[3];
}

static dati_mnist carica_mnist(const char* prefisso, int numero)
{
    char percorso[128];
    snprintf(percorso, sizeof percorso, "data/mnist/%s-images-idx3-ubyte", prefisso);
    FILE* immagini = fopen(percorso, "rb");
    snprintf(percorso, sizeof percorso, "data/mnist/%s-labels-idx1-ubyte", prefisso);
    FILE* etichette = fopen(percorso, "rb");
    verifica(immagini != NULL && etichette != NULL,
             "File MNIST assenti: avvia il programma dalla cartella del progetto.");
    verifica(leggi_intero_idx(immagini) == 2051 && leggi_intero_idx(immagini) == (uint32_t)numero &&
             leggi_intero_idx(immagini) == 28 && leggi_intero_idx(immagini) == 28,
             "Formato delle immagini MNIST.");
    verifica(leggi_intero_idx(etichette) == 2049 && leggi_intero_idx(etichette) == (uint32_t)numero,
             "Formato delle etichette MNIST.");

    dati_mnist dati = {.numero = numero};
    size_t pixel = (size_t)numero * PIXEL;
    dati.immagini = malloc(pixel);
    dati.etichette = malloc((size_t)numero);
    verifica(dati.immagini != NULL && dati.etichette != NULL, "Allocazione del dataset.");
    verifica(fread(dati.immagini, 1, pixel, immagini) == pixel &&
             fread(dati.etichette, 1, (size_t)numero, etichette) == (size_t)numero,
             "Dataset MNIST incompleto.");
    fclose(immagini);
    fclose(etichette);
    for (int i = 0; i < numero; ++i)
        verifica(dati.etichette[i] < CLASSI, "Etichetta MNIST fuori dall'intervallo 0..9.");
    printf("MNIST %s: %d immagini 28 x 28.\n", prefisso, numero);
    return dati;
}

static void prepara_batch(const dati_mnist* dati, const int* ordine, int inizio, int numero,
                          af_array* input, af_array* target)
{
    double* x = malloc((size_t)numero * PIXEL * sizeof *x);
    double* y = calloc((size_t)numero * CLASSI, sizeof *y);
    verifica(x != NULL && y != NULL, "Allocazione del batch.");
    for (int riga = 0; riga < numero; ++riga) {
        int campione = ordine != NULL ? ordine[inizio + riga] : inizio + riga;
        for (int pixel = 0; pixel < PIXEL; ++pixel)
            x[riga + (size_t)pixel * numero] = dati->immagini[(size_t)campione * PIXEL + pixel] / 255.0;
        y[riga + (size_t)dati->etichette[campione] * numero] = 1.0;
    }
    *input = matrice(x, numero, PIXEL);
    *target = matrice(y, numero, CLASSI);
    free(x);
    free(y);
}

/*
################################################################
# MNIST: TRAINING E VALUTAZIONE CON TRAIN E FORWARD              #
################################################################
*/

static void imposta_input_mnist(array_node* graph, af_array input)
{
    // train e forward leggono l'input dal nodo del modello.
    node* nodo_input = graph->node[graph->length - 1];
    if (nodo_input->valore != NULL) af_release_array(nodo_input->valore);
    af_retain_array(&nodo_input->valore, input);
}

static double valuta_mnist(array_node* graph, const dati_mnist* dati, int mostra_esempi)
{
    int corretti = 0;
    for (int inizio = 0; inizio < dati->numero; inizio += BATCH) {
        int numero = dati->numero - inizio;
        if (numero > BATCH) numero = BATCH;
        af_array input = NULL, target = NULL;
        prepara_batch(dati, NULL, inizio, numero, &input, &target);
        imposta_input_mnist(graph, input);
        node* output = forward(graph, RELU);
        verifica(output != NULL, "Output del forward assente.");
        verifica_forma(output->valore, numero, CLASSI);
        double probabilita[BATCH * CLASSI];
        verifica(af_get_data_ptr(probabilita, output->valore) == AF_SUCCESS, "Lettura delle predizioni.");
        for (int riga = 0; riga < numero; ++riga) {
            int cifra = 0;
            double somma = 0.0;
            for (int classe = 0; classe < CLASSI; ++classe) {
                double p = probabilita[riga + classe * numero];
                verifica(isfinite(p) && p >= 0.0 && p <= 1.0, "Probabilita non valida.");
                somma += p;
                if (p > probabilita[riga + cifra * numero]) cifra = classe;
            }
            verifica(fabs(somma - 1.0) < 1e-9, "Le probabilita non sommano a uno.");
            int atteso = dati->etichette[inizio + riga];
            if (cifra == atteso) ++corretti;
            if (mostra_esempi && inizio == 0 && riga < 10)
                printf("Immagine %d: cifra vera %d, predetta %d.\n", riga, atteso, cifra);
        }
        // L'output appartiene al modello; forward lo libera alla chiamata successiva.
        af_release_array(input);
        af_release_array(target);
    }
    return 100.0 * corretti / dati->numero;
}

static void allena_mnist(void)
{
    banner("MNIST - TRAINING");
    dati_mnist training = carica_mnist("train", 60000);
    dati_mnist test = carica_mnist("t10k", 10000);
    int* ordine = malloc((size_t)training.numero * sizeof *ordine);
    verifica(ordine != NULL, "Allocazione degli indici di training.");
    for (int i = 0; i < training.numero; ++i) ordine[i] = i;
    srand(42);
    verifica(af_set_seed(42) == AF_SUCCESS, "Seed ArrayFire.");
    int dimensioni[] = {PIXEL, NASCOSTI, CLASSI};
    // init_graph legge anche layers[-1], cioe la dimensione dell'input.
    array_node* graph = init_graph(&dimensioni[1], 2, NULL);
    verifica(graph != NULL && graph->length == 5, "Modello MNIST non valido.");
    verifica_forma(graph->node[0]->valore, PIXEL, NASCOSTI);
    verifica_forma(graph->node[1]->valore, 1, NASCOSTI);
    verifica_forma(graph->node[2]->valore, NASCOSTI, CLASSI);
    verifica_forma(graph->node[3]->valore, 1, CLASSI);
    printf("Rete %d -> %d -> %d; batch %d; epoche %d.\n", PIXEL, NASCOSTI, CLASSI, BATCH, EPOCHE);
    puts("Input: una riga per immagine, pixel / 255; target one-hot.");
    printf("Accuratezza iniziale sul test set: %.2f%%.\n", valuta_mnist(graph, &test, 0));

    // train aggiorna sempre i pesi dello stesso modello, anche tra epoche diverse.
    for (int epoca = 0; epoca < EPOCHE; ++epoca) {
        clock_t inizio_epoca = clock();
        for (int i = training.numero - 1; i > 0; --i) {
            int j = (int)((rand() / (RAND_MAX + 1.0)) * (i + 1));
            int tmp = ordine[i]; ordine[i] = ordine[j]; ordine[j] = tmp;
        }
        double totale_loss = 0.0;
        for (int inizio = 0; inizio < training.numero; inizio += BATCH) {
            int numero = training.numero - inizio;
            if (numero > BATCH) numero = BATCH;
            af_array input = NULL, target = NULL;
            prepara_batch(&training, ordine, inizio, numero, &input, &target);
            imposta_input_mnist(graph, input);
            train(CROSS_ENTROPY, target, graph, RELU);
            if (epoca == 0 && inizio == 0) {
                puts("\nStruttura del modello MNIST al primo batch di training:");
                stampa_modello(3, graph, NULL, 0);
            }

            // train non ha ancora un return: la loss resta accessibile nel grafo.
            node* trovate[1];
            array_node foglie = {.node = trovate, .size = 1, .length = 0};
            find_leaf(graph, &foglie);
            verifica(foglie.length == 1 && foglie.node[0]->operazione == OP_CROSS_ENTROPY,
                     "Il training deve produrre una loss.");
            double loss = scalare(foglie.node[0]->valore);
            totale_loss += loss * numero;
            af_release_array(input);
            af_release_array(target);
            if ((inizio / BATCH + 1) % 100 == 0 || inizio + numero == training.numero)
                printf("Epoca %d: %d/%d immagini; loss batch %.6f.\n",
                       epoca + 1, inizio + numero, training.numero, loss);
        }
        double accuratezza = valuta_mnist(graph, &test, epoca == EPOCHE - 1);
        printf("Epoca %d/%d completata: loss media %.6f; accuratezza test %.2f%%; %.1f s.\n",
               epoca + 1, EPOCHE, totale_loss / training.numero, accuratezza,
               (double)(clock() - inizio_epoca) / CLOCKS_PER_SEC);
    }

    // Il rilascio della libreria e ancora privato: liberiamo i nodi posseduti dal main.
    libera_grafo(graph);
    free(ordine);
    free(training.immagini);
    free(training.etichette);
    free(test.immagini);
    free(test.etichette);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    test_struttura_forward();
    test_gradienti(0);
    test_gradienti(1);
    test_riuso_grafo();
    puts("\nTutti i test del grafo sono superati.");
    allena_mnist();
    return EXIT_SUCCESS;
}
