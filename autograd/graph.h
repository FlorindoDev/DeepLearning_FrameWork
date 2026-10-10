#ifndef GRAPH_HEADER
    #define GRAPH_HEADER 0
    #include "../general.h"
    #include "../activation_functions/activation_function.h"

    typedef enum op{
        OP_SUM,
        OP_MATMUL,
        OP_RELU,
        OP_SOFTMAX,
        OP_NULL
    } op;

    typedef enum node_type{
        TYPE_OP,
        TYPE_VAL
    } node_type;

    typedef struct node node;

    struct node{
        af_array valore;
        af_array gradiente;
        op operazione;
        node_type tipo;
        node* left;
        node* right;
    };


    node* alloc_node(node_type);

    void init_node(node* nodo, af_array valore,
                   af_array gradiente, op operazione,
                   node_type tipo, node* left, node* right);

    node* add_op(node*,node*,op);
    void add_child(node*,node*);
    node* create_val(af_array);
    node* add_matmul(node*,node*,op);
    node* add_sum(node*,node*,op);
    node* add_relu(node*,op);
    node* add_softmax(node*,op);
    af_array matrix_map(af_array,float (*fun)(float));
    
    


#endif
