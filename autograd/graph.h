#ifndef GRAPH_HEADER
    #define GRAPH_HEADER 0
    #include "../general.h"
    #include "../activation_functions/activation_function.h"
    #include "../loss/loss.h"

    typedef enum op{
        OP_SUM,
        OP_MATMUL,
        OP_RELU,
        OP_SOFTMAX,
        OP_CROSS_ENTROPY,
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
        node* father_left;
        node* father_right;
    };

    typedef struct array_node{
        node** node;
        int size;
        int length;
    } array_node;


    node* alloc_node(node_type);

    void init_node(node* nodo, af_array valore,
                   af_array gradiente, op operazione,
                   node_type tipo, node* left, node* right,
                   node* father_left, node* father_right
                );

    node* add_op(node*,node*,af_array,op);
    void add_child(node*,node*);
    node* create_val(af_array);
    node* add_matmul(node*,node*,op);
    node* add_sum(node*,node*,op);
    node* add_activation(node*,float (*fun)(float),op);
    node* add_softmax(node*,op);
    af_array matrix_map(af_array,float (*fun)(float));
    node* add_loss(node*,af_array,af_array (*fun)(af_array,af_array),op);
    int is_node_in_array(array_node*, node*);
    
    


#endif
