#ifndef CONFIG_H
#define CONFIG_H

#define DS_VOCAB_SIZE                102400
#define DS_HIDDEN_DIM                2048
#define DS_FIRST_K_DENSE             1
#define DS_MAX_SEQ_LEN               163840

#define DS_N_LAYERS                  27

#define DS_MOE_HIDDEN_SIZE           1408
#define DS_N_ROUTED_EXPERTS          64
#define DS_N_SHARED_EXPERTS          2
#define DS_N_EXPERTS_PER_TOK         6

#define DS_N_ATTN_HEADS              16
#define DS_KV_LORA_RANK              512
#define DS_QK_NOPE_HEAD_DIM          128
#define DS_QK_ROPE_HEAD_DIM          64
#define DS_MLP_HIDDEN                10944
#define DS_RMS_NORM_EPS              1e-6f

#define DS_MAX_TOKEN_LEN             128 // this was determined by inspecting the tokenizer.json
#define DS_N_MERGE_TARGETS           99757
#define DS_TOKENIZER_FILE_DELIMITTER " "
#define DS_MERGE_PAIR_OFFSET_IDX     243

typedef struct {
    // some global configs
    int vocab_size;
    int hidden_dim;
    int first_k_dense_replace;
    int max_sequence_len;

    int n_layers;

    // moe configs
    int moe_hidden_size;
    int n_routed_experts;
    int n_shared_experts;
    int n_experts_per_token;

    // attention configs
    int n_attn_heads;
    int kv_lora_rank;
    int qk_nope_head_dim;
    int qk_rope_head_dim;

    float rms_norm_eps;

    // mlp configs
    int mlp_hidden;
} DeepseekConfig;

#define QC_GROUP_SIZE 64
#define QC_BITS       4

typedef struct {
    int group_size;
    int bits;
} QuantizationConfig;

#endif