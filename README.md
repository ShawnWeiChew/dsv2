# Deepseek V2 Lite Inference Engine in C

![Smol-Whale](assets/smol-whale.jpg)

## Todos:
- [x] Base operations
    - [x] Scaled Matmul (decode-oriented)
    - [x] YaRN RoPE encoding
    - [x] Softmax
    - [x] RMSNorm
    - [x] Silu
- [ ] Forward pass operations
    - [x] Tokenization
    - [ ] Embedding Generation
    - [x] MLA
    - [x] MoE gate
    - [x] MoE ffn
- [x] State to manage
    - [x] JSON parsing
    - [x] Weight loading
        - [x] Figure out weight layout
    - [x] KV cache

** Disclaimer: While I try to replicate the MLX output as best as I can, MLX's native fp16 operations and my CPU's lack of native fp16 support mean that there is eventually divergence in outputs **

### Some Notes / Mistakes I made
- Tokenizer
    - `tokenizer.json` contains the following:
        - Vocab: this gives a mapping between tokens to the token idx
        - Merge: gives the order in which the tokens should be merged
    - UTF-8 is really wacky
        - When the `ascii()` function returns 120, this really means that you should split it into `0b010100` and `0b100000` (5 + 6)
        - I used the cast as `uint16_t` trick, but that assumes that the bytes are in little endian, and reverses them, giving me my final value of `DS_SPACE_CHARACTER_UTF_IDX`
        - The right way to check this should have been with the `encode("utf-8")`
- YaRN
    - Should always verify whether the sum is up to `head_dim` or `head_dim / 2`
- Quantized MatMul formula
    - MLX defines a different quantized matmul formula. I should not have assumed that there is only one formula :(
- MLA ([source](https://github.com/flashinfer-ai/flashinfer/pull/551))
    - `kv_a_proj` fuses the KV down and RoPE shared key generation, it can be split
    - `kv_b_proj` fuses the K and V up projection generation, so it can also be split
    - There are currently 2 ways of implementing MLA (assume current sequence length is `S`)
        - Use original weights:
            - Ops
                - `in` (2048) @ `kv_a_proj` (576, 2048) -> (kv_lora, k_rope) (576) [common op skipped]
                - `in` (2048) @ `q_proj` (3072, 2048) -> (q_nope, q_rope) (1, 3072)
                - `kv_lora` (S, 512) @ `kv_b_proj` (4096, 512) -> (S, k_nope, v) (S, 2048 * 2)
                - `k_nope, k_rope` (16, S, 192) @ `q_nope, q_rope` (16, 1, 192) -> `attention_scores` (16, S, 1)
                - `attn_scores` (16, S, 1) @ `v.view(S, 16, 128).transpose(16, S, 128)` -> out (16, 1, 128) 
            - Calcs: 
                ```python
                flops = (2048 * 3072) + (S * 512 * 4096) + (16 * S * 192) + (16 * S * 128) = 6_291_456 + 2_102_272 * S
                memops = (2048 + 3072 * 2048 + 3072) + (S * 512 + 4096 * 512 + S * 4096) + (16 * S * 192 + 16 * 192 + 16 * S) + (16 * S + 16 * S * 128 + 16 * 128) = 8_398_848 + 9_760 * S
                ```
        - Using matmul absorbtion trick for WukWq (requries splitting of weights): (Suspect my calculations here could be wrong)
            - Ops
                - `in` (2048) @ `kv_a_proj` (576, 2048) -> (kv_lora, k_rope) (576) [common op skipped]
                - `q_proj.view(16, 192, 2048)[:, :128, :]` (16, 128, 2048) @ `kv_b_proj[:2048, :]` (16, 128, 512) -> WukWq (16, 2048, 512) 
                - `in` (2048) @ `WukWq` (16, 2048, 512)  -> (q_nope) (16, 512) 
                - `in` (2048) @ `q_proj[2048:, :]` (1024, 2048) -> (q_rope) (1024)
                - `kv_lora, kv_rope` (S, 576) @ `q_rope, q_nope` (16, 1, 576) -> `attention_scores` (16, S, 1)
                - `kv_lora` (S, 512) @ `kv_b_proj[2048:, :]` (2048, 512) -> `v` (S, 2048)
                - `attn_scores` (16, S, 1) @ `v` (16, S, 128) -> out (16, 1, 128)
            - Calcs
                ```python
                WqWuk pre-computation flops = 16 * 128 * 2048 * 512 = 2147483648
                WqWuk pre-computation memops = 16 * 128 * 2048 + 16 * 128 * 512 + 2048 + 16 * 2048 * 512 (but can be ammortized)

                flops = (2048 * 16 * 512) + (2048 * 1024) + (16 * S * 576) + (S * 512 * 2048) + (16 * S * 128) = 18_874_368 + 1_059_840 * S
                memops = (2048 + 16 * 2048 * 512 + 16 * S * 512) + (2048 + 1024 * 2048 + 1024) + (S * 576 + 16 * 576 + 16 * S) + (S * 512 + 2048 * 512 + S * 2048) + (16 * S + 16 * S * 128 + 16 * 128) = 19_939_328 + 20_016 * S
                ``` 
        - Using matmul absorbtion trick for WukWq (requries splitting of weights), but without materializing `WukWq`:
            - Ops
                - `in` (2048) @ `kv_a_proj` (576, 2048) -> (kv_lora, k_rope) (576) [common op skipped]
                - `in` (2048) @ `q_proj` (3072, 2048) -> (q_nope, q_rope) (3072)
                - `q[:2048]` (16, 128) @ `kv_b_proj[:2048, :]` (16, 128, 512)  -> (q_nope) (16, 512) 
                - `kv_lora, kv_rope` (S, 576) @ `q_rope, q_nope` (16, 1, 576) -> (attn scores) (16, S)
                - `kv_lora` (S, 512) @ `kv_b_proj[2048:, :]` (2048, 512) -> v (S, 2048)
                - `attn_scores` (16, S) @ `v` (16, S, 128) -> out (16, 128)
            - Calcs
                ```python
                flops = 2048 * 3072 + 16 * 128 * 512 + 2048 * 512 = 7_340_032 + 1_059_840 * S
                memops = (2048 + 3072 * 2048 + 3072) + (16 * 128 + 16 * 128 * 512 + 16 * 512) + (S * 576 + 16 * 576 + 16 * S) + (S * 512 + 2048 * 512 + S * 2048) + (16 * S + 16 * S * 128 + 16 * 128)  = 8_415_232 + 3184 * S
                ``` 
    - Implementation notes
        - When doing the matmul absorbtion trick, the WukWq matrix is generated per head
        - `k_rope` gets shared and broadcasted for all styles of computation
        - During matmul absorbtion, `kv_lora` also gets broadcasted
    - The composition of the KV projection is such that each head has the k and v values interleaved. Based on the implementation code, here is a quick test to show what that looks like:
        ```python
        import mlx.core as mx
        import numpy as np

        mx.random.seed(42)

        # 2 sequence
        # 4 heads, each head has a dim of 4 (2 key, 2 value)
        # (S, HEAD, HIDDEN)
        seq = mx.arange(16 * 2).reshape(2, 4, 4)
        print(np.array(seq))
        print("\n\n")

        # within each sequence, the 16 activations are split up
        # (HEAD, S, HIDDEN)
        seq = seq.transpose(1, 0, 2)

        print(np.array(seq))
        print("\n\n")

        a, b = mx.split(seq, [2], axis=-1)

        # if it is what i am doing, then it should be
        # a: 0..15
        # b: 16..31

        print(np.array(a.reshape(-1)))
        print(np.array(b.reshape(-1)))
        ```
