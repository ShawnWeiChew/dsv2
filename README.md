# Deepseek V2 Lite Inference Engine in C

![Smol-Whale](assets/smol-whale.jpg)

## Todos:
- [ ] Base operations
    - [ ] Scaled Matmul (decode-oriented)
    - [ ] YaRN RoPE encoding
    - [x] Softmax
    - [x] RMSNorm
    - [x] Silu
- [ ] Forward pass operations
    - [x] Tokenization
    - [ ] Embedding Generation
    - [ ] MLA
    - [ ] MoE gate
    - [ ] MoE ffn
- [ ] State to manage
    - [x] JSON parsing
    - [ ] Weight loading
        - [ ] Figure out weight layout
    - [ ] KV cache

### Some Notes
- Tokenizer
    - `tokenizer.json` contains the following:
        - Vocab: this gives a mapping between tokens to the token idx
        - Merge: gives the order in which the tokens should be merged
    - UTF-8 is really wacky
        - When the `ascii()` function returns 120, this really means that you should split it into `0b010100` and `0b100000` (5 + 6)
        - I used the cast as `uint16_t` trick, but that assumes that the bytes are in little endian, and reverses them, giving me my final value of `DS_SPACE_CHARACTER_UTF_IDX`
        - The right way to check this should have been with the `encode("utf-8")`