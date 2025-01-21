from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B-Preview")
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-70B")


symbols = ["◖", "◗", "◢", "◣", "◒", "◓","→","←","»","«"]
for symbol in symbols:
    tokens = tokenizer(symbol, return_tensors="pt")
    print(f"Symbol: {symbol}, Token IDs: {tokens.input_ids}")


exit(0)

for i in range(130000):


    tok = tokenizer.decode([i])

    if len(tok) > 0:
        if tok != '�':
            print(f"token({i}): '{tok}'")


# token(8674): '→'
# token(57258): '←'

# token(3807): '»'
# token(12389): '«'

# byte_sequence = b'\xe2\x85'
# byte_sequence = b'\xe2\x85\xa2'
# decoded_character = byte_sequence.decode('utf-8')
# print(decoded_character)


# decoded_text = tokenizer.decode([146634])
# print(f"Token ID 146634 decodes to: {decoded_text}")
