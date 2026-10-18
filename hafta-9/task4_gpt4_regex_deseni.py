# --- GÖREV 4: GPT-4 REGEX DESENİ ---
import regex as re

# GPT-4 (cl100k_base) regex deseni
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
gpt4_regex = re.compile(GPT4_SPLIT_PATTERN)

# Kesme işaretli kelimeleri deneyelim
kelimeler = ["Ankara'dan", "İstanbul'da", "Ahmet'in"]

for k in kelimeler:
    bolunmus = gpt4_regex.findall(k)
    print(f"{k:<15} -> {bolunmus}")