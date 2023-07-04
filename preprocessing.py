"""
"""


from transformers import PreTrainedTokenizer


def get_group_texts_fn(block_size=2**14):
    """
                                   block_size
                              1024 | 4096 | 16384
                            -------------------
                      256  |   6             5
    vocab_size       4096  |
                    65536  |
    """
    
    def fn(examples):
        concatenated = "".join(examples["text"])
        total_length = len(concatenated)
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        chopped = [concatenated[i : i + block_size] for i in range(0, total_length, block_size)]
        examples["text"] = chopped
        return examples
    
    return fn


def get_tokenize_fn(tokenizer: PreTrainedTokenizer, max_length: int, truncation=True):
    
    def fn(examples):
        return tokenizer(examples["text"], max_length=max_length, truncation=truncation)

    return fn
