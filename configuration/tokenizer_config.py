from dataclasses import dataclass, field


@dataclass
class TokenizerArgs:
    vocab_size: int = field(default=6400)
    special_tokens: list[str] = field(
        default_factory=lambda: [
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
        ]
    )
    additional_tokens: list[str] = field(
        default_factory=lambda: [
           "<tool_call>", 
           "</tool_call>",
           "<tool_response>",
           "</tool_response>",
           "<think>",
           "</think>"
        ]
    )
    num_special_tokens: int = field(default=36)
    tokenizer_dir: str = field(default="./tokenizer") 
    data_dir: str = field(default="D:\datasets\minimind\pretrained")
    sample_size: int = field(default=100000)   
