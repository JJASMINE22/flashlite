from dataclasses import dataclass, field

@dataclass
class DataArgs:
    tokenizer_dir: str = field(
        default="./tokenizer"
    ) 
    
    data_dir: str = field(
        default=r"E:\datasets\sft"
    )
    
    cache_dir: str = field(
        default=r"E:\.cache"
    )
    
    packed_data_dir: str = field(
        default=r"E:\packed_dataset"
    )
    
    seed: int = field(
        default=42
    )

    train_ratio: float = field(
        default=0.9
    )

    max_length: int = field(
        default=2048
    )

    max_prompt_length: int = field(
        default=768
    )
