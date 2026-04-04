from transformers import PreTrainedTokenizer
from typing import Union, cast


class Tokenizer:
    """分词器封装"""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """编码文本为 token IDs"""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """解码 token IDs 为文本"""
        return cast(str, self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens))

    def __call__(self, text: Union[str, list[str]], **kwargs):
        """调用分词器"""
        return self.tokenizer(text, **kwargs)

    @property
    def eos_token_id(self) -> int | None:
        return cast(int | None, self.tokenizer.eos_token_id)

    @property
    def bos_token_id(self) -> int | None:
        return cast(int | None, self.tokenizer.bos_token_id)

    @property
    def pad_token_id(self) -> int | None:
        return cast(int | None, self.tokenizer.pad_token_id)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)