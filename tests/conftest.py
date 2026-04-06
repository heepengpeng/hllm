"""
pytest 配置和 fixtures
"""

import pytest
from unittest.mock import Mock, MagicMock
import torch


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer fixture"""
    mock = Mock()
    # encode 被调用时返回 tensor
    mock.encode.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock.decode.return_value = "Mocked response"
    mock.eos_token_id = 100
    mock.pad_token_id = 0
    return mock


@pytest.fixture
def mock_model():
    """Mock model fixture"""
    mock = Mock()
    # 设置 logits shape: [batch, seq_len, vocab_size]
    logits = torch.zeros(1, 6, 1000)
    logits[0, -1, 500] = 10.0  # 高概率 token
    mock.return_value.logits = logits
    return mock


@pytest.fixture
def sample_logits():
    """Sample logits tensor"""
    return torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
