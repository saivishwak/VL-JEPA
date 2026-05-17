import torch

from vl_jepa.data.collators import VLJEPADataCollator
from vl_jepa.data.video import uniform_indices
from vl_jepa.modeling.tokenizer import SimpleTokenizer


def test_uniform_indices_include_temporal_extent():
    assert uniform_indices(10, 4) == [0, 3, 6, 9]


def test_vl_jepa_collator_shapes():
    tokenizer = SimpleTokenizer()
    collator = VLJEPADataCollator(tokenizer, tokenizer)
    batch = collator(
        [
            {
                "id": "a",
                "task_type": "captioning",
                "query_input_ids": [3, 1],
                "query_attention_mask": [1, 1],
                "target_input_ids": [4, 1],
                "target_attention_mask": [1, 1],
                "target_text": "a",
                "candidates": [],
                "metadata": {},
                "pixel_values": torch.zeros(1, 3, 32, 32),
            },
            {
                "id": "b",
                "task_type": "captioning",
                "query_input_ids": [3, 5, 1],
                "query_attention_mask": [1, 1, 1],
                "target_input_ids": [6, 1],
                "target_attention_mask": [1, 1],
                "target_text": "b",
                "candidates": [],
                "metadata": {},
                "pixel_values": torch.ones(1, 3, 32, 32),
            },
        ]
    )
    assert batch["query_input_ids"].shape == (2, 3)
    assert batch["pixel_values"].shape == (2, 1, 3, 32, 32)


def test_vl_jepa_collator_allows_query_free_pretraining():
    tokenizer = SimpleTokenizer()
    collator = VLJEPADataCollator(tokenizer, tokenizer)

    batch = collator(
        [
            {
                "id": "a",
                "task_type": "captioning",
                "query_input_ids": [],
                "query_attention_mask": [],
                "target_input_ids": [4, 1],
                "target_attention_mask": [1, 1],
                "target_text": "a",
                "candidates": [],
                "metadata": {},
                "pixel_values": torch.zeros(1, 3, 32, 32),
            },
            {
                "id": "b",
                "task_type": "captioning",
                "query_input_ids": [],
                "query_attention_mask": [],
                "target_input_ids": [6, 1],
                "target_attention_mask": [1, 1],
                "target_text": "b",
                "candidates": [],
                "metadata": {},
                "pixel_values": torch.ones(1, 3, 32, 32),
            },
        ]
    )

    assert batch["query_input_ids"].shape == (2, 0)
    assert batch["query_attention_mask"].shape == (2, 0)
