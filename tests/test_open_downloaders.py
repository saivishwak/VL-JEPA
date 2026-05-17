from PIL import Image

from vl_jepa.data.downloaders import OPEN_DATASET_PRESETS, record_to_vl_jepa_sample, resolve_split


def test_caption_record_converts_to_vl_jepa_sample(tmp_path):
    sample = record_to_vl_jepa_sample(
        {"id": "img-1", "image": Image.new("RGB", (8, 8)), "caption": ["a red square"]},
        preset=OPEN_DATASET_PRESETS["flickr30k"],
        output_dir=tmp_path,
        split="train",
        index=0,
    )

    assert sample.task_type == "captioning"
    assert sample.visual_kind == "image"
    assert sample.query == "Caption the image."
    assert sample.target == "a red square"
    assert sample.visual_path is not None


def test_vqa_record_converts_answers_to_candidates(tmp_path):
    sample = record_to_vl_jepa_sample(
        {
            "question_id": 7,
            "image": Image.new("RGB", (8, 8)),
            "question": "What color is the square?",
            "answers": ["red", "crimson"],
        },
        preset=OPEN_DATASET_PRESETS["vqav2"],
        output_dir=tmp_path,
        split="validation",
        index=0,
    )

    assert sample.task_type == "vqa"
    assert sample.query == "What color is the square?"
    assert sample.target == "red"
    assert sample.candidates[0].text == "red"
    assert sample.candidates[0].is_correct


def test_vqa_legacy_split_alias_maps_to_validation():
    assert resolve_split(OPEN_DATASET_PRESETS["ok_vqa"], "val2014") == "validation"
    assert resolve_split(OPEN_DATASET_PRESETS["vqav2"], "val") == "validation"
