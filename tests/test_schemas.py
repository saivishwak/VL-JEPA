import pytest

from vl_jepa.data.schemas import VLJEPASample


def test_vl_jepa_sample_requires_visual_path_for_images():
    with pytest.raises(ValueError):
        VLJEPASample.model_validate(
            {
                "id": "missing-image",
                "task_type": "captioning",
                "visual_kind": "image",
                "query": "Caption the image.",
                "target": "A scene.",
            }
        )


def test_vl_jepa_sample_accepts_paper_triplet():
    sample = VLJEPASample.model_validate(
        {
            "id": "sample-1",
            "task_type": "captioning",
            "visual_kind": "image",
            "visual_path": "image.jpg",
            "query": "",
            "target": "A person is cooking.",
            "source_dataset": "datacomp",
            "split": "train",
        }
    )
    assert sample.target == "A person is cooking."
