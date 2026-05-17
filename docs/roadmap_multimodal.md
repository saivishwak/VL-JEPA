# Multimodal Roadmap

V1 is text-generation only. The project keeps the future image/V-JEPA path explicit without
mixing unfinished multimodal code into the current trainer.

## Planned V2 Direction

- Add `image_text` and `video_text` dataset schemas alongside the current chat schema.
- Add visual encoders and projection heads under `llm_jepa/modeling/vision/`.
- Add V-JEPA-style latent predictors that consume visual embeddings and predict target latents.
- Extend collators to handle pixel tensors, patch/tube masks, and text tokens together.
- Add image-conditioned generation inference with a model backend that supports multimodal inputs.
- Add retrieval, captioning, VQA, and image-text alignment evaluation.

## Boundary Preserved in V1

The current trainer works over named views and representation losses. That means future visual
views can be introduced without changing the public CLI contract for preparing data, training,
inference, or evaluation.
