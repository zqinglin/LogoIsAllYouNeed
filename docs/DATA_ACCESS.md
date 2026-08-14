# Data Access and Placement

This package does **not** include the large generated video corpus (`Videos/GenVideos/my_videos/*`) to keep supplementary size manageable.

## Included Metadata

Under `data_metadata/`:

- `data_download.py`
- `data_get.py`
- `generate_alpha_gradients.sh`
- `prompt_video_mapping.csv`
- `video_to_prompt_full.csv`

These files are enough to reconstruct mapping and download workflow.

## Recommended Download Source

The helper scripts point to:

- Hugging Face dataset: `Rapidata/text-2-video-human-preferences`

## Suggested Local Placement

After downloading, place videos under:

- `data/videos/` (or another fixed path you use in scripts)

Then update script paths consistently in:

- `src/watermark_tools/*`
- `src/evaluation/wrappers/*`
- `src/evaluation/LargeScaleEval/*`

## Notes

- Keep original filenames (for prompt/video mapping consistency).
- Do not rename model-specific prefixes/suffixes in video filenames.
- If required by your artifact checklist, provide a short `SHA256` manifest for downloaded files.
