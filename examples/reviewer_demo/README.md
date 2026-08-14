# Reviewer Minimal Demo

This folder contains one paired example to quickly verify score inflation:

- `videos/original/0000_sora_0.mp4`
- `videos/watermarked/0000_sora_0.mp4`

Run:

```bash
bash Code/examples/reviewer_demo/run_demo.sh
```

Output:

- Terminal prints original vs watermarked scores and delta.
- JSON result saved to `Code/examples/reviewer_demo/output/demo_result.json`.
