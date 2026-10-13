# ave results — official

Metrics come only from the supplied completed evaluations.
Single-seed values are not multi-seed averages. Multi-seed values show mean ± sample standard deviation.

| Model | Condition | Accuracy | Macro-F1 | Balanced accuracy | Macro precision | Minority macro-F1 | Minority macro-recall | Seeds |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Reference | audio_video | 0.9002 | 0.8945 | 0.8995 | 0.8984 | 0.8709 | 0.8921 | 17 |
| Reference | audio_only | 0.8554 | 0.8233 | 0.8258 | 0.8338 | 0.6965 | 0.6762 | 17 |
| Reference | video_only | 0.5411 | 0.4804 | 0.5256 | 0.5140 | 0.3621 | 0.4413 | 17 |
| CI-only | audio_video | 0.8953 | 0.8934 | 0.9001 | 0.8995 | 0.8929 | 0.9302 | 17 |
| CI-only | audio_only | 0.8429 | 0.8052 | 0.8119 | 0.8392 | 0.6442 | 0.6218 | 17 |
| CI-only | video_only | 0.4239 | 0.3677 | 0.4226 | 0.4333 | 0.2756 | 0.4313 | 17 |
| Proposed CI + MI | audio_video | 0.8903 | 0.8880 | 0.8960 | 0.8914 | 0.8845 | 0.9365 | 17 |
| Proposed CI + MI | audio_only | 0.8204 | 0.7982 | 0.7996 | 0.8344 | 0.7150 | 0.6841 | 17 |
| Proposed CI + MI | video_only | 0.4663 | 0.4158 | 0.4692 | 0.4667 | 0.2885 | 0.4810 | 17 |
