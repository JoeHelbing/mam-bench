# Models under evaluation

Continuing roster selected on October 15, 2026. The models below remain in scope;
earlier Nemotron Nano, GPT-OSS, and Muse Spark configurations are historical.
Serving validation and completed benchmark scores are separate milestones.

| Model | Selected checkpoint | Modes |
| --- | --- | --- |
| Qwen3.5-0.8B | Zerul/Qwen3.5-0.8B-W4A16-AutoRound-AWQ | Thinking off / on |
| Qwen3.5-2B | QuantTrio/Qwen3.5-2B-AWQ | Thinking off / on |
| Qwen3.5-4B | QuantTrio/Qwen3.5-4B-AWQ | Thinking off / on |
| Qwen3.5-9B | QuantTrio/Qwen3.5-9B-AWQ | Thinking off / on |
| Qwen3.6-35B-A3B MoE | nvidia/Qwen3.6-35B-A3B-NVFP4 | Thinking off / on |
| Qwen3.8-27B | nvidia/Qwen3.8-27B-NVFP4 | Thinking off / on |
| Gemma-4-26B-A4B | nvidia/Gemma-4-26B-A4B-NVFP4 | Thinking off / on |
| Nemotron-3.5 Bolt 74B-A9B | nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB | Thinking off / on |

Bolt passed its independent PGX serving trial on October 15, 2026, using
NVIDIA's pinned vLLM nightly, mixed IQ2 weights, b12x kernels, FP8 attention KV,
prefix caching, and seven-token LiLiCorr/DFlash. Thinking off/on, parsed tools,
native MAM-Bench sessions, cache reuse, and repetition protection passed.

Both one- and sixteen-sequence configurations completed a real 262,144-token
request in about four minutes. Sixteen clients each completed a 16K prompt;
samples observed a peak of seven active requests. These are separate context
and queuing checks, not sixteen simultaneous full-context requests. Available
memory stayed above 38 GiB during the probes, and swap use did not increase.

All workaround implementations are collected in [kludge.py](../src/mam_bench/kludge.py),
including the Qwen3.8-27B thinking fix, OpenAI fixed-array encoding, and exported
Qwen, Gemma and Bolt serving plugins. All six Qwen serving recipes use the
[empty-tool grammar correction](../serving/README.md), in both thinking modes.
The Bolt serving recipe also requires
the documented NVIDIA parser correction. Exact recipes, revisions and receipts
are retained locally under `.scratch/model-runs/pgx-nemotron35-bolt-20261015/`.
Serving is stopped after validation; existing campaigns remain stopped. These
checks do not produce a MAM-Bench score.

Official Bolt recipe:
<https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB>
