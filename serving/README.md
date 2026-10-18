# Temporary serving corrections

All workaround implementations live in
[kludge.py](../src/mam_bench/kludge.py), including the OpenAI fixed-array schema
adapter and the Qwen, Gemma and Bolt serving plugins. Serving plugins run in
the pinned PGX vLLM image (Python 3.12), separately from MAM-Bench's Python 3.14
package. Their standalone source is stored in `SERVING_PLUGINS` in that module
and exported without importing vLLM or installing server packages locally:

```sh
PYTHONPATH=src uv run python -m mam_bench.kludge --export-serving .scratch/serving-plugins
```

This produces `qwen_empty_tool_kludge.py`, `gemma4_tuplefix_plugin.py` and
`ultra_v3_reasoning_parser.py`. Edit only `kludge.py` and regenerate; exported
files are deployment artifacts. Exporting does not deploy or restart a server.
Keep existing frozen campaign copies and their hashes as historical evidence.

`qwen_empty_tool_kludge.py` is loaded with vLLM's `--tool-parser-plugin` and
selected as `--tool-call-parser qwen3_coder_empty_tool_kludge`. It inherits the
native Qwen parser and only replaces the content of a closed, empty Qwen XML
argument section with an empty string. Tools with parameters, open schemas,
other XML styles, and non-strict tools keep their existing grammar. Sampling,
tool schemas, tool extraction, simulation rules, and timeouts are unchanged.

The installed XGrammar 0.2.8 otherwise permits unbounded whitespace there:
[vLLM #50989](https://github.com/vllm-project/vllm/issues/50989),
[XGrammar #802](https://github.com/mlc-ai/xgrammar/issues/802).
Remove this plugin when the serving image passes the matcher regression
without the correction; [XGrammar #837](https://github.com/mlc-ai/xgrammar/pull/837)
tracks the upstream empty-argument correction.

The regression test runs in the serving image, without loading a model or
performing GPU computation. Mount the exported plugin and
`serving/test_qwen_empty_tool_kludge.py` under `/grammar-kludge`:

```sh
UV_CACHE_DIR=/runtime-cache/uv PYTHONPATH=/grammar-kludge \
  uv run --offline --no-project --python /usr/bin/python3 \
  /usr/bin/python3 /grammar-kludge/test_qwen_empty_tool_kludge.py -v
```

Campaign-specific image/model hashes, the original and corrected launch
scripts, red/green matcher output, and bounded streamed probe results belong
under `.scratch/model-runs/`; preserve interrupted benchmark artifacts there.

## Qwen rollout on PGX

Use this correction for every Qwen model in the current roster: Qwen3.5 0.8B,
2B, 4B and 9B, Qwen3.6-35B-A3B MoE, and Qwen3.8-27B, in both thinking modes.
The correction applies to the empty tool grammar regardless of thinking mode.
The original grammar reproduced whitespace loops on 0.8B, 2B, 4B and 27B;
9B did not reproduce it in the bounded 20-call sample.

`pgx/` holds the four dense vLLM configurations and their existing Pitchfork
launch definitions. The two NVFP4 SparkRun recipes remain in
`/home/joehe/git/jh-sparkruns/`. Both launch paths mount the shared PGX copy at
`/home/joehe/git/jh-sparkruns/grammar-kludge` read-only as `/grammar-kludge`.
That directory contains exported copies of the plugin and the matcher test
above. Generate plugins from `kludge.py` before an approved deployment; the
matcher test remains maintained here. The 27B diagnostic launcher uses the
same shared copy.

The SparkRun recipes intentionally use a PGX-specific absolute bind mount.
Their validator warns about portability; the deployed source path must exist
before launch. The MoE's old `qwen3_xml` and the dense models' `qwen3_coder`
resolve to the same native parser class in the pinned serving image.
The MoE SparkRun recipe now pins that same cached image instead of the absent
local `vllm-node` tag, so its plugin API matches the other Qwen deployments.

Keep models sequential on PGX. Applying these recipes does not resume stopped
benchmark campaigns or change their archived serving provenance.

## Gemma and Bolt

The Gemma export keeps the `gemma4_tuplefix` tool parser registration and its
strict, schema-scoped nullable integer coordinate repair. The Bolt export keeps
the `ultra_v3` reasoning parser registration and its explicit thinking-off
handling. Existing recipe filenames and parser names remain valid. Causes,
evidence and removal conditions are recorded beside their source in `kludge.py`.
