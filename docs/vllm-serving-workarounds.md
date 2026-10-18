# vLLM serving workarounds

These corrections address grammar or parser defects observed while using
MAM-Bench with vLLM. Apply them in the server environment if its installed
versions reproduce the defect. MAM-Bench does not install server plugins or
manage model deployment; its client-side adapters live in
[kludge.py](../src/mam_bench/kludge.py).

The descriptions below explain what to change and how to check it. Parser
classes and plugin APIs vary between vLLM versions, so inspect the source in
your serving image before adapting a subclass. Keep the original request,
generated output, parsed output and version information together when testing.

## Qwen: whitespace loops in strict tools with no arguments

**Symptom.** A strict, parameterless tool call opens its XML function tag and
then generates spaces or tabs instead of closing it. The server can count
thousands of output tokens while returning almost no parsed text or arguments.
This can occur with thinking disabled; whitespace is not evidence of reasoning.

**Cause.** The affected XGrammar XML converter represents an empty parameter
section as `[ \n\t]*`. This permits unlimited whitespace without advancing to
the function's closing tag. The defect was reported with XGrammar 0.2.3 and
reproduced locally with 0.2.8. See
[vLLM #50989](https://github.com/vllm-project/vllm/issues/50989) and
[XGrammar #802](https://github.com/mlc-ai/xgrammar/issues/802).

**Correction.** Prefer a serving build containing the upstream empty-XML
grammar correction. [XGrammar PR #837](https://github.com/mlc-ai/xgrammar/pull/837)
proposes that correction; it was still open on October 18, 2026. Check the
version actually installed in your server rather than assuming a vLLM upgrade
includes it.

If maintaining a temporary vLLM tool-parser plugin instead:

1. Subclass the Qwen tool parser selected by your server. The tested API used
   `Qwen3EngineToolParser.get_structural_tag()`.
2. Obtain the native structural tag, copy it, and visit its nested tag formats.
3. For a `TagFormat` whose content is `JSONSchemaFormat(style="qwen_xml")`,
   replace the content with `ConstStringFormat(value="")` only when its schema
   describes a closed object with no properties and no required arguments.
   Do not narrow open objects, parameterized tools, other formats or unfamiliar
   schema constraints.
4. Leave the native tag boundaries, tool extraction, sampling and schemas intact.
   The empty content lets the existing closing tag take over immediately.
5. Register the subclass with `ToolParserManager` under a distinct name. Load
   your file with `--tool-parser-plugin` and select that name with
   `--tool-call-parser` in your existing server command.

For example, append these options to your own `vllm serve` command after
implementing the plugin. The path and registration name are placeholders:

```text
--tool-parser-plugin /path/to/qwen_empty_arguments.py
--tool-call-parser qwen_empty_arguments
```

**Verify before a benchmark.** Compile the structural tag with XGrammar and
use `GrammarMatcher` to check that an empty function can close, repeated spaces
or tabs cannot keep it open, and a parameterized integer tool still accepts
integers and rejects invalid values. Open/non-strict schemas must retain their
native behavior. These checks need no model or GPU. Then make bounded streamed
requests for both a parameterless tool and a parameterized control, in both
thinking modes. Compare generated token IDs with parsed tool calls.

**Remove when** the unmodified grammar passes those checks in your serving
build. Changing tool schemas, turning strict mode off or lowering timeouts
does not correct the grammar defect.

## Gemma: integer coordinate arrays parsed as strings

**Symptom.** A model emits native integer coordinates such as `[2, 3]`, but the
tool parser returns `["2", "3"]`. MAM-Bench's strict coordinate validation then
rejects the tool call. A parser can also emit an incomplete JSON prefix before
it has enough information to repair the types.

**Cause.** The affected Gemma parser does not preserve integer elements for
the nullable fixed-length tuple schema used by movement tools. See the observed
defect and validation limits in [research issue R04](research-issue-log.md#r04---nullable-integer-tuples-were-parsed-as-string-arrays).
This is a serving conversion problem only when the raw model output contained
integers; model-generated quoted strings remain invalid.

**Correction.** Adapt the native Gemma tool parser in your vLLM version. The
tested implementation subclassed `Gemma4Parser` and selected it through a
`Gemma4EngineToolParser` subclass's `_parser_engine_cls`.

- Scope the conversion to the exact strict tool schemas you use: a nullable
  array of exactly two integers, with the expected required fields and no
  additional properties. For MAM-Bench these are the coordinate fields on
  `move` and `act`.
- Retain the raw native argument span before the native argument converter
  runs. Split top-level fields with awareness of Gemma string delimiters and
  array nesting; a global numeric-string replacement changes legitimate text.
- After native conversion, repair a coordinate only if the raw field contains
  two unquoted JSON integer literals and the parsed values match those same
  literals as strings. Preserve `null` and verify other fields, including
  booleans, before applying the repair. Quoted strings, extra/missing fields
  and mismatched schemas must retain native validation behavior.
- For those affected tools, buffer streaming argument deltas until conversion
  is final. Already emitted JSON prefixes cannot be retracted to change their
  types. Leave unrelated tools on the native streaming path.

Register the tool parser under a distinct name and select it using your own
`--tool-parser-plugin` file and `--tool-call-parser` name, as above.

**Verify** both streamed and non-streamed calls with integer coordinates,
`null`, negative coordinates and multiple coordinate fields. Quoted numeric
strings must remain strings and fail strict integer validation; unrelated
tools and unknown schemas must remain unchanged. Compare the raw argument
span, parsed JSON and application validation result.

**Remove when** your native parser preserves these types and passes the same
streaming checks. Coercing every numeric-looking string in the client would
hide invalid model output.

## Nemotron Bolt: thinking-off output classified as reasoning

**Symptom.** A thinking-off request produces an answer or tool markup, but the
reasoning parser places it in the reasoning channel and leaves content empty.

**Cause.** In the affected serving path, vLLM resolves
`reasoning_effort="none"` into chat-template rendering arguments without
updating the request object seen by NVIDIA's `ultra_v3` parser. The template
prefills `<think></think>`, so generated output has no closing thinking marker
for a parser that assumes reasoning comes first.

**Correction.** In the reasoning parser, resolve the thinking-off condition
using the same precedence as the renderer:

1. An explicit `chat_template_kwargs.enable_thinking` takes precedence.
2. If that key is absent, `reasoning_effort="none"` disables thinking.
3. When thinking is disabled, return no reasoning and put the generated span
   in content (`None, model_output or None`). Otherwise retain the native
   reasoning extraction behavior.

Use a subclass of the reasoning parser supplied with your chosen model/server
recipe. Keep its other behavior intact. Register the correction with
`ReasoningParserManager` under a distinct name, load your plugin with
`--reasoning-parser-plugin`, and select that name with `--reasoning-parser`.
This is a reasoning parser correction, separate from the tool-call parser.

**Verify** effort `none` with no template override, explicit thinking off,
and explicit thinking on overriding effort `none`. Check ordinary text and
tool-call output, including the streaming path actually used by your server.
Thinking-on requests must retain the native reasoning/content split.

**Remove when** the native parser receives the resolved rendering settings or
honors effort `none` with the same precedence. Client-side thinking forwarding
and server-side reasoning extraction are separate checks; fixing one does not
establish that the other works.
