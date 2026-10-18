"""Run inside the pinned serving image; no model or GPU computation needed."""

import unittest

import xgrammar as xgr
from qwen_empty_tool_kludge import correct_empty_tool_arguments
from xgrammar import get_model_structural_tag


class EmptyQwenToolTests(unittest.TestCase):
    def matcher(
        self,
        schema: dict[str, object],
        *,
        strict: bool = True,
        model: str = "qwen_3_coder",
        choice: str | dict[str, object] = "required",
    ) -> xgr.GrammarMatcher:
        tag = get_model_structural_tag(
            model,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "probe",
                        "strict": strict,
                        "parameters": schema,
                    },
                }
            ],
            tool_choice=choice,
            reasoning=False,
        )
        corrected = correct_empty_tool_arguments(tag)
        vocab = [bytes([b]).decode("latin-1") for b in range(256)]
        compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(vocab, vocab_type=xgr.VocabType.RAW))
        return xgr.GrammarMatcher(compiler.compile_structural_tag(corrected))

    def test_empty_closed_tool_must_close_without_whitespace_loop(self) -> None:
        schema = {"type": "object", "properties": {}, "additionalProperties": False}
        prefix = "<tool_call>\n<function=probe>\n"
        for choice in ("required", {"type": "function", "function": {"name": "probe"}}):
            for whitespace in (" ", "\t", "\r", "\n\n"):
                with self.subTest(choice=choice, whitespace=repr(whitespace)):
                    matcher = self.matcher(schema, choice=choice)
                    self.assertTrue(matcher.accept_string(prefix))
                    self.assertFalse(matcher.accept_string(whitespace))
            matcher = self.matcher(schema, choice=choice)
            self.assertTrue(matcher.accept_string(prefix + "\n</function>\n</tool_call>"))

    def test_parameter_schema_still_accepts_valid_and_rejects_invalid_arguments(self) -> None:
        schema = {
            "type": "object",
            "properties": {"total": {"type": "integer"}},
            "required": ["total"],
            "additionalProperties": False,
        }
        prefix = "<tool_call>\n<function=probe>\n<parameter=total>"
        matcher = self.matcher(schema)
        self.assertTrue(matcher.accept_string(prefix + "12</parameter>\n</function>\n</tool_call>"))
        matcher = self.matcher(schema)
        self.assertTrue(matcher.accept_string(prefix))
        self.assertFalse(matcher.accept_string("wrong"))

    def test_non_strict_open_tools_are_not_narrowed(self) -> None:
        schema = {"type": "object", "properties": {}, "additionalProperties": False}
        matcher = self.matcher(schema, strict=False)
        self.assertTrue(matcher.accept_string("<tool_call>\n<function=probe>\n" + " " * 256))


if __name__ == "__main__":
    unittest.main()
