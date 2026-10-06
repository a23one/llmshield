"""Test utility functions and helper methods.

Description:
    Comprehensive tests for all utility functions including conversation
    hashing, entity wrapping, text processing, input validation, protocol
    definitions, and ask_helper functionality.

Author:
    LLMShield by brainpolo, 2025-2026
"""

import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

from parameterized import parameterized

from llmshield import LLMShield
from llmshield.entity_detector import EntityType
from llmshield.utils import (
    PydanticLike,
    _should_cloak_input,
    ask_helper,
    conversation_hash,
    is_valid_delimiter,
    is_valid_stream_response,
    normalise_spaces,
    split_fragments,
    wrap_entity,
)


class TestSplitFragments(unittest.TestCase):
    """Test text fragment splitting."""

    @parameterized.expand(
        [
            ("empty_string", "", []),
            ("no_punctuation", "This is a test", ["This is a test"]),
            (
                "multiple_punctuation",
                "First sentence!! Second sentence??? Third sentence...",
                ["First sentence", "Second sentence", "Third sentence..."],
            ),
            (
                "newlines",
                "First line\n\nSecond line\nThird line",
                ["First line", "Second line", "Third line"],
            ),
            (
                "mixed_boundaries",
                "First sentence! Second line\nThird sentence? Fourth.",
                [
                    "First sentence",
                    "Second line",
                    "Third sentence",
                    "Fourth.",
                ],
            ),
        ]
    )
    def test_split_fragments(self, description, text, expected):
        """Test split_fragments with various inputs."""
        self.assertEqual(split_fragments(text), expected)


class TestNormaliseSpaces(unittest.TestCase):
    """Test space normalisation."""

    @parameterized.expand(
        [
            (
                "multiple_spaces",
                "Hello    world   test",
                "Hello world test",
            ),
            (
                "tabs_and_newlines",
                "Hello\t\tworld\n\ntest",
                "Hello world test",
            ),
            (
                "leading_trailing",
                "   Hello world   ",
                "Hello world",
            ),
            ("empty_string", "", ""),
            ("only_whitespace", "   \t\n   ", ""),
        ]
    )
    def test_normalise_spaces(self, description, text, expected):
        """Test normalise_spaces with various inputs."""
        self.assertEqual(normalise_spaces(text), expected)


class TestIsValidDelimiter(unittest.TestCase):
    """Test delimiter validation."""

    @parameterized.expand(
        [
            ("single_char", "<", True),
            ("double_char", ">>", True),
            ("triple_char", "|||", True),
            ("space", " ", True),
            ("empty_string", "", False),
            ("none", None, False),
            ("integer", 123, False),
            ("list", [], False),
        ]
    )
    def test_is_valid_delimiter(self, description, delimiter, expected):
        """Test delimiter validation with various inputs."""
        self.assertEqual(is_valid_delimiter(delimiter), expected)


class TestWrapEntity(unittest.TestCase):
    """Test entity wrapping with delimiters."""

    @parameterized.expand(
        [
            (
                "angle_brackets",
                EntityType.PERSON,
                0,
                "<",
                ">",
                "<PERSON_0>",
            ),
            (
                "square_brackets",
                EntityType.EMAIL,
                5,
                "[",
                "]",
                "[EMAIL_5]",
            ),
            (
                "double_braces",
                EntityType.ORGANISATION,
                1,
                "{{",
                "}}",
                "{{ORGANISATION_1}}",
            ),
            (
                "asterisks",
                EntityType.PLACE,
                10,
                "***",
                "***",
                "***PLACE_10***",
            ),
        ]
    )
    def test_wrap_entity(  # noqa: PLR0913, PLR0917
        self, description, entity_type, suffix, start, end, expected
    ):
        """Test entity wrapping with various delimiters."""
        self.assertEqual(
            wrap_entity(entity_type, suffix, start, end), expected
        )


class TestIsValidStreamResponse(unittest.TestCase):
    """Test stream response validation."""

    @parameterized.expand(
        [
            ("list", [1, 2, 3], True),
            ("tuple", (1, 2, 3), True),
            ("set", {1, 2, 3}, True),
            ("range", range(5), True),
            ("iterator", iter([1, 2, 3]), True),
            ("string", "string", False),
            ("bytes", b"bytes", False),
            ("bytearray", bytearray(b"bytearray"), False),
            ("dict", {"key": "value"}, False),
            ("integer", 123, False),
            ("none", None, False),
        ]
    )
    def test_is_valid_stream_response(self, description, obj, expected):
        """Test stream response validation with various types."""
        self.assertEqual(is_valid_stream_response(obj), expected)


class TestShouldCloakInput(unittest.TestCase):
    """Test input cloaking decisions."""

    @parameterized.expand(
        [
            ("empty_string", "", True),
            ("normal_string", "hello", True),
            ("empty_list", [], True),
            ("string_list", ["hello", "world"], True),
            ("mixed_list", ["string", 123], True),
            ("integer", 123, False),
            ("dict", {"key": "value"}, False),
            ("bytes", b"bytes", False),
            ("none", None, False),
            ("float", 3.14, False),
            ("path", Path("/test/path"), False),
            ("file_like", BytesIO(), False),
            ("tuple", ("tuple", "data"), False),
        ]
    )
    def test_should_cloak_input(self, description, input_value, expected):
        """Test cloaking decisions for various input types."""
        self.assertEqual(_should_cloak_input(input_value), expected)


class TestPydanticLikeProtocol(unittest.TestCase):
    """Test PydanticLike protocol runtime checking."""

    def test_valid_implementation(self):
        """Test that complete implementation satisfies protocol."""

        class ValidModel:
            """Minimal PydanticLike implementation."""

            def model_dump(self) -> dict:
                """Serialise to dict."""
                return {"test": "data"}

            @classmethod
            def model_validate(cls, data: dict):
                """Create from dict."""
                return cls()

        obj = ValidModel()
        self.assertIsInstance(obj, PydanticLike)
        self.assertEqual(obj.model_dump(), {"test": "data"})
        self.assertIsInstance(ValidModel.model_validate({}), ValidModel)

    @parameterized.expand(
        [
            ("complete_model", True, True, True),
            ("empty_model", False, False, False),
            ("missing_validate", True, False, False),
            ("missing_dump", False, True, False),
        ]
    )
    def test_protocol_matching(
        self, description, has_dump, has_validate, should_match
    ):
        """Test protocol matching with various implementations."""
        attrs = {}
        if has_dump:

            def model_dump(self) -> dict:
                """Serialise to dict."""
                return {}

            attrs["model_dump"] = model_dump
        if has_validate:

            @classmethod
            def model_validate(cls, data: dict):
                """Create from dict."""
                return cls()

            attrs["model_validate"] = model_validate

        Model = type("Model", (), attrs)
        result = isinstance(Model(), PydanticLike)
        self.assertEqual(result, should_match)


class TestConversationHash(unittest.TestCase):
    """Test conversation hashing."""

    def test_same_message_same_hash(self):
        """Test identical messages produce identical hashes."""
        msg = {"role": "user", "content": "Hello world"}
        self.assertEqual(
            conversation_hash(msg),
            conversation_hash({"role": "user", "content": "Hello world"}),
        )

    def test_different_message_different_hash(self):
        """Test different messages produce different hashes."""
        self.assertNotEqual(
            conversation_hash({"role": "user", "content": "Hello"}),
            conversation_hash({"role": "user", "content": "Different"}),
        )

    @parameterized.expand(
        [
            ("empty_content", {"role": "user", "content": ""}, ("user", "")),
            ("missing_content", {"role": "user"}, ("user", "")),
            ("empty_dict", {}, ("", "")),
            ("missing_role", {"content": "hello"}, ("", "hello")),
        ]
    )
    def test_edge_cases(self, description, message, expected_tuple):
        """Test conversation_hash with edge cases."""
        self.assertEqual(conversation_hash(message), hash(expected_tuple))

    def test_list_of_messages(self):
        """Test hashing a list of messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        hash1 = conversation_hash(messages)

        # Different order must produce different hash
        messages_reversed = list(reversed(messages))
        hash2 = conversation_hash(messages_reversed)
        self.assertNotEqual(hash1, hash2)

    def test_empty_list(self):
        """Test hashing empty list."""
        self.assertIsInstance(conversation_hash([]), int)


class TestAskHelper(unittest.TestCase):
    """Test ask_helper functionality."""

    def test_no_cloaking_needed(self):
        """Test ask_helper when no cloaking is needed."""
        mock_shield = Mock()
        mock_shield.provider.execute_raw.return_value = "Direct response"

        kwargs = {"prompt": {"key": "value"}, "model": "test"}
        result = ask_helper(mock_shield, stream=False, **kwargs)

        mock_shield.provider.execute_raw.assert_called_once_with(**kwargs)
        self.assertEqual(result, "Direct response")
        mock_shield.cloak.assert_not_called()

    def test_message_param(self):
        """Test ask_helper with 'message' parameter."""
        mock_shield = Mock()
        mock_shield.cloak.return_value = (
            "cloaked",
            {"<PERSON_0>": "John"},
        )
        mock_shield.uncloak.return_value = "uncloaked response"
        mock_shield.provider.execute_single_message.return_value = (
            "llm response",
            False,
        )

        kwargs = {"message": "Hello John", "model": "test"}
        result = ask_helper(mock_shield, stream=False, **kwargs)

        mock_shield.cloak.assert_called_once_with("Hello John", allowlist=None)
        self.assertEqual(result, "uncloaked response")

    def test_streaming_invalid_response(self):
        """Test ask_helper when LLM returns invalid stream response."""
        mock_shield = Mock()
        mock_shield.cloak.return_value = (
            "cloaked",
            {"<PERSON_0>": "John"},
        )
        mock_shield.uncloak.return_value = "uncloaked response"
        mock_shield.provider.execute_single_message.return_value = (
            "not_a_stream",
            True,
        )

        result = ask_helper(mock_shield, stream=True, prompt="Hello John")
        result_list = list(result)
        self.assertEqual(result_list, ["uncloaked response"])

    def test_streaming_valid_response(self):
        """Test ask_helper with valid streaming response."""
        mock_shield = Mock()
        mock_shield.cloak.return_value = (
            "cloaked",
            {"<PERSON_0>": "John"},
        )
        mock_shield.stream_uncloak.return_value = iter(["chunk1", "chunk2"])
        mock_shield.provider.execute_single_message.return_value = (
            ["stream", "chunks"],
            True,
        )

        ask_helper(mock_shield, stream=True, prompt="Hello John")
        mock_shield.stream_uncloak.assert_called_once()

    @parameterized.expand(
        [
            ("integer_prompt", "prompt", 123),
            ("path_message", "message", Path("/test/path")),
            ("bytes_prompt", "prompt", b"binary data"),
            ("none_prompt", "prompt", None),
        ]
    )
    def test_non_cloak_inputs_use_execute_raw(
        self, description, param_name, input_value
    ):
        """Test non-cloakable inputs are passed to execute_raw."""

        def mock_llm(**kwargs):
            """Simulate an LLM response."""
            return f"Response to: {kwargs.get(param_name, 'unknown')}"

        shield = LLMShield(llm_func=mock_llm)
        kwargs = {param_name: input_value, "stream": False}
        result = ask_helper(shield=shield, **kwargs)
        self.assertIn("Response to:", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
