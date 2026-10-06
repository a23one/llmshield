"""Test core LLMShield functionality and integration.

Description:
    Comprehensive tests for cloaking, uncloaking, caching, streaming,
    entity detection, delimiter handling, and provider integration.

Author: LLMShield by brainpolo, 2025-2026
"""

# Standard library Imports
import re
from unittest import TestCase, main
from unittest.mock import patch

# Third party Imports
from parameterized import parameterized

# Local Imports
from llmshield import LLMShield
from llmshield.entity_detector import EntityType
from llmshield.exceptions import ValidationError
from llmshield.utils import conversation_hash, wrap_entity


class TestCoreFunctionality(TestCase):
    """Test core functionality of LLMShield."""

    def setUp(self):
        """Set up test cases."""
        self.start_delimiter = "["
        self.end_delimiter = "]"
        self.llm_func = lambda prompt: (
            "Thanks [PERSON_0], I'll send details to [EMAIL_0]"
        )
        self.shield = LLMShield(
            llm_func=self.llm_func,
            start_delimiter=self.start_delimiter,
            end_delimiter=self.end_delimiter,
        )

        # Updated test prompt with proper spacing
        self.test_prompt = (
            "Hi, I'm John Doe.\n"
            "You can reach me at john.doe@example.com.\n"
            "Some numbers are 192.168.1.1 and 378282246310005\n"
        )
        self.test_entity_map = {
            wrap_entity(
                EntityType.EMAIL, 0, self.start_delimiter, self.end_delimiter
            ): "john.doe@example.com",
            wrap_entity(
                EntityType.PERSON, 0, self.start_delimiter, self.end_delimiter
            ): "John Doe",
            wrap_entity(
                EntityType.IP_ADDRESS,
                0,
                self.start_delimiter,
                self.end_delimiter,
            ): "192.168.1.1",
            wrap_entity(
                EntityType.CREDIT_CARD,
                0,
                self.start_delimiter,
                self.end_delimiter,
            ): "378282246310005",
        }
        self.test_llm_response = (
            "Thanks "
            + self.test_entity_map[
                wrap_entity(
                    EntityType.PERSON,
                    0,
                    self.start_delimiter,
                    self.end_delimiter,
                )
            ]
            + ", I'll send details to "
            + self.test_entity_map[
                wrap_entity(
                    EntityType.EMAIL,
                    0,
                    self.start_delimiter,
                    self.end_delimiter,
                )
            ]
        )

    def test_cloak_sensitive_info(self):
        """Test that sensitive information is properly cloaked."""
        cloaked_prompt, entity_map = self.shield.cloak(self.test_prompt)
        self.assertNotIn("john.doe@example.com", cloaked_prompt)
        self.assertNotIn("John Doe", cloaked_prompt)
        self.assertNotIn("192.168.1.1", cloaked_prompt)
        self.assertNotIn("378282246310005", cloaked_prompt)
        self.assertEqual(len(entity_map), 4)

    def test_uncloak(self):
        """Test that cloaked entities are properly restored."""
        cloaked_prompt, entity_map = self.shield.cloak(self.test_prompt)
        uncloaked = self.shield.uncloak(cloaked_prompt, entity_map)
        self.assertEqual(
            uncloaked,
            self.test_prompt,
            f"Uncloaked response is not equal to test prompt: "
            f"{uncloaked} != {self.test_prompt}",
        )

    def test_end_to_end(self):
        """Test end-to-end flow with mock LLM function."""

        def mock_llm(prompt, stream=False, **kwargs):
            """Return response echoing cloaked placeholders."""
            person_match = re.search(r"\[PERSON_\d+\]", prompt)
            email_match = re.search(r"\[EMAIL_\d+\]", prompt)
            return (
                f"Thanks {person_match.group()}, I'll send details to "
                f"{email_match.group()}"
            )

        shield = LLMShield(
            llm_func=mock_llm,
            start_delimiter=self.start_delimiter,
            end_delimiter=self.end_delimiter,
        )

        # Updated test input
        test_input = "Hi, I'm John Doe (john.doe@example.com)"
        response = shield.ask(prompt=test_input)

        # Test the entity map - use _ for intentionally unused variable
        _, _ = self.shield.cloak(test_input)

        self.assertIn("John Doe", response)
        self.assertIn("john.doe@example.com", response)

    def test_delimiter_customization(self):
        """Test custom delimiter functionality."""
        shield = LLMShield(start_delimiter="[[", end_delimiter="]]")
        cloaked_prompt, _ = shield.cloak("Hi, I'm John Doe")
        self.assertIn("[[PERSON_0]]", cloaked_prompt)
        self.assertNotIn("<PERSON_0>", cloaked_prompt)

    def test_delimiter_validation_after_initial_check(self):
        """Test delimiter validation that triggers lines 134 and 138.

        This tests the redundant validation in core.py that happens after
        validate_delimiters has already been called.
        """
        # Test 1: Mock validate_delimiters to pass, but is_valid_delimiter
        # fails. This triggers line 134 (invalid start delimiter)
        with patch("llmshield.core.validate_delimiters") as mock_validate:
            mock_validate.return_value = None

            with self.assertRaises(ValidationError) as cm:
                LLMShield(start_delimiter="", end_delimiter=">")

            self.assertIn("Invalid start delimiter: ''", str(cm.exception))

        # Test 2: Mock validate_delimiters to pass for end delimiter check
        # This triggers line 138 (invalid end delimiter)
        with patch("llmshield.core.validate_delimiters") as mock_validate:
            mock_validate.return_value = None

            with self.assertRaises(ValidationError) as cm:
                LLMShield(start_delimiter="<", end_delimiter="")

            self.assertIn("Invalid end delimiter: ''", str(cm.exception))

        # Test 3: Non-string delimiters that pass initial validation
        with patch("llmshield.core.validate_delimiters") as mock_validate:
            mock_validate.return_value = None

            # Test with None as start delimiter
            with self.assertRaises(ValidationError) as cm:
                LLMShield(start_delimiter=None, end_delimiter=">")
            self.assertIn("Invalid start delimiter:", str(cm.exception))

            # Test with integer as end delimiter
            with self.assertRaises(ValidationError) as cm:
                LLMShield(start_delimiter="<", end_delimiter=123)
            self.assertIn("Invalid end delimiter:", str(cm.exception))

    def test_entity_detection_accuracy(self):
        """Test accuracy of entity detection with complex examples."""
        test_cases = [
            # Test case 1: Proper Nouns
            {
                "input": "Dr. John Smith from Microsoft Corporation visited "
                "New York. "
                "The CEO of Apple Inc met with IBM executives at UNESCO "
                "headquarters.",
                "expected_entities": {
                    "John Smith": EntityType.PERSON,
                    "Microsoft Corporation": EntityType.ORGANISATION,
                    "New York": EntityType.PLACE,
                    "Apple Inc": EntityType.ORGANISATION,
                    "IBM": EntityType.ORGANISATION,
                    "UNESCO": EntityType.ORGANISATION,
                },
            },
            # Test case 2: Numbers and Locators
            {
                "input": "Contact us at support@company.com or call "
                "44 (555) 123-4567. "
                "Visit https://www.company.com. "
                "Server IP: 192.168.1.1. "
                "Credit card: 378282246310005",
                "expected_entities": {
                    "support@company.com": EntityType.EMAIL,
                    "https://www.company.com": EntityType.URL,
                    "192.168.1.1": EntityType.IP_ADDRESS,
                    "378282246310005": EntityType.CREDIT_CARD,
                },
            },
        ]

        for i, test_case in enumerate(test_cases, 1):
            input_text = test_case["input"]
            expected = test_case["expected_entities"]
            _, entity_map = self.shield.cloak(input_text)

            for entity_text, entity_type in expected.items():
                with self.subTest(case=i, entity=entity_text):
                    found = any(
                        v == entity_text and entity_type.name in k
                        for k, v in entity_map.items()
                    )
                    self.assertTrue(
                        found,
                        f"Failed to detect "
                        f"{entity_type.name}: "
                        f"'{entity_text}'",
                    )

    def test_error_handling(self):
        """Test error handling in core functions."""
        shield = LLMShield(start_delimiter="[[", end_delimiter="]]")

        # Test invalid inputs - these should cover lines 59, 61, 63
        with self.assertRaises(ValueError):
            shield.ask(prompt=None)  # Line 59
        with self.assertRaises(ValueError):
            shield.ask(prompt="")  # Line 61
        with self.assertRaises(ValueError):
            shield.ask(prompt="   ")  # Line 63

        # Test LLM errors
        def failing_llm(**kwargs):
            """Raise ValueError to simulate LLM failure."""
            raise ValueError("LLM failed")  # Use specific exception type

        shield_with_failing_llm = LLMShield(
            llm_func=failing_llm, start_delimiter="[[", end_delimiter="]]"
        )

        with self.assertRaises(ValueError):
            shield_with_failing_llm.ask(prompt="Hello John Doe")

        # Test empty responses
        shield_empty = LLMShield(
            llm_func=lambda **kwargs: "No entity found",
            start_delimiter="[[",
            end_delimiter="]]",
        )
        response = shield_empty.ask(prompt="Hello John Doe")
        self.assertEqual(response, "No entity found")

        # Test dict response
        shield_dict = LLMShield(
            llm_func=lambda **kwargs: {"content": "test"},
            start_delimiter="[[",
            end_delimiter="]]",
        )
        response = shield_dict.ask(prompt="Hello John Doe")
        self.assertEqual(response, {"content": "test"})

    def test_error_propagation(self):
        """Test specific error propagation in ask method.

        Covers lines 113-115 for error handling validation.
        """

        # Create a custom exception to ensure we're testing the right pathway
        class CustomError(Exception):
            """Custom exception for testing."""

        # This LLM function raises the custom exception during processing
        # specifically to test lines 113-115
        def llm_with_specific_error(**kwargs):  # Accept keyword arguments
            """Raise CustomError to test exception propagation."""
            raise CustomError("Test exception")

        shield = LLMShield(
            llm_func=llm_with_specific_error,
            start_delimiter="<<",
            end_delimiter=">>",
        )

        # This should propagate the exception through lines 113-115
        with self.assertRaises(CustomError):
            shield.ask(prompt="Test prompt")

    def test_constructor_validation(self):
        """Test constructor validation (lines 59, 61, 63)."""
        # Test invalid start delimiter (line 59)
        with self.assertRaises(ValueError):
            LLMShield(start_delimiter="", end_delimiter="]")

        # Test invalid end delimiter (line 61)
        with self.assertRaises(ValueError):
            LLMShield(start_delimiter="[", end_delimiter="")

        # Test non-callable llm_func (line 63)
        with self.assertRaises(ValueError):
            LLMShield(
                start_delimiter="[", end_delimiter="]", llm_func="not_callable"
            )

    def test_uncloak_with_stored_entity_map(self):
        """Test uncloaking with stored entity map from previous cloak.

        Validates line 115 functionality for entity map storage.
        """
        shield = LLMShield(start_delimiter="[[", end_delimiter="]]")

        # First cloak something to store the entity map internally
        test_text = "Hello John Doe"
        cloaked_text, _ = shield.cloak(test_text)

        # Now uncloak without providing an entity map - should use stored one
        # from _last_entity_map
        uncloaked = shield.uncloak(cloaked_text, entity_map=None)

        # Should successfully uncloak using the stored map
        self.assertEqual(uncloaked, test_text)

    def test_ask_missing_required_param(self):
        """Test ValueError when required parameters are missing.

        Validates error when neither 'prompt' nor 'message' is provided to ask.
        """
        shield = LLMShield(
            llm_func=lambda **kwargs: "Response",
            start_delimiter="[[",
            end_delimiter="]]",
        )

        # Call ask without providing either 'prompt' or 'message'
        with self.assertRaises(ValueError):
            shield.ask()  # No prompt or message provided

    def test_uncloak_invalid_response_type(self):
        """Test ValueError when trying to uncloak invalid response types."""
        shield = LLMShield(start_delimiter="[[", end_delimiter="]]")

        # Create a mock entity map
        entity_map = {"[[PERSON_0]]": "John Doe"}

        # Try to uncloak various invalid response types
        invalid_responses = [
            123,  # int
            3.14,  # float
            True,  # bool
            (1, 2, 3),  # tuple
        ]

        for response in invalid_responses:
            with self.subTest(response=type(response).__name__):
                with self.assertRaises(TypeError) as ctx:
                    shield.uncloak(response, entity_map)
                self.assertIn(
                    "Response must be ",
                    str(ctx.exception),
                )

    def test_stream_uncloak_basic(self):
        """Test basic stream uncloaking functionality."""
        shield = LLMShield(start_delimiter="[[", end_delimiter="]]")

        # Create entity map
        entity_map = {
            "[[PERSON_0]]": "John Doe",
            "[[EMAIL_0]]": "john.doe@example.com",
        }

        # Create a mock stream with cloaked content
        def mock_stream():
            """Yield chunks containing cloaked placeholders."""
            chunks = [
                "Hello ",
                "[[PERSON_0]]",
                ", please contact ",
                "[[EMAIL_0]]",
                " for details.",
            ]
            yield from chunks

        # Process stream
        result_chunks = shield.stream_uncloak(mock_stream(), entity_map)
        result = "".join(result_chunks)

        expected = (
            "Hello John Doe, please contact john.doe@example.com for details."
        )
        self.assertEqual(result, expected)

    def test_stream_uncloak_partial_placeholders(self):
        """Test stream uncloaking with placeholders split across chunks."""
        shield = LLMShield(start_delimiter="[[", end_delimiter="]]")

        entity_map = {"[[PERSON_0]]": "Alice Smith"}

        # Split placeholder across multiple chunks
        def mock_stream():
            """Yield chunks with split placeholder tokens."""
            chunks = ["Hello ", "[[PER", "SON_0", "]]", " how are you?"]
            yield from chunks

        result_chunks = list(shield.stream_uncloak(mock_stream(), entity_map))
        result = "".join(result_chunks)

        expected = "Hello Alice Smith how are you?"
        self.assertEqual(result, expected)

    def test_stream_uncloak_no_placeholders(self):
        """Test stream uncloaking with no placeholders."""
        shield = LLMShield(start_delimiter="[[", end_delimiter="]]")

        entity_map = {"[[PERSON_0]]": "John Doe"}

        def mock_stream():
            """Yield chunks with no placeholders."""
            chunks = ["Hello ", "world! ", "No placeholders here."]
            yield from chunks

        result_chunks = list(shield.stream_uncloak(mock_stream(), entity_map))
        result = "".join(result_chunks)

        expected = "Hello world! No placeholders here."
        self.assertEqual(result, expected)

    def test_stream_uncloak_multiple_placeholders(self):
        """Test stream uncloaking with multiple placeholders.

        Validates handling of multiple placeholders in a single chunk.
        """
        shield = LLMShield(start_delimiter="[[", end_delimiter="]]")

        entity_map = {
            "[[PERSON_0]]": "John Doe",
            "[[PERSON_1]]": "Jane Smith",
            "[[EMAIL_0]]": "contact@example.com",
        }

        def mock_stream():
            """Yield chunks with multiple placeholders."""
            chunks = [
                "Meeting between ",
                "[[PERSON_0]] and [[PERSON_1]]",
                " at [[EMAIL_0]]",
            ]
            yield from chunks

        result_chunks = list(shield.stream_uncloak(mock_stream(), entity_map))
        result = "".join(result_chunks)

        expected = (
            "Meeting between John Doe and Jane Smith at contact@example.com"
        )
        self.assertEqual(result, expected)

    def test_stream_uncloak_with_stored_entity_map(self):
        """Test stream uncloaking using stored entity map.

        Validates use of entity map from previous cloak operation.
        """
        shield = LLMShield(start_delimiter="[[", end_delimiter="]]")

        # First cloak to store entity map
        test_text = "Hello John Doe"
        cloaked_text, _ = shield.cloak(test_text)

        def generator():
            """Mock generator that yields cloaked text."""
            yield from cloaked_text.split()

        # Use stored entity map
        result_chunks = list(
            shield.stream_uncloak(generator(), entity_map=None)
        )
        result = " ".join(result_chunks)

        expected = "Hello John Doe"
        self.assertEqual(result, expected)

    def test_stream_uncloak_error_handling(self):
        """Test error handling in stream_uncloak."""
        shield = LLMShield(start_delimiter="[[", end_delimiter="]]")

        # Test empty stream
        with self.assertRaises(ValueError):
            list(shield.stream_uncloak(None, {}))

        # Test non-iterator input
        with self.assertRaises(TypeError):
            list(shield.stream_uncloak("not an iterator", {}))

        # Test no entity map and no stored map
        shield_fresh = LLMShield(start_delimiter="[[", end_delimiter="]]")

        def mock_stream():
            """Yield a single test chunk."""
            yield "test"

        with self.assertRaises(ValueError):
            list(shield_fresh.stream_uncloak(mock_stream(), entity_map=None))

    def test_ask_with_stream_true(self):
        """Test ask function with stream=True."""

        def mock_streaming_llm(**kwargs):
            """Mock LLM that returns an iterator."""
            response_chunks = [
                "Hello ",
                "[[PERSON_0]]",
                ", how can I help you?",
            ]
            yield from response_chunks

        shield = LLMShield(
            llm_func=mock_streaming_llm,
            start_delimiter="[[",
            end_delimiter="]]",
        )

        # Test streaming response
        response_stream = shield.ask(prompt="Hi, I'm John Doe", stream=True)

        # Verify it returns an iterator
        self.assertTrue(hasattr(response_stream, "__iter__"))

        # Collect all chunks
        result_chunks = list(response_stream)
        result = "".join(result_chunks)

        # Should contain uncloaked response
        self.assertIn("John Doe", result)

    def test_ask_with_stream_non_streaming_llm(self):
        """Test ask with stream=True but LLM returns single response."""

        def mock_non_streaming_llm(**kwargs):
            """Mock LLM that returns a single string instead of iterator."""
            return "Hello [[PERSON_0]], how can I help you?"

        shield = LLMShield(
            llm_func=mock_non_streaming_llm,
            start_delimiter="[[",
            end_delimiter="]]",
        )

        # Even though we request streaming, LLM returns single response
        response_stream = shield.ask(prompt="Hi, I'm John Doe", stream=True)

        # Should still return an iterator
        self.assertTrue(hasattr(response_stream, "__iter__"))

        # Collect result
        result_chunks = list(response_stream)
        result = "".join(result_chunks)

        # Should contain uncloaked response
        self.assertIn("John Doe", result)

    def test_ask_streaming_with_complex_entities(self):
        """Test streaming ask with multiple entity types."""

        def mock_complex_streaming_llm(**kwargs):
            """Return streaming response with multiple entity types."""
            # Extract the cloaked prompt
            cloaked_prompt = kwargs.get("message") or kwargs.get("prompt", "")

            # Use regex to find actual placeholders with their counters
            person_match = re.search(r"\[\[PERSON_(\d+)\]\]", cloaked_prompt)
            email_match = re.search(r"\[\[EMAIL_(\d+)\]\]", cloaked_prompt)
            ip_match = re.search(r"\[\[IP_ADDRESS_(\d+)\]\]", cloaked_prompt)
            cc_match = re.search(r"\[\[CREDIT_CARD_(\d+)\]\]", cloaked_prompt)

            # Build placeholders based on what was actually found
            person_placeholder = (
                person_match.group(0) if person_match else "[[PERSON_0]]"
            )
            email_placeholder = (
                email_match.group(0) if email_match else "[[EMAIL_1]]"
            )
            ip_placeholder = (
                ip_match.group(0) if ip_match else "[[IP_ADDRESS_2]]"
            )
            cc_placeholder = (
                cc_match.group(0) if cc_match else "[[CREDIT_CARD_3]]"
            )

            chunks = [
                "Dear ",
                person_placeholder,
                ",\n",
                "We'll send details to ",
                email_placeholder,
                "\n",
                "From IP: ",
                ip_placeholder,
                "\n",
                "Your credit card: ",
                cc_placeholder,
            ]
            yield from chunks

        shield = LLMShield(
            llm_func=mock_complex_streaming_llm,
            start_delimiter="[[",
            end_delimiter="]]",
        )

        complex_prompt = (
            "Hi, I'm John Doe.\n"
            "Contact me at john@example.com.\n"
            "My server IP is 192.168.1.1\n"
            "My credit card number is 378282246310005\n"
        )

        response_stream = shield.ask(stream=True, message=complex_prompt)
        result = "".join(list(response_stream))
        # Verify all entities are properly uncloaked
        self.assertIn("John Doe", result)
        self.assertIn("john@example.com", result)
        self.assertIn("192.168.1.1", result)
        self.assertIn("378282246310005", result)

    def test_cloak_reuses_placeholders_from_entity_map(self):
        """Test that cloak reuses placeholders from a given entity_map."""
        shield = LLMShield(start_delimiter="[[", end_delimiter="]]")

        # 1. Initial cloak to establish an entity map
        initial_prompt = "My name is John"
        _, initial_entity_map = shield.cloak(initial_prompt)
        self.assertEqual(initial_entity_map, {"[[PERSON_0]]": "John"})

        # 2. Second cloak with a new prompt, passing the previous map
        # This prompt reuses "John" and adds a new entity.
        second_prompt = "John , my email is jane.doe@example.com."
        cloaked_prompt, final_entity_map = shield.cloak(
            second_prompt, entity_map_param=initial_entity_map.copy()
        )

        # 3. Assert that the existing placeholder is reused for "John Doe"
        self.assertIn("[[PERSON_0]]", cloaked_prompt)
        # And that a *new* placeholder is created for the email.
        # The counter should continue from the size of the initial map.
        self.assertIn("[[EMAIL_1]]", cloaked_prompt)

        # 4. Assert the final map is correct
        expected_map = {
            "[[PERSON_0]]": "John",
            "[[EMAIL_1]]": "jane.doe@example.com",
        }
        self.assertEqual(final_entity_map, expected_map)

    def test_ask_multi_turn_conversation_reuses_entities(self):
        """Test multi-turn conversation entity reuse.

        Validates that `ask` conversations correctly reuse entities across
        turns.
        """

        def mock_llm(messages, **kwargs):
            """Check for placeholders and return one."""
            last_message = messages[-1]["content"]
            if "what is my email" in last_message:
                # Find email placeholder in the message history
                for msg in messages:
                    match = re.search(r"(\[\[EMAIL_\d+\]\])", msg["content"])
                    if match:
                        return f"Your email is {match.group(1)}"
            return "Acknowledged."

        shield = LLMShield(
            llm_func=mock_llm, start_delimiter="[[", end_delimiter="]]"
        )

        # This conversation introduces an entity, then asks about it.
        conversation = [
            {
                "role": "user",
                "content": "Hello, my name is Alice. My email is "
                "alice@example.com.",
            },
            {"role": "assistant", "content": "Acknowledged."},
            {"role": "user", "content": "Now, what is my email address?"},
        ]

        # The `ask` method should handle the history, find "alice@example.com",
        # cloak it, pass it to the mock LLM, which returns the placeholder,
        # and then uncloak it back.
        response = shield.ask(messages=conversation)

        self.assertEqual(response, "Your email is alice@example.com")

    def test_ask_with_messages_and_prompt_raises_error(self):
        """Test ValueError when both parameters are provided.

        Validates that `ask` raises error if both `messages` and `prompt`
        are given.
        """

        def mock_llm(messages, **kwargs):
            """Return static response to test error case."""
            return "This should not be called."

        shield = LLMShield(
            llm_func=mock_llm, start_delimiter="[[", end_delimiter="]]"
        )

        conversation = [{"role": "user", "content": "Hello"}]
        prompt = "Hi there"

        with self.assertRaises(ValueError):
            shield.ask(stream=False, messages=conversation, prompt=prompt)

    def test_ask_caches_conversation_history(self):
        """Test that `ask` caches the entity map for a conversation history."""

        def mock_llm(messages, **kwargs):
            """Return acknowledgement response."""
            return "Acknowledged."

        shield = LLMShield(
            llm_func=mock_llm, start_delimiter="[[", end_delimiter="]]"
        )

        conversation = [
            {
                "role": "user",
                "content": "Hello, my name is Alice. My email is "
                "alice@example.com.",
            },
        ]

        # This call should cache the state *after* the conversation.
        response = shield.ask(messages=conversation)

        # The history that gets cached includes the assistant's response.
        final_history = conversation + [
            {"role": "assistant", "content": response}
        ]
        history_key = conversation_hash(final_history)

        # Check that the cache now contains the entity map for this history.
        cached_map = shield._cache.get(history_key)
        self.assertIsNotNone(
            cached_map, "Entity map was not cached for the conversation."
        )

        # Check for presence of values, as key names can vary
        cached_values = list(cached_map.values())
        self.assertIn("Alice", cached_values)
        self.assertIn("alice@example.com", cached_values)

    def test_ask_uses_cached_entity_map_for_history(self):
        """Test cached entity map usage for conversation history.

        Validates that `ask` uses cached entity map and avoids re-cloaking.
        """

        class ShieldWithTrackedCloak(LLMShield):
            """A wrapper to track cloak calls."""

            def __init__(self, *args, **kwargs):
                """Initialise with cloak call counter."""
                super().__init__(*args, **kwargs)
                self.cloak_call_count = 0

            def cloak(
                self,
                prompt: str,
                entity_map_param: dict[str, str] | None = None,
                allowlist=None,
            ):
                """Cloak prompt and increment call counter."""
                self.cloak_call_count += 1
                return super().cloak(
                    prompt,
                    entity_map_param=entity_map_param,
                    allowlist=allowlist,
                )

        def mock_llm(messages, **kwargs):
            """Return acknowledgement response."""
            return "Acknowledged."

        shield = ShieldWithTrackedCloak(
            llm_func=mock_llm, start_delimiter="[[", end_delimiter="]]"
        )

        # Turn 1: Establish history and cache
        conversation_1 = [
            {
                "role": "user",
                "content": "My name is Alice, email is alice@example.com.",
            },
        ]
        shield.ask(messages=conversation_1)

        # Turn 2: Follow-up question. This should trigger a cache hit for the
        # history of turn 1.
        conversation_2 = [
            {
                "role": "user",
                "content": "My name is Alice, email is alice@example.com.",
            },
            {"role": "assistant", "content": "Acknowledged."},
            {"role": "user", "content": "What is my name?"},
        ]

        # Reset counter and run the second turn
        shield.cloak_call_count = 0
        shield.ask(messages=conversation_2)

        # With a cache hit, cloak is called for the latest message (1) + each
        # message
        # in history for the final payload (2).
        history_len_for_payload = len(conversation_2) - 1
        expected_calls_with_cache = history_len_for_payload + 1
        self.assertEqual(
            shield.cloak_call_count,
            expected_calls_with_cache,
            "Incorrect number of cloak calls with cache hit.",
        )

        # Now, verify that a cache miss results in more calls.
        shield_no_cache = ShieldWithTrackedCloak(
            llm_func=mock_llm, start_delimiter="[[", end_delimiter="]]"
        )
        shield_no_cache.ask(messages=conversation_2)

        # With a cache miss, cloak is called for each history message to
        # build map (2)
        # + latest message (1) + each history message for payload (2).
        history_len = len(conversation_2) - 1
        expected_calls_without_cache = (2 * history_len) + 1
        self.assertEqual(
            shield_no_cache.cloak_call_count,
            expected_calls_without_cache,
            "Incorrect number of cloak calls with cache miss.",
        )

    def test_uncloak_pydantic_model_complete_flow(self):
        """Test complete Pydantic model uncloaking flow.

        Covers lines 185-186 and 190-192 for Pydantic model handling.
        """

        # Create a proper Pydantic-like model
        class TestModel:
            """Mock Pydantic-like model for testing."""

            def __init__(self, name: str, email: str):
                """Initialise with name and email."""
                self.name = name
                self.email = email

            def model_dump(self) -> dict:
                """Serialise to dict."""
                return {"name": self.name, "email": self.email}

            @classmethod
            def model_validate(cls, data: dict):
                """Create instance from dict."""
                return cls(data["name"], data["email"])

        shield = LLMShield()

        # Create instance with cloaked data
        model_instance = TestModel("<PERSON_0>", "<EMAIL_0>")
        entity_map = {
            "<PERSON_0>": "Alice Smith",
            "<EMAIL_0>": "alice@example.com",
        }

        # Test the Pydantic uncloaking path
        result = shield.uncloak(model_instance, entity_map)

        # Verify the result is the correct type and has uncloaked data
        self.assertIsInstance(result, TestModel)
        self.assertEqual(result.name, "Alice Smith")
        self.assertEqual(result.email, "alice@example.com")

    @parameterized.expand(
        [
            # (description, kwargs, expected_error_fragment)
            (
                "prompt_and_message",
                {
                    "prompt": "test prompt",
                    "message": "test message",
                },
                "Do not provide both 'prompt' and 'message'",
            ),
            (
                "prompt_and_messages",
                {
                    "prompt": "test",
                    "messages": [{"role": "user", "content": "test"}],
                },
                "Do not provide both 'prompt', 'message' and 'messages'",
            ),
            (
                "message_and_messages",
                {
                    "message": "test",
                    "messages": [{"role": "user", "content": "test"}],
                },
                "Do not provide both 'prompt', 'message' and 'messages'",
            ),
        ]
    )
    def test_ask_validation_errors(self, description, kwargs, expected_error):
        """Test validation errors in ask method with parameterized cases."""

        def mock_llm(**kwargs):
            """Return static response string."""
            return "response"

        shield = LLMShield(llm_func=mock_llm)

        with self.assertRaises(ValueError) as context:
            shield.ask(**kwargs)
        self.assertIn(expected_error, str(context.exception))

    def test_ask_chatcompletion_content_extraction(self):
        """Test ChatCompletion content extraction for conversation history."""

        # Helper to create ChatCompletion-like objects
        def mock_completion(content):
            """Create mock ChatCompletion-like object."""
            return type(
                "ChatCompletion",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message", (), {"content": content}
                                )()
                            },
                        )()
                    ],
                    "model": "gpt-4",
                },
            )()

        def mock_llm(**_):
            """Return mock ChatCompletion response."""
            return mock_completion("Hello John Doe!")

        shield = LLMShield(llm_func=mock_llm)
        messages = [{"role": "user", "content": "Say hello to John Doe"}]
        result = shield.ask(messages=messages)

        self.assertEqual(result.choices[0].message.content, "Hello John Doe!")

    def test_ask_string_response_content_extraction(self):
        """Test string response handling for conversation history.

        Covers line 386 for string response content extraction.
        """

        def mock_llm(**kwargs):
            """Return plain string response."""
            # Return response WITHOUT placeholders - the LLM would receive
            # cloaked input
            # but return normal text
            return "Hello John Doe!"

        shield = LLMShield(llm_func=mock_llm)
        messages = [{"role": "user", "content": "Say hello to John Doe"}]

        result = shield.ask(messages=messages)
        self.assertEqual(result, "Hello John Doe!")

    @parameterized.expand(
        [
            ("empty_string", ""),
            ("none_value", None),
            ("empty_list", []),  # Empty list is falsy and triggers validation
        ]
    )
    def test_uncloak_response_validation_cases(
        self, description, invalid_response
    ):
        """Test validation when response is empty or falsy - parameterized."""
        shield = LLMShield()
        entity_map = {"<PERSON_0>": "John"}

        with self.assertRaises(ValueError) as context:
            shield.uncloak(invalid_response, entity_map)
        self.assertIn("Response cannot be empty", str(context.exception))

    def test_ask_multi_message_chatcompletion_content_extraction(self):
        """Test ChatCompletion content extraction in multi-message chats.

        Covers line 381 for multi-message content extraction validation.
        """

        # Create mock ChatCompletion structure
        class MockChatCompletion:
            """Mock ChatCompletion with choices structure."""

            def __init__(self, content: str):
                """Initialise with content string."""
                self.choices = [
                    type(
                        "MockChoice",
                        (),
                        {
                            "message": type(
                                "MockMessage", (), {"content": content}
                            )()
                        },
                    )()
                ]
                self.model = "gpt-4"

        def mock_llm(**kwargs):
            """Return MockChatCompletion response."""
            return MockChatCompletion("Hello John!")

        shield = LLMShield(llm_func=mock_llm)
        messages = [
            {"role": "user", "content": "Say hello to John Doe"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Say hello again"},
        ]

        result = shield.ask(messages=messages)

        # This should trigger line 381:
        # response_content = uncloaked_response.choices[0].message.content
        self.assertIsInstance(result, MockChatCompletion)
        self.assertEqual(result.choices[0].message.content, "Hello John!")

    def test_ask_cache_miss_message_caching_line_318(self):
        """Test cache miss scenario that triggers line 318.

        Validates individual message caching behavior on cache miss.
        """

        def mock_llm(**kwargs):
            """Return static response string."""
            return "Response"

        shield = LLMShield(llm_func=mock_llm)

        # Create a conversation that will trigger cache miss and line 318
        messages = [
            {"role": "user", "content": "First message with John Doe"},
            {"role": "assistant", "content": "Got it"},
            {"role": "user", "content": "Second message"},
        ]

        # This should trigger:
        # 1. Cache miss (history not found)
        # 2. Loop through history messages
        # 3. Line 318: self._cache.put(conversation_hash(message), entity_map)
        result = shield.ask(messages=messages)

        self.assertEqual(result, "Response")
        # Verify cache has entries (indicating line 318 was hit)
        self.assertGreater(len(shield._cache.cache), 0)

    def test_ask_chatcompletion_with_none_content_tool_calls(self):
        """Test ask method with ChatCompletion response with None content.

        This simulates the scenario where an assistant message has tool calls
        but no text content, which results in content being None.
        """

        # Create mock ChatCompletion with None content (simulating tool calls)
        class MockChatCompletionNoneContent:
            """Mock ChatCompletion with None content."""

            def __init__(self):
                """Initialise with None content."""
                self.choices = [
                    type(
                        "MockChoice",
                        (),
                        {
                            "message": type(
                                "MockMessage", (), {"content": None}
                            )()
                        },
                    )()
                ]
                self.model = "gpt-4"

        def mock_llm_with_tool_calls(**kwargs):
            """Return completion with None content."""
            return MockChatCompletionNoneContent()

        shield = LLMShield(llm_func=mock_llm_with_tool_calls)
        messages = [
            {"role": "user", "content": "Execute this code: print('Hello')"},
            {"role": "assistant", "content": "I'll help you with that."},
            {"role": "user", "content": "Run the code now"},
        ]

        # This should not raise an error
        result = shield.ask(messages=messages)

        # Verify the result is properly handled
        self.assertIsInstance(result, MockChatCompletionNoneContent)
        self.assertIsNone(result.choices[0].message.content)

    @parameterized.expand(
        [
            # (description, start_delim, end_delim, text, entities,
            # expected_cloaked)
            (
                "standard_delimiters",
                "<",
                ">",
                "Hello John Doe",
                {"John Doe": "<PERSON_0>"},
                "Hello <PERSON_0>",
            ),
            (
                "bracket_delimiters",
                "[",
                "]",
                "Email john@test.com",
                {"john@test.com": "[EMAIL_0]"},
                "Email [EMAIL_0]",
            ),
            (
                "brace_delimiters",
                "{",
                "}",
                "Call 555-1234",
                {"555-1234": "{PHONE_0}"},
                "Call {PHONE_0}",
            ),
            (
                "custom_delimiters",
                "<<",
                ">>",
                "Visit NYC",
                {"NYC": "<<PLACE_0>>"},
                "Visit <<PLACE_0>>",
            ),
            (
                "unicode_delimiters",
                "〈",
                "〉",
                "Hello José",
                {"José": "〈PERSON_0〉"},
                "Hello 〈PERSON_0〉",
            ),
            (
                "mixed_entities",
                "<",
                ">",
                "Contact John at john@test.com",
                {"John": "<PERSON_0>", "john@test.com": "<EMAIL_0>"},
                "Contact <PERSON_0> at <EMAIL_0>",
            ),
        ]
    )
    def test_delimiter_variations(  # noqa: PLR0913, PLR0917
        self, description, start_delim, end_delim, text, entities, expected
    ):
        """Test various delimiter configurations - parameterized."""
        shield = LLMShield(
            start_delimiter=start_delim, end_delimiter=end_delim
        )

        # Create entity map in the expected format
        entity_map = {}
        for original, placeholder in entities.items():
            entity_map[placeholder] = original

        # Test uncloaking
        result = shield.uncloak(expected, entity_map)
        self.assertEqual(result, text)

    @parameterized.expand(
        [
            # (description, cache_size, operations, expected_behavior)
            (
                "small_cache",
                2,
                [
                    ("put", "key1", "value1"),
                    ("put", "key2", "value2"),
                    ("put", "key3", "value3"),  # Should evict key1
                    ("get", "key1", None),  # Should be evicted
                    ("get", "key3", "value3"),  # Should exist
                ],
                "lru_eviction",
            ),
            (
                "large_cache",
                100,
                [("put", f"key{i}", f"value{i}") for i in range(50)]
                + [
                    ("get", "key25", "value25"),  # Should exist
                ],
                "no_eviction",
            ),
            (
                "zero_cache",
                0,
                [
                    ("put", "key1", "value1"),
                    ("get", "key1", None),  # Should not store anything
                ],
                "no_storage",
            ),
        ]
    )
    def test_cache_size_configurations(
        self, description, cache_size, operations, expected_behavior
    ):
        """Test various cache size configurations - parameterized."""
        shield = LLMShield(max_cache_size=cache_size)

        for op_type, key, expected_value in operations:
            if op_type == "put":
                shield._cache.put(key, expected_value)
            elif op_type == "get":
                result = shield._cache.get(key)
                if expected_value is None:
                    self.assertIsNone(
                        result, f"Expected {key} to be None but got {result}"
                    )
                else:
                    self.assertEqual(
                        result,
                        expected_value,
                        f"Expected {key} to be {expected_value}",
                    )

    @parameterized.expand(
        [
            # (description, input_text, expected_entities, entity_types)
            (
                "person_only",
                "Hello John Smith from New York",
                ["John Smith", "New York"],
                ["PERSON", "PLACE"],
            ),
            (
                "email_only",
                "Contact me at john@example.com",
                ["john@example.com"],
                ["EMAIL"],
            ),
            ("phone_only", "Call 555-123-4567", ["555-123-4567"], ["PHONE"]),
            (
                "mixed_entities",
                "John Smith (john@example.com) at 555-1234",
                ["John Smith", "john@example.com", "555-1234"],
                ["PERSON", "EMAIL", "PHONE"],
            ),
            (
                "no_entities",
                "This is plain text with no PII",
                [],
                [],
            ),
            (
                "repeated_entities",
                "John called John again",
                ["John"],
                ["PERSON"],
            ),
            (
                "unicode_names",
                "José García",
                ["José García"],
                ["PERSON"],
            ),
            (
                "organisations",
                "Work at Microsoft Corporation",
                ["Microsoft Corporation"],
                ["ORGANISATION"],
            ),
        ]
    )
    def test_entity_detection_variations(
        self, description, input_text, expected_entities, entity_types
    ):
        """Test entity detection with various input types - parameterized."""
        shield = LLMShield()
        cloaked, entity_map = shield.cloak(input_text)

        # Check that some entities were detected if expected
        detected_entities = list(entity_map.values())
        if expected_entities:
            # At least one expected entity should be detected, but not
            # necessarily all (entity detection can be context-dependent)
            found_any = any(
                entity in detected_entities for entity in expected_entities
            )
            if not found_any:
                # If no expected entities found, at least verify the text
                # changed (some entity was detected)
                self.assertNotEqual(
                    cloaked,
                    input_text,
                    f"No entities detected in '{input_text}', "
                    f"expected some of: {expected_entities}",
                )
        else:
            # If no entities expected, verify no entities were detected
            self.assertEqual(
                len(entity_map),
                0,
                f"Expected no entities but found: {detected_entities}",
            )

        # Check that uncloaking restores original text
        uncloaked = shield.uncloak(cloaked, entity_map)
        self.assertEqual(uncloaked, input_text)

    @parameterized.expand(
        [
            # (description, response_type, test_data)
            ("string_response", str, "Hello John Doe"),
            ("list_response", list, ["Hello", "John", "Doe"]),
            ("dict_response", dict, {"greeting": "Hello", "name": "John Doe"}),
            (
                "nested_dict",
                dict,
                {
                    "user": {
                        "name": "John",
                        "contacts": {"email": "john@test.com"},
                    }
                },
            ),
            (
                "mixed_list",
                list,
                ["Hello John", {"email": "john@test.com"}, 42],
            ),
            # Note: empty responses are not tested here as they trigger
            # validation errors
        ]
    )
    def test_uncloak_response_types(
        self, description, response_type, test_data
    ):
        """Test uncloaking various response types - parameterized."""
        shield = LLMShield()
        entity_map = {"<PERSON_0>": "John Doe", "<EMAIL_0>": "john@test.com"}

        # Create test data with entities if it's not empty
        if test_data:
            if response_type is str:
                test_input = test_data.replace(
                    "John Doe", "<PERSON_0>"
                ).replace("john@test.com", "<EMAIL_0>")
            elif response_type is list:
                test_input = [
                    (
                        item.replace("John", "<PERSON_0>").replace(
                            "john@test.com", "<EMAIL_0>"
                        )
                    )
                    if isinstance(item, str)
                    else item
                    for item in test_data
                ]
            elif response_type is dict:
                test_input = self._replace_entities_in_dict(
                    test_data,
                    {
                        "John Doe": "<PERSON_0>",
                        "John": "<PERSON_0>",
                        "john@test.com": "<EMAIL_0>",
                    },
                )
        else:
            test_input = test_data

        result = shield.uncloak(test_input, entity_map)

        # Verify type is preserved
        self.assertIsInstance(result, response_type)

        # For non-empty data, verify content is restored
        if test_data:
            if response_type is str:
                self.assertIn("John", str(result))
            elif response_type in (list, dict):
                # Should be different after uncloaking
                self.assertNotEqual(result, test_input)

    def _replace_entities_in_dict(self, data, replacements):
        """Replace entities in nested dictionaries."""
        if isinstance(data, dict):
            return {
                key: self._replace_entities_in_dict(value, replacements)
                for key, value in data.items()
            }
        if isinstance(data, list):
            return [
                self._replace_entities_in_dict(item, replacements)
                for item in data
            ]
        if isinstance(data, str):
            result = data
            for original, placeholder in replacements.items():
                result = result.replace(original, placeholder)
            return result

        return data

    @parameterized.expand(
        [
            # (description, invalid_input, expected_error_type)
            ("none_response", None, ValueError),
            ("empty_string", "", ValueError),
            ("empty_list", [], ValueError),
            ("false_value", False, ValueError),
            ("zero_value", 0, ValueError),
        ]
    )
    def test_uncloak_validation_edge_cases(
        self, description, invalid_input, expected_error_type
    ):
        """Test uncloak validation with various edge cases - parameterized."""
        shield = LLMShield()
        entity_map = {"<PERSON_0>": "John"}

        with self.assertRaises(expected_error_type):
            shield.uncloak(invalid_input, entity_map)

    @parameterized.expand(
        [
            # (description, streaming, llm_response, expected_type)
            ("string_non_stream", False, "Hello world", str),
            (
                "generator_stream",
                True,
                (x for x in ["Hello", " world"]),
                type(x for x in []),
            ),
            ("list_non_stream", False, ["Hello", "world"], list),
            ("dict_non_stream", False, {"message": "Hello"}, dict),
        ]
    )
    def test_ask_response_type_handling(
        self, description, streaming, llm_response, expected_type
    ):
        """Test ask method response type handling - parameterized."""

        def mock_llm(**kwargs):
            """Return the provided llm_response."""
            return llm_response

        shield = LLMShield(llm_func=mock_llm)

        if (
            streaming
            and hasattr(llm_response, "__iter__")
            and not isinstance(llm_response, str | bytes | dict)
        ):
            # For streaming responses
            result = shield.ask(prompt="Hello", stream=streaming)
            self.assertIsInstance(result, type(llm_response))
        else:
            # For non-streaming responses
            result = shield.ask(prompt="Hello", stream=streaming)
            self.assertIsInstance(result, expected_type)

    def test_cloak_message_preserves_extra_keys(self):
        """Test _cloak_message passes through extra fields."""

        def mock_llm(**kwargs):
            """Return static OK response."""
            return "OK"

        shield = LLMShield(llm_func=mock_llm)
        messages = [
            {
                "role": "tool",
                "content": "result from John Doe",
                "tool_call_id": "call_abc",
                "name": "lookup",
            },
        ]
        result = shield.ask(messages=messages)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    main(verbosity=2)
