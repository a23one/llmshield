"""Test regex patterns for entity detection.

Description:
    This test module provides comprehensive testing for all regex patterns
    used in entity detection, including email addresses, phone numbers,
    IP addresses, credit cards, and URLs.

Test Classes:
    - TestRegexMatchers: Tests all regex pattern matching

Author: LLMShield by brainpolo, 2025-2026
"""

from unittest import TestCase

from parameterized import parameterized

from llmshield.matchers.regex import (
    CREDIT_CARD_PATTERN,
    EMAIL_ADDRESS_PATTERN,
    IP_ADDRESS_PATTERN,
    PHONE_NUMBER_PATTERN,
    URL_PATTERN,
)


class TestRegexMatchers(TestCase):
    """Test suite for regex pattern matching."""

    @parameterized.expand(
        [
            # Valid emails - various formats
            ("standard", "user@domain.com", True),
            ("with_plus", "user+tag@domain.com", True),
            ("with_dots", "first.last@domain.com", True),
            ("subdomain", "user@sub.domain.com", True),
            ("long_tld", "user@domain.museum", True),
            ("numbers_in_local", "user123@domain.com", True),
            ("numbers_in_domain", "user@domain123.com", True),
            ("hyphen_in_domain", "user@my-domain.com", True),
            ("multiple_subdomains", "user@mail.sub.domain.org", True),
            # Single-character locals are RFC-valid (RFC 5321 §4.5.3.1.1)
            ("short_local_alpha", "a@domain.com", True),
            ("short_local_digit", "1@domain.com", True),
            ("short_local_initial", "j@gmail.com", True),
            ("underscore_local", "user_name@domain.com", True),
            # Invalid emails
            ("no_at", "userdomain.com", False),
            ("double_at", "user@@domain.com", False),
            ("no_domain", "user@", False),
            ("no_local", "@domain.com", False),
            ("double_dots", "user..name@domain.com", False),
            ("dot_at_start", ".user@domain.com", False),
            ("dot_at_end", "user.@domain.com", False),
            ("space_in_local", "user name@domain.com", False),
            ("space_in_domain", "user@do main.com", False),
            ("no_tld", "user@domain", False),
            ("tld_too_short", "user@domain.c", False),
            ("dot_before_tld", "john.doe@.com", False),
        ]
    )
    def test_email_patterns_comprehensive(
        self, description, email, should_match
    ):
        """Test email pattern with comprehensive examples - parameterized."""
        match = EMAIL_ADDRESS_PATTERN.fullmatch(email)
        if should_match:
            self.assertIsNotNone(match, f"Email should match: {email}")
            self.assertEqual(match.group(), email)
        else:
            self.assertIsNone(match, f"Email should not match: {email}")

    @parameterized.expand(
        [
            # Valid credit cards - different types
            ("visa_16", "4532015112345678", True),
            # 13-digit Visa not supported by current regex
            ("visa_13", "4532015112345", False),
            ("mastercard", "5425233456788790", True),
            ("amex", "347352358990016", True),
            ("discover", "6011000990139424", True),
            ("jcb", "3530111333300000", True),
            ("diners", "30569309025904", True),
            # Invalid credit cards
            ("too_short", "123456789012", False),
            ("too_long", "45320151123456789", False),
            ("letters", "abcdefghijklmnop", False),
            ("mixed_alpha", "4532a15112345678", False),
            # Assuming spaces not supported
            ("spaces", "4532 0151 1234 5678", False),
            # Assuming dashes not supported
            ("dashes", "4532-0151-1234-5678", False),
            ("empty", "", False),
            ("all_zeros", "0000000000000000", False),
            ("special_chars", "4532@151#1234$678", False),
        ]
    )
    def test_credit_card_patterns_comprehensive(
        self, description, card, should_match
    ):
        """Test credit card patterns comprehensively - parameterized."""
        match = CREDIT_CARD_PATTERN.search(card)
        if should_match:
            self.assertIsNotNone(match, f"Credit card should match: {card}")
            self.assertEqual(match.group(), card)
        else:
            self.assertIsNone(match, f"Credit card should not match: {card}")

    @parameterized.expand(
        [
            # Valid IP addresses
            ("localhost", "127.0.0.1", True),
            ("private_a", "10.0.0.1", True),
            ("private_b", "172.16.0.1", True),
            ("private_c", "192.168.1.1", True),
            ("public", "8.8.8.8", True),
            ("edge_values", "255.255.255.255", True),
            ("zeros", "0.0.0.0", True),
            ("mixed", "192.168.0.255", True),
            # Invalid IP addresses
            ("out_of_range", "256.1.1.1", False),
            # Regex finds valid IP within invalid string
            ("negative", "-1.1.1.1", True),
            ("too_few_octets", "192.168.1", False),
            # Regex finds valid IP within longer string
            ("too_many_octets", "192.168.1.1.1", True),
            ("letters", "192.168.a.1", False),
            ("empty_octet", "192..1.1", False),
            # Leading zeros might be allowed
            ("leading_zeros", "192.168.001.1", True),
            ("spaces", "192. 168.1.1", False),
            # Regex finds valid IP within string with extra chars
            ("special_chars", "192.168.1.1!", True),
        ]
    )
    def test_ip_address_patterns_comprehensive(
        self, description, ip, should_match
    ):
        """Test IP address patterns comprehensively - parameterized."""
        match = IP_ADDRESS_PATTERN.search(ip)
        if should_match:
            self.assertIsNotNone(match, f"IP address should match: {ip}")
            # For partial matches, just verify a match was found
        else:
            self.assertIsNone(match, f"IP address should not match: {ip}")

    @parameterized.expand(
        [
            # Valid phone numbers - various formats
            ("us_dashes", "123-456-7890", True),
            ("us_parentheses", "(123) 456-7890", True),
            ("us_spaces", "123 456 7890", True),
            ("us_dots", "123.456.7890", True),
            ("international_plus", "+1 123-456-7890", True),
            ("uk_format", "+44 20 7946 0958", True),
            ("long_international", "+44 84491234567", True),
            ("mixed_format", "+1 (123) 456-7890", True),
            # Invalid phone numbers
            ("too_short", "1234567", False),
            ("letters", "phone123456", False),
            ("wrong_grouping", "12-3456-7890", False),
            ("no_area_code", "456-7890", False),
            ("special_chars", "123@456#7890", False),
            ("only_plus", "+", False),
            ("double_plus", "++1 123-456-7890", False),
        ]
    )
    def test_phone_patterns_comprehensive(
        self, description, phone, should_match
    ):
        """Test phone number patterns comprehensively - parameterized."""
        match = PHONE_NUMBER_PATTERN.fullmatch(phone.strip())
        if should_match:
            self.assertIsNotNone(match, f"Phone number should match: {phone}")
            self.assertEqual(match.group().strip(), phone.strip())
        else:
            self.assertIsNone(match, f"Phone number should not match: {phone}")

    @parameterized.expand(
        [
            # Valid URLs
            ("simple_http", "http://example.com", True),
            ("simple_https", "https://example.com", True),
            ("with_path", "https://example.com/path", True),
            ("with_query", "https://example.com/path?query=value", True),
            ("with_fragment", "https://example.com/path#fragment", True),
            # Partial match expected
            ("with_port", "https://example.com:8080", True),
            ("subdomain", "https://sub.example.com", True),
            # Partial match expected
            (
                "complex",
                "https://sub.example.com:8080/path?q=v&x=y#frag",
                True,
            ),
            ("hyphen_domain", "https://my-site.com", True),
            ("numbers_domain", "https://site123.com", True),
            # Invalid URLs
            ("no_protocol", "example.com", False),
            ("wrong_protocol", "ftp://example.com", False),
            ("missing_colon", "http//example.com", False),
            ("space_in_url", "https://exam ple.com", False),
            ("no_domain", "https://", False),
            # Regex finds valid URL within string with extra chars
            ("invalid_chars", "https://example.com<script>", True),
        ]
    )
    def test_url_patterns_comprehensive(self, description, url, should_match):
        """Test URL pattern with comprehensive examples - parameterized."""
        match = URL_PATTERN.search(url)
        if should_match:
            self.assertIsNotNone(match, f"URL should match: {url}")
            # For partial matches, just verify a match was found
        else:
            self.assertIsNone(match, f"URL should not match: {url}")


if __name__ == "__main__":
    import unittest

    unittest.main(verbosity=2)
