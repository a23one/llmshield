"""List of regex-based matchers used to detect entities.

Description:
    This module contains compiled regular expression patterns for detecting
    various types of sensitive information including email addresses, credit
    card numbers, IP addresses, URLs, and phone numbers. Each pattern is
    carefully crafted to balance accuracy and performance.

Patterns:
    EMAIL_ADDRESS_PATTERN: Matches email addresses
    CREDIT_CARD_PATTERN: Matches credit card numbers (various formats)
    IP_ADDRESS_PATTERN: Matches IPv4 addresses
    URL_PATTERN: Matches HTTP/HTTPS URLs
    PHONE_NUMBER_PATTERN: Matches phone numbers (US and international)

Author:
    LLMShield by brainpolo, 2025-2026
"""

import re

# * EMAIL
# --------------------------------------------------------------------------
# Matches email addresses like:
# - john.doe@example.com
# - user+tag@domain.co.uk
# - first.last@subdomain.company.org
EMAIL_ADDRESS_PATTERN = re.compile(
    r"\b[a-zA-Z0-9]"  # Start with alphanumeric (single-char local is valid)
    r"(?:"  # Optional rest of local part:
    r"(?:[a-zA-Z0-9\+]"  # middle chars (alnum or +)
    r"|[._-](?![._-]))*"  # or single ./_/- (no consecutive separators)
    r"[a-zA-Z0-9]"  # ending in alphanumeric
    r")?"
    r"@"
    r"(?:[a-zA-Z0-9][a-zA-Z0-9-]*\.)+"  # Domain parts
    r"[a-zA-Z]{2,}\b",  # TLD
)

# * CREDIT CARD
# --------------------------------------------------------------------------
# Matches credit card numbers like:
# - 4532015112345678    (Visa)
# - 4532 0151 1234 5678 (Visa with spaces)
# - 5425-2334-5678-8790 (Mastercard with dashes)
# - 347352358990016     (American Express)
# - 3530111333300000    (JCB)
# - 6011000990139424    (Discover)
CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:4[0-9]{15}"  # Visa 16 digits
    # Mastercard
    r"|(?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}"
    r"|27[01][0-9]|2720)[0-9]{12}"
    r"|3[47][0-9]{13}"  # American Express
    r"|3(?:0[0-5]|[68][0-9])[0-9]{11}"  # Diners Club
    r"|6(?:011|5[0-9]{2})[0-9]{12}"  # Discover
    r"|(?:2131|1800|35\d{3})\d{11}"  # JCB
    r")\b",
)

# * IP ADDRESS
# --------------------------------------------------------------------------
# Matches IPv4 addresses like:
# - 192.168.1.1
# - 10.0.0.0
# - 172.16.254.1
# - 256.1.2.3 (invalid, won't match)
IP_ADDRESS_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\b",
)

# * URL
# --------------------------------------------------------------------------
# Matches URLs like:
# - https://example.com
# - http://subdomain.example.com/path
# - https://my-site.org/path?query=value
# - http://domain.anything/path#fragment
URL_PATTERN = re.compile(
    r"(?:https?://)"  # Protocol (required)
    r"(?:[\w-]+\.)*[\w-]+\."  # Domain name
    r"[\w-]+"  # TLD
    r"(?:/[^\s]*)?",  # Path (optional)
)

# * PHONE NUMBER
# --------------------------------------------------------------------------
# Matches various phone number formats including:
# - US styles like "123-456-7890", "(123) 456-7890"
# - International numbers like "+44 (123) 456-7890", "+44 84491234567"
PHONE_NUMBER_PATTERN = re.compile(
    r"(?<!\d)"  # Negative lookbehind: ensure no digit immediately precedes.
    r"("  # Capture the entire phone number.
    r"(?:(?:\+?\d{1,3}[-.\s]*)?"  # Optional country code with '+'
    r"(?:\(?\d{3}\)?[-.\s]*\d{3}[-.\s]*\d{4}))"  # US/strict format: 3-3-4
    # (area code may be parenthesized).
    r"|"  # OR
    r"(?:\+\d{1,3}[-.\s]*(?:\d[-.\s]*){7,14}\d)"  # International format
    # and 8 to 15 digits.
    r")"
    r"(?!\d)",  # Negative lookahead: ensure no digit immediately follows.
)
