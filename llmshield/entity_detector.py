"""Entity detection and classification module.

Description:
    This module implements comprehensive entity detection algorithms to
    identify personally identifiable information (PII) and sensitive data in
    text. It uses a multi-layered approach combining regex patterns,
    dictionary lookups, and contextual analysis to accurately detect various
    entity types.

Classes:
    EntityDetector: Main class for detecting entities in text
    Entity: Data class representing a detected entity
    EntityType: Enumeration of supported entity types
    EntityGroup: Grouping of entity types into categories
    EntityConfig: Configuration for selective entity detection

Detection Methods:
    - Regex patterns for structured data (emails, URLs, phone numbers)
    - Dictionary lookups for known entities (cities, countries, organisations)
    - Contextual analysis for proper nouns and person names
    - Heuristic rules for complex entity patterns

Author:
    LLMShield by brainpolo, 2025-2026
"""

# Standard Library Imports
from dataclasses import dataclass
from enum import StrEnum

# Local imports
from llmshield.cache.entity_cache import get_entity_cache
from llmshield.matchers.functions import _luhn_check
from llmshield.matchers.lists import (
    EN_ORG_COMPONENTS,
    EN_PERSON_INITIALS,
    EN_PLACE_COMPONENTS,
    EN_PUNCTUATION,
)
from llmshield.matchers.regex import (
    CREDIT_CARD_PATTERN,
    EMAIL_ADDRESS_PATTERN,
    IP_ADDRESS_PATTERN,
    PHONE_NUMBER_PATTERN,
    URL_PATTERN,
)
from llmshield.utils import normalise_spaces, split_fragments

ENT_REPLACEMENT = "\n"  # Use to void overlap with another entity

SPACE = " "


class EntityType(StrEnum):
    """Primary classification of entity types."""

    # Proper Nouns
    PERSON = "PERSON"
    ORGANISATION = "ORGANISATION"  # British spelling maintained
    PLACE = "PLACE"
    CONCEPT = "CONCEPT"

    # Numbers
    PHONE = "PHONE"  # Simplified name
    CREDIT_CARD = "CREDIT_CARD"

    # Locators
    EMAIL = "EMAIL"
    URL = "URL"
    IP_ADDRESS = "IP_ADDRESS"

    @classmethod
    def all(cls) -> frozenset["EntityType"]:
        """Return frozenset of all entity types."""
        return frozenset(cls)

    @classmethod
    def default_types(cls) -> frozenset["EntityType"]:
        """Return default entity types for PII detection."""
        return frozenset(
            [
                cls.PERSON,
                cls.ORGANISATION,
                cls.PLACE,
                cls.EMAIL,
                cls.PHONE,
                cls.CREDIT_CARD,
                cls.URL,
                cls.IP_ADDRESS,
            ]
        )

    @classmethod
    def locators(cls) -> frozenset["EntityType"]:
        """Return entity types that are location-based identifiers."""
        return frozenset([cls.EMAIL, cls.IP_ADDRESS, cls.URL])

    @classmethod
    def numbers(cls) -> frozenset["EntityType"]:
        """Return entity types that are numeric identifiers."""
        return frozenset([cls.PHONE, cls.CREDIT_CARD])

    @classmethod
    def proper_nouns(cls) -> frozenset["EntityType"]:
        """Return entity types that are proper nouns."""
        return frozenset(
            [cls.PERSON, cls.PLACE, cls.ORGANISATION, cls.CONCEPT]
        )

    # Legacy compatibility
    PHONE_NUMBER = "PHONE"  # Alias for backward compatibility


class EntityGroup(StrEnum):
    """Groups of related entity types."""

    PNOUN = "PNOUN"
    NUMBER = "NUMBER"
    LOCATOR = "LOCATOR"

    def get_types(self) -> set[EntityType]:
        """Get all entity types belonging to this group."""
        group_map = {
            self.PNOUN: {
                EntityType.PERSON,
                EntityType.ORGANISATION,
                EntityType.PLACE,
                EntityType.CONCEPT,
            },
            self.NUMBER: {EntityType.PHONE, EntityType.CREDIT_CARD},
            self.LOCATOR: {
                EntityType.EMAIL,
                EntityType.URL,
                EntityType.IP_ADDRESS,
            },
        }
        return group_map[self]


@dataclass(frozen=True)
class Entity:
    """Represents a detected entity in text."""

    type: EntityType
    value: str

    @property
    def group(self) -> EntityGroup:
        """Get the group this entity belongs to."""
        for group in EntityGroup:
            if self.type in group.get_types():
                return group
        msg = f"Unknown entity type: {self.type}"
        raise ValueError(msg)


class EntityConfig:
    """Configuration for selective entity detection and cloaking."""

    def __init__(self, enabled_types: frozenset[EntityType] | None = None):
        """Initialize entity configuration.

        Args:
            enabled_types: Set of entity types to detect and cloak.
                          If None, default PII types are enabled.

        """
        self.enabled_types = (
            enabled_types
            if enabled_types is not None
            else EntityType.default_types()
        )

    def is_enabled(self, entity_type: EntityType) -> bool:
        """Check if an entity type is enabled for detection."""
        return entity_type in self.enabled_types

    def with_disabled(self, *disabled_types: EntityType) -> "EntityConfig":
        """Create new config with specified types disabled."""
        new_enabled = self.enabled_types - frozenset(disabled_types)
        return EntityConfig(new_enabled)

    def with_enabled(  # skipcq: PYL-R0201
        self, *enabled_types: EntityType
    ) -> "EntityConfig":
        """Create new config with only specified types enabled."""
        return EntityConfig(frozenset(enabled_types))

    @classmethod
    def disable_locations(cls) -> "EntityConfig":
        """Create config with location-based entities disabled."""
        return cls().with_disabled(
            EntityType.PLACE, EntityType.IP_ADDRESS, EntityType.URL
        )

    @classmethod
    def disable_persons(cls) -> "EntityConfig":
        """Create config with person entities disabled."""
        return cls().with_disabled(EntityType.PERSON)

    @classmethod
    def disable_contacts(cls) -> "EntityConfig":
        """Create config with contact information disabled."""
        return cls().with_disabled(EntityType.EMAIL, EntityType.PHONE)

    @classmethod
    def only_financial(cls) -> "EntityConfig":
        """Create config with only financial entities enabled."""
        return cls().with_enabled(EntityType.CREDIT_CARD)

    def without_locations(self) -> "EntityConfig":
        """Disable location-based entities."""
        return self.with_disabled(
            EntityType.PLACE, EntityType.IP_ADDRESS, EntityType.URL
        )

    def without_persons(self) -> "EntityConfig":
        """Disable person entities."""
        return self.with_disabled(EntityType.PERSON)

    def without_contacts(self) -> "EntityConfig":
        """Disable contact information."""
        return self.with_disabled(EntityType.EMAIL, EntityType.PHONE)

    def without_organisations(self) -> "EntityConfig":
        """Disable organisation entities."""
        return self.with_disabled(EntityType.ORGANISATION)

    def without_concepts(self) -> "EntityConfig":
        """Disable concept entities."""
        return self.with_disabled(EntityType.CONCEPT)

    def without_credit_cards(self) -> "EntityConfig":
        """Disable credit card entities."""
        return self.with_disabled(EntityType.CREDIT_CARD)

    def without_emails(self) -> "EntityConfig":
        """Disable email entities."""
        return self.with_disabled(EntityType.EMAIL)

    def without_phones(self) -> "EntityConfig":
        """Disable phone entities."""
        return self.with_disabled(EntityType.PHONE)

    def without_urls(self) -> "EntityConfig":
        """Disable URL entities."""
        return self.with_disabled(EntityType.URL)

    def without_ips(self) -> "EntityConfig":
        """Disable IP address entities."""
        return self.with_disabled(EntityType.IP_ADDRESS)

    def without_places(self) -> "EntityConfig":
        """Disable place entities."""
        return self.with_disabled(EntityType.PLACE)

    def with_all_enabled(self) -> "EntityConfig":  # skipcq: PYL-R0201
        """Enable all entity types."""
        return EntityConfig(EntityType.all())

    def with_only_financial(self) -> "EntityConfig":  # skipcq: PYL-R0201
        """Enable only financial detection."""
        return EntityConfig(frozenset([EntityType.CREDIT_CARD]))

    @classmethod
    def enable_all(cls) -> "EntityConfig":
        """Create config with all entity types enabled."""
        return cls(EntityType.all())


class EntityDetector:
    """Main entity detection system using rule-based and pattern approaches.

    Identifies sensitive information in text using a waterfall approach where
    each detection method is tried in order, and the text is reduced as each
    entity is found. This eliminates potential overlapping entities and
    improves detection accuracy.
    """

    def __init__(self, config: EntityConfig | None = None) -> None:
        """Initialize entity detector with optional selective configuration.

        Args:
            config: Configuration for selective entity detection.
                   If None, all entity types are enabled.

        """
        self.config = config or EntityConfig()
        self.cache = get_entity_cache()

        # Utility functions
        self.split_fragments = split_fragments
        self.normalise_spaces = normalise_spaces

        # Static data (lightweight references)
        self.en_person_initials = EN_PERSON_INITIALS
        self.en_org_components = EN_ORG_COMPONENTS
        self.en_place_components = EN_PLACE_COMPONENTS
        self.en_punctuation = EN_PUNCTUATION

        # Compiled regex patterns (shared across instances)
        self.email_pattern = EMAIL_ADDRESS_PATTERN
        self.credit_card_pattern = CREDIT_CARD_PATTERN
        self.ip_address_pattern = IP_ADDRESS_PATTERN
        self.url_pattern = URL_PATTERN
        self.phone_number_pattern = PHONE_NUMBER_PATTERN

        # Utility functions
        self.luhn_check = _luhn_check

    def detect_entities(self, text: str) -> set[Entity]:
        """Detect entities using waterfall methodology with filtering."""
        detection_methods = [
            (self._detect_locators, EntityGroup.LOCATOR),
            (self._detect_numbers, EntityGroup.NUMBER),
            (self._detect_proper_nouns, EntityGroup.PNOUN),
        ]

        entities: set[Entity] = set()
        working_text: str = text

        for method, group in detection_methods:
            # Skip entire groups if no types are enabled
            group_types = group.get_types()
            if not any(self.config.is_enabled(t) for t in group_types):
                continue

            new_entities, working_text = method(working_text)

            # Filter entities based on configuration
            filtered_entities = {
                entity
                for entity in new_entities
                if self.config.is_enabled(entity.type)
            }
            entities.update(filtered_entities)

        return entities

    def _detect_proper_nouns(self, text: str) -> tuple[set[Entity], str]:
        """Umbrella method for proper noun detection.

        It collects candidate proper nouns from the text, then classifies each.
        If an entity is classified (and potentially cleaned), it is added as
        an Entity.
        """
        entities = set()
        reduced_text = text

        # Step 1: Collect sequential proper nouns.
        sequential_pnouns = self._collect_proper_nouns(text)

        # Step 2: Process each proper noun.
        for p_noun in sequential_pnouns:
            if not p_noun or p_noun not in reduced_text:
                continue

            result = self._classify_proper_noun(p_noun)
            if result is None:
                continue

            cleaned_value, entity_type = result
            entities.add(Entity(type=entity_type, value=cleaned_value))
            reduced_text = reduced_text.replace(p_noun, ENT_REPLACEMENT)

        return entities, reduced_text

    def _collect_proper_nouns(self, text: str) -> list[str]:
        """Collect sequential proper nouns from text."""
        sequential_pnouns = []
        normalised_text = self.normalise_spaces(text)
        fragments = self.split_fragments(normalised_text)

        for fragment in fragments:
            fragment_nouns = self._process_fragment(fragment)
            sequential_pnouns.extend(fragment_nouns)

        return sorted(
            [p for p in sequential_pnouns if p], key=len, reverse=True
        )

    def _process_fragment(self, fragment: str) -> list[str]:
        """Process a single text fragment to extract proper nouns."""
        # Split on common contractions first
        for split_word in ["I'm", "I've", "I'll"]:
            if split_word in fragment:
                fragment = fragment.replace(split_word, f"{split_word} ")

        fragment_words = fragment.split(SPACE)
        sequential_pnouns: list[str] = []
        pending_p_noun = ""

        for word in fragment_words:
            if not word:
                continue

            # Handle personal pronouns and contractions
            if EntityDetector._should_skip_pronoun(
                word, pending_p_noun, sequential_pnouns
            ):
                pending_p_noun = ""
                continue

            # Process potential proper noun
            pending_p_noun = self._process_word(
                word, pending_p_noun, sequential_pnouns
            )

        if pending_p_noun:
            sequential_pnouns.append(pending_p_noun.strip())

        return sequential_pnouns

    @staticmethod
    def _should_skip_pronoun(
        word: str, pending_p_noun: str, sequential_pnouns: list[str]
    ) -> bool:
        """Check if word is a pronoun that should be skipped."""
        if word in {"I'm", "I've", "I'll", "I"}:
            if pending_p_noun:
                sequential_pnouns.append(pending_p_noun.strip())
            return True
        return False

    def _process_word(
        self, word: str, pending_p_noun: str, sequential_pnouns: list[str]
    ) -> str:
        """Process a word for proper noun detection."""
        normalized_word = word.strip(".,!?;:")
        is_honorific = normalized_word in self.en_person_initials
        is_capitalised = word and word[0].isupper()

        if is_honorific or (
            not any(c in word for c in self.en_punctuation if c != ".")
            and is_capitalised
        ):
            return pending_p_noun + SPACE + word if pending_p_noun else word
        if pending_p_noun:
            sequential_pnouns.append(pending_p_noun.strip())
            return ""
        return pending_p_noun

    def _clean_person_name(self, p_noun: str) -> str:
        """Remove a leading honorific (if any) from a person proper noun.

        For example "Dr. John Doe" becomes "John Doe".
        """
        words = p_noun.split()
        if not words:
            return p_noun
        first_word_norm = words[0].strip(".,!?;:")
        if first_word_norm in self.en_person_initials and len(words) > 1:
            return " ".join(words[1:]).strip()
        return p_noun

    def _classify_proper_noun(
        self, p_noun: str
    ) -> tuple[str, EntityType] | None:
        """Classify a proper noun into its entity type and clean it.

        Returns tuple (modified_value, EntityType) or None if no
        classification.
        In the PERSON case, if a honorific is detected at the start, it is
        removed.
        All punctuation is stripped from the final value.
        """
        if not p_noun:
            return None

        def clean_value(value: str) -> str:
            """Remove all punctuation from the value."""
            return "".join(
                c for c in value if c not in self.en_punctuation
            ).strip()

        # 1. Check for organisations first.
        if self._is_organisation(p_noun):
            return (clean_value(p_noun), EntityType.ORGANISATION)

        # 2. Check for places.
        if self._is_place(p_noun):
            return (clean_value(p_noun), EntityType.PLACE)

        # 3. Check for concepts (all uppercase, single word).
        # Should be checked before persons.
        if self._is_concept(p_noun):
            return (clean_value(p_noun), EntityType.CONCEPT)

        # 4. Check for persons.
        if self._is_person(p_noun):
            cleaned = self._clean_person_name(p_noun)
            final_value = clean_value(cleaned)
            return (final_value, EntityType.PERSON)

        # 5. Default to None.
        return None

    def _is_concept(self, p_noun: str) -> bool:
        """Check if proper noun is a concept."""
        return (
            all(word.isupper() for word in p_noun.split())
            and len(p_noun.split()) == 1
            and not any(c in p_noun for c in self.en_punctuation)
            and not self.cache.is_english_word(p_noun.lower())
        )

    def _is_organisation(self, p_noun: str) -> bool:
        """Check if proper noun is an organisation.

        Uses the organisations.txt dictionary.
        """
        p_noun_lower = p_noun.lower()
        return self.cache.is_organisation(p_noun_lower)

    def _is_place(self, p_noun: str) -> bool:
        """Check if proper noun is a place."""
        p_noun_lower = p_noun.lower()
        return self.cache.is_place(p_noun_lower) or any(
            comp in p_noun.split() for comp in self.en_place_components
        )

    def _is_person(self, p_noun: str) -> bool:
        """Check if proper noun is a person."""
        words = p_noun.split()

        # Handle possessives
        words = [w.removesuffix("'s") for w in words]

        # Must have at least one word after cleaning
        if not words:
            return False

        # Skip honorifics at start
        if words[0].strip(".,!?;:") in self.en_person_initials:
            words = words[1:]

        # Must have remaining words after removing honorifics
        if not words:
            return False

        # Check each word
        for word in words:
            clean_word = word.strip(".,!?;:")

            # Skip empty words
            if not clean_word:
                continue

            # Allow hyphenated names
            if "-" in clean_word:
                parts = clean_word.split("-")
                if not all(part and part[0].isupper() for part in parts):
                    return False
                continue

            # Each word must:
            # 1. Start with capital letter
            # 2. Not be in common words
            # 3. Not contain digits
            if (
                not clean_word[0].isupper()
                or self.cache.is_english_word(clean_word.lower())
                or any(c.isdigit() for c in clean_word)
            ):
                return False

        return True

    def _detect_numbers(self, text: str) -> tuple[set[Entity], str]:
        """Detect numbers in the text."""
        entities = set()
        reduced_text = text

        # * 1. Split on sentence boundaries (punctuation / new line)
        emails = self.email_pattern.finditer(text)
        for email in emails:
            entities.add(
                Entity(
                    type=EntityType.EMAIL,
                    value=email.group(),
                ),
            )
            reduced_text = reduced_text.replace(email.group(), ENT_REPLACEMENT)

        # * 2. Detect credit cards
        credit_cards = self.credit_card_pattern.finditer(text)
        for credit_card in credit_cards:
            if self.luhn_check(credit_card.group()):
                entities.add(
                    Entity(
                        type=EntityType.CREDIT_CARD,
                        value=credit_card.group(),
                    ),
                )
                reduced_text = reduced_text.replace(
                    credit_card.group(), ENT_REPLACEMENT
                )

        # * 3. Detect phone numbers
        phone_numbers = self.phone_number_pattern.finditer(text)
        for phone_number in phone_numbers:
            entities.add(
                Entity(
                    type=EntityType.PHONE,
                    value=phone_number.group(),
                ),
            )
            reduced_text = reduced_text.replace(
                phone_number.group(), ENT_REPLACEMENT
            )

        # * 4. Return the reduced text and entities found
        return entities, reduced_text

    def _detect_locators(self, text: str) -> tuple[set[Entity], str]:
        """Detect locators in the text."""
        entities = set()
        reduced_text = text

        # * 1. Detect URLs
        urls = self.url_pattern.finditer(text)
        for url in urls:
            entities.add(
                Entity(
                    type=EntityType.URL,
                    value=url.group(),
                ),
            )
            reduced_text = reduced_text.replace(url.group(), ENT_REPLACEMENT)

        # * 2. Detect emails
        emails = self.email_pattern.finditer(text)
        for email in emails:
            entities.add(
                Entity(
                    type=EntityType.EMAIL,
                    value=email.group(),
                ),
            )
            reduced_text = reduced_text.replace(email.group(), ENT_REPLACEMENT)

        # * 3. Detect IP addresses
        ip_addresses = self.ip_address_pattern.finditer(text)
        for ip_address in ip_addresses:
            entities.add(
                Entity(
                    type=EntityType.IP_ADDRESS,
                    value=ip_address.group(),
                ),
            )
            reduced_text = reduced_text.replace(
                ip_address.group(), ENT_REPLACEMENT
            )

        # * 4. Return the reduced text and entities found
        return entities, reduced_text
