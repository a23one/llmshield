"""Test entity detection and classification functionality.

Description:
    Comprehensive tests for the EntityDetector class, covering entity
    types, grouping, proper noun collection, person/organisation/place/
    concept detection, number and locator detection, entity configuration,
    and the full detection pipeline.

Author: LLMShield by brainpolo, 2025-2026
"""

import unittest

from parameterized import parameterized

from llmshield.entity_detector import (
    Entity,
    EntityConfig,
    EntityDetector,
    EntityGroup,
    EntityType,
)


class TestEntityTypeAndGroup(unittest.TestCase):
    """Test EntityType, EntityGroup, and Entity model classes."""

    def test_entity_group_types(self):
        """Test EntityGroup.get_types() returns correct sets."""
        self.assertEqual(
            EntityGroup.PNOUN.get_types(),
            {
                EntityType.PERSON,
                EntityType.ORGANISATION,
                EntityType.PLACE,
                EntityType.CONCEPT,
            },
        )
        self.assertEqual(
            EntityGroup.NUMBER.get_types(),
            {EntityType.PHONE, EntityType.CREDIT_CARD},
        )
        self.assertEqual(
            EntityGroup.LOCATOR.get_types(),
            {
                EntityType.EMAIL,
                EntityType.URL,
                EntityType.IP_ADDRESS,
            },
        )

    @parameterized.expand(
        [
            (
                "locators",
                EntityType.locators,
                frozenset(
                    [
                        EntityType.EMAIL,
                        EntityType.IP_ADDRESS,
                        EntityType.URL,
                    ]
                ),
            ),
            (
                "numbers",
                EntityType.numbers,
                frozenset([EntityType.PHONE, EntityType.CREDIT_CARD]),
            ),
            (
                "proper_nouns",
                EntityType.proper_nouns,
                frozenset(
                    [
                        EntityType.PERSON,
                        EntityType.PLACE,
                        EntityType.ORGANISATION,
                        EntityType.CONCEPT,
                    ]
                ),
            ),
        ]
    )
    def test_entity_type_class_methods(self, description, method, expected):
        """Test EntityType class methods return correct frozensets."""
        self.assertEqual(method(), expected)

    @parameterized.expand(
        [
            (
                "person",
                EntityType.PERSON,
                "John Doe",
                EntityGroup.PNOUN,
            ),
            (
                "email",
                EntityType.EMAIL,
                "test@example.com",
                EntityGroup.LOCATOR,
            ),
            (
                "phone",
                EntityType.PHONE,
                "555-1234",
                EntityGroup.NUMBER,
            ),
        ]
    )
    def test_entity_group_property(
        self, description, entity_type, value, expected_group
    ):
        """Test Entity.group property returns correct group."""
        entity = Entity(type=entity_type, value=value)
        self.assertEqual(entity.group, expected_group)

    def test_entity_group_unknown_type(self):
        """Test Entity.group raises ValueError for unknown type."""
        entity = Entity(type="INVALID_TYPE", value="test")
        with self.assertRaises(ValueError) as context:
            _ = entity.group
        self.assertIn("Unknown entity type", str(context.exception))


class TestProperNounCollection(unittest.TestCase):
    """Test proper noun collection from text."""

    def setUp(self):
        """Initialise detector for each test."""
        self.detector = EntityDetector()

    def test_detect_proper_nouns_empty(self):
        """Test proper noun detection with empty input."""
        entities, text = self.detector._detect_proper_nouns("")
        self.assertEqual(len(entities), 0)
        self.assertEqual(text, "")

    @parameterized.expand(
        [
            ("contraction_im", "I'm John", ["John"]),
            (
                "contraction_ive",
                "I've met Alice",
                ["Alice"],
            ),
            ("contraction_ill", "I'll see Bob", ["Bob"]),
            (
                "pending_noun_contraction",
                "Dr. Smith I'm going to see Johnson",
                ["Dr", "Smith", "Johnson"],
            ),
            (
                "lookahead_contraction",
                "I'm Alice going home",
                ["Alice"],
            ),
            (
                "skip_next_logic",
                "Hello I've seen Mary before",
                ["Hello", "Mary"],
            ),
            (
                "empty_word_spaces",
                "Dr.   Smith",
                ["Dr", "Smith"],
            ),
            (
                "pending_noun_reset",
                "Dr. Smith went to the store quickly",
                ["Dr", "Smith"],
            ),
            (
                "contraction_with_place",
                "I'm Alice going to London",
                ["Alice", "London"],
            ),
            (
                "contraction_with_hyphenated",
                "I'll see Mary-Jane tomorrow",
                ["Mary-Jane"],
            ),
            (
                "contraction_with_title",
                "I'm calling Professor Johnson",
                ["Professor", "Johnson"],
            ),
            (
                "contraction_with_long_name",
                "I'll Krishnamurthy",
                ["Krishnamurthy"],
            ),
        ]
    )
    def test_proper_noun_collection(self, description, text, expected_names):
        """Test proper noun collection with various inputs."""
        result = self.detector._collect_proper_nouns(text)
        for name in expected_names:
            self.assertTrue(
                any(name in entity for entity in result),
                f"Expected '{name}' in {result} for: {description}",
            )

    def test_collect_proper_nouns_honorifics(self):
        """Test honorific handling in proper noun collection."""
        proper_nouns = self.detector._collect_proper_nouns(
            "Dr. Smith and Ms. Johnson"
        )
        for name in ("Dr", "Smith", "Ms", "Johnson"):
            self.assertIn(name, proper_nouns)

    def test_detect_proper_nouns_no_classification(self):
        """Test proper nouns when all words are lowercase."""
        entities, _ = self.detector._detect_proper_nouns(
            "this is lowercase text"
        )
        self.assertEqual(len(entities), 0)


class TestPersonDetection(unittest.TestCase):
    """Test person name detection and validation."""

    def setUp(self):
        """Initialise detector for each test."""
        self.detector = EntityDetector()

    @parameterized.expand(
        [
            ("simple_name", "John", True),
            ("possessive_name", "John's", True),
            ("name_with_digits", "John2", False),
            ("only_honorifics", "Dr. Ms.", False),
            ("empty_string", "", False),
            ("whitespace_only", "  ", False),
            ("honorific_dr_smith", "Dr. Smith", False),
            ("honorific_only_dr", "Dr.", False),
            ("honorific_only_ms", "Ms.", False),
            ("honorific_only_prof", "Prof.", False),
            (
                "honorific_full_name",
                "Dr. John Smith",
                False,
            ),
            (
                "honorific_ms_full",
                "Ms. Mary Johnson",
                False,
            ),
            (
                "non_common_name",
                "Rajesh Krishnamurthy",
                True,
            ),
            ("non_english_name", "Aleksandr Volkov", True),
            (
                "honorific_professor",
                "Professor Johnson",
                True,
            ),
            ("honorific_only", "Professor", False),
            ("honorific_sir_only", "Sir", False),
            (
                "honorific_punct_skip",
                "Sir .:. Volkov",
                True,
            ),
        ]
    )
    def test_is_person(self, description, text, expected):
        """Test person detection with various inputs."""
        self.assertEqual(self.detector._is_person(text), expected)

    @parameterized.expand(
        [
            # Regression: possessive stripping must remove the literal
            # "'s" suffix only, not arbitrary trailing 's' / "'" chars.
            # Previously used rstrip("'s"), which mangled "Grass" -> "Gra"
            # and let common nouns through as persons.
            ("common_noun_grass", "Grass", False),
            ("common_noun_glass", "Glass", False),
            ("common_noun_class", "Class", False),
            ("common_noun_cross", "Cross", False),
            ("possessive_james", "James's", True),
            ("possessive_chris", "Chris's", False),
        ]
    )
    def test_possessive_suffix_handling(self, description, text, expected):
        """Possessive "'s" must be stripped as a suffix, not char-set."""
        self.assertEqual(self.detector._is_person(text), expected)

    @parameterized.expand(
        [
            ("valid_mary_jane", "Mary-Jane", True),
            ("valid_john_paul", "John-Paul", True),
            ("incomplete_trailing", "Mary-", False),
            ("incomplete_leading", "-Jane", False),
            ("lowercase_second", "mary-Jane", False),
            ("lowercase_first", "Mary-jane", False),
            ("double_hyphen", "Mary--Jane", False),
        ]
    )
    def test_hyphenated_names(self, description, name, expected):
        """Test validation of hyphenated person names."""
        self.assertEqual(self.detector._is_person(name), expected)


class TestCleanPersonName(unittest.TestCase):
    """Test person name cleaning."""

    def setUp(self):
        """Initialise detector for each test."""
        self.detector = EntityDetector()

    @parameterized.expand(
        [
            ("no_honorific", "Jane Doe", "Jane Doe"),
            ("empty_string", "", ""),
            (
                "with_dr",
                "Dr. John Smith",
                "Dr. John Smith",
            ),
            (
                "with_prof",
                "Prof. Mary Johnson",
                "Prof. Mary Johnson",
            ),
            ("honorific_only_dr", "Dr.", "Dr."),
            ("honorific_only_ms", "Ms.", "Ms."),
            ("whitespace", "   ", "   "),
            (
                "professor_removed",
                "Professor Smith",
                "Smith",
            ),
            ("sir_removed", "Sir John", "John"),
        ]
    )
    def test_clean_person_name(self, description, input_name, expected):
        """Test person name cleaning with various inputs."""
        self.assertEqual(
            self.detector._clean_person_name(input_name),
            expected,
        )


class TestOrganisationDetection(unittest.TestCase):
    """Test organisation detection."""

    def setUp(self):
        """Initialise detector for each test."""
        self.detector = EntityDetector()

    @parameterized.expand(
        [
            ("known_org", "Microsoft", True),
            ("regular_name", "John Smith", False),
            ("google_inc", "Google Inc", True),
            (
                "microsoft_corp",
                "Microsoft Corporation",
                True,
            ),
            ("multi_word_times", "New York Times", True),
        ]
    )
    def test_is_organisation(self, description, text, expected):
        """Test organisation detection with various inputs."""
        self.assertEqual(self.detector._is_organisation(text), expected)


class TestPlaceDetection(unittest.TestCase):
    """Test place detection."""

    def setUp(self):
        """Initialise detector for each test."""
        self.detector = EntityDetector()

    @parameterized.expand(
        [
            ("major_city", "New York", True),
            ("world_city", "London", True),
            ("non_place", "Not A Place", False),
            ("street_component", "Main Street", True),
            ("avenue_component", "Oak Avenue", True),
            ("road_component", "Park Road", True),
            (
                "custom_place",
                "Washington Street",
                True,
            ),
        ]
    )
    def test_is_place(self, description, text, expected):
        """Test place detection with various inputs."""
        self.assertEqual(self.detector._is_place(text), expected)


class TestConceptDetection(unittest.TestCase):
    """Test concept detection."""

    def setUp(self):
        """Initialise detector for each test."""
        self.detector = EntityDetector()

    @parameterized.expand(
        [
            ("custom_acronym", "XYZ", True),
            ("custom_concept", "BAZ", True),
            ("lowercase_single", "api", False),
            ("multi_word", "API KEY", False),
            ("with_punctuation", "API!", False),
        ]
    )
    def test_is_concept(self, description, text, expected):
        """Test concept detection with various inputs."""
        self.assertEqual(self.detector._is_concept(text), expected)


class TestClassifyProperNoun(unittest.TestCase):
    """Test proper noun classification."""

    def setUp(self):
        """Initialise detector for each test."""
        self.detector = EntityDetector()

    def test_classify_empty_input(self):
        """Test classification returns None for empty input."""
        self.assertIsNone(self.detector._classify_proper_noun(""))

    @parameterized.expand(
        [
            (
                "org_multi_word",
                "New York Times",
                EntityType.ORGANISATION,
            ),
            (
                "org_corp_suffix",
                "Microsoft Corporation",
                EntityType.ORGANISATION,
            ),
            (
                "person_possessive",
                "John's",
                EntityType.PERSON,
            ),
            (
                "person_hyphenated",
                "Mary-Jane",
                EntityType.PERSON,
            ),
        ]
    )
    def test_linguistic_classification(self, description, text, expected_type):
        """Test classification of various linguistic patterns."""
        result = self.detector._classify_proper_noun(text)
        self.assertIsNotNone(result, f"'{text}' should be classified")
        _, detected_type = result
        self.assertEqual(detected_type, expected_type)


class TestNumberDetection(unittest.TestCase):
    """Test number entity detection."""

    def setUp(self):
        """Initialise detector for each test."""
        self.detector = EntityDetector()

    def test_detect_numbers_empty(self):
        """Test number detection with empty input."""
        entities, text = self.detector._detect_numbers("")
        self.assertEqual(len(entities), 0)
        self.assertEqual(text, "")

    def test_detect_invalid_credit_card(self):
        """Test invalid credit card is not detected."""
        entities, _ = self.detector._detect_numbers("1234567890123456")
        cc_entities = [e for e in entities if e.type == EntityType.CREDIT_CARD]
        self.assertEqual(len(cc_entities), 0)

    def test_phone_number_detection(self):
        """Test phone number detection and value extraction."""
        entities, _ = self.detector._detect_numbers(
            "Call me at +1 (555) 123-4567"
        )
        phone_entities = [e for e in entities if e.type == EntityType.PHONE]
        self.assertEqual(len(phone_entities), 1)
        self.assertEqual(phone_entities[0].value, "+1 (555) 123-4567")

    def test_email_detection_in_numbers_method(self):
        """Test email is detected during number phase."""
        entities, reduced = self.detector._detect_numbers(
            "Contact john@example.com for details"
        )
        email_entities = [e for e in entities if e.type == EntityType.EMAIL]
        self.assertEqual(len(email_entities), 1)
        self.assertEqual(email_entities[0].value, "john@example.com")
        self.assertNotIn("john@example.com", reduced)


class TestLocatorDetection(unittest.TestCase):
    """Test locator entity detection."""

    def setUp(self):
        """Initialise detector for each test."""
        self.detector = EntityDetector()

    def test_detect_locators_empty(self):
        """Test locator detection with empty input."""
        entities, text = self.detector._detect_locators("")
        self.assertEqual(len(entities), 0)
        self.assertEqual(text, "")


class TestEntityConfig(unittest.TestCase):
    """Test entity configuration and selective filtering."""

    @parameterized.expand(
        [
            (
                "disable_locations",
                EntityConfig.disable_locations,
                [
                    EntityType.PLACE,
                    EntityType.IP_ADDRESS,
                    EntityType.URL,
                ],
                [EntityType.PERSON],
            ),
            (
                "disable_persons",
                EntityConfig.disable_persons,
                [EntityType.PERSON],
                [EntityType.EMAIL],
            ),
            (
                "disable_contacts",
                EntityConfig.disable_contacts,
                [EntityType.EMAIL, EntityType.PHONE],
                [EntityType.PERSON],
            ),
            (
                "only_financial",
                EntityConfig.only_financial,
                [EntityType.PERSON, EntityType.EMAIL],
                [EntityType.CREDIT_CARD],
            ),
        ]
    )
    def test_factory_methods(
        self,
        description,
        factory,
        disabled_types,
        enabled_types,
    ):
        """Test EntityConfig factory methods."""
        config = factory()
        for t in disabled_types:
            self.assertFalse(config.is_enabled(t))
        for t in enabled_types:
            self.assertTrue(config.is_enabled(t))

    def test_selective_group_filtering(self):
        """Test disabling all types in a group."""
        config = EntityConfig().with_disabled(
            EntityType.EMAIL,
            EntityType.URL,
            EntityType.IP_ADDRESS,
        )
        detector = EntityDetector(config)
        entities = detector.detect_entities(
            "Contact john@example.com at https://example.com or 192.168.1.1"
        )
        locator_entities = [
            e for e in entities if e.type in EntityType.locators()
        ]
        self.assertEqual(len(locator_entities), 0)

    def test_partial_group_filtering(self):
        """Test disabling one type preserves others."""
        config = EntityConfig().with_disabled(EntityType.EMAIL)
        detector = EntityDetector(config)
        entities = detector.detect_entities(
            "Email john@example.com or visit https://example.com"
        )
        emails = [e for e in entities if e.type == EntityType.EMAIL]
        urls = [e for e in entities if e.type == EntityType.URL]
        self.assertEqual(len(emails), 0)
        self.assertEqual(len(urls), 1)


class TestFullDetectionPipeline(unittest.TestCase):
    """Test the full entity detection pipeline."""

    def setUp(self):
        """Initialise detector for each test."""
        self.detector = EntityDetector()

    def test_complex_multi_entity_text(self):
        """Test detection of multiple entity types."""
        text = (
            "Dr. Sarah Johnson from Microsoft called "
            "about the London project. She said I'm "
            "meeting with Prof. Williams tomorrow at "
            "sarah@microsoft.com or visit "
            "https://microsoft.com"
        )
        entities = self.detector.detect_entities(text)
        entity_types = {e.type for e in entities}

        for expected_type in (
            EntityType.PERSON,
            EntityType.ORGANISATION,
            EntityType.PLACE,
            EntityType.EMAIL,
            EntityType.URL,
        ):
            self.assertIn(expected_type, entity_types)


if __name__ == "__main__":
    unittest.main()
