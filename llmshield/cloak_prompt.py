"""Prompt cloaking module.

Description:
    This module handles the replacement of sensitive entities in prompts with
    secure placeholders before sending to LLMs. It maintains a mapping of
    placeholders to original values for later restoration.

Functions:
    cloak_prompt: Replace sensitive entities with placeholders

Note:
    This module is intended for internal use only. Users should interact
    with the LLMShield class rather than calling these functions directly.

Author:
    LLMShield by brainpolo, 2025-2026

"""

# Standard Library Imports
import re
from collections import OrderedDict

# Local Imports
from .entity_detector import Entity, EntityConfig, EntityDetector
from .utils import wrap_entity


# pylint: disable=too-many-locals
def cloak_prompt(  # noqa: PLR0913
    prompt: str,
    start_delimiter: str,
    end_delimiter: str,
    *,
    entity_map: dict[str, str] | None = None,
    entity_config: EntityConfig | None = None,
    allowlist: frozenset[str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Cloak sensitive entities in prompt with selective configuration.

    Args:
        prompt: Text to cloak entities in
        start_delimiter: Opening delimiter for placeholders
        end_delimiter: Closing delimiter for placeholders
        entity_map: Existing placeholder mappings for consistency
        entity_config: Configuration for selective entity detection
        allowlist: Terms to exclude from cloaking (case-insensitive)

    Returns:
        Tuple of (cloaked_prompt, entity_mapping)

    Note:
        - Collects all match positions from the original prompt
        - Sorts matches in descending order by start index
        - Replaces matches in one pass for optimal performance
        - Maintains placeholder consistency across calls

    """
    if entity_map is None:
        entity_map = OrderedDict()

    # Create a reverse map for quick lookups of existing values
    reversed_entity_map = {v: k for k, v in entity_map.items()}

    detector = EntityDetector(entity_config)
    entities: set[Entity] = detector.detect_entities(prompt)

    # Filter out allowlisted terms (case-insensitive)
    if allowlist:
        allowlist_lower = {v.lower() for v in allowlist}
        entities = {
            e for e in entities if e.value.lower() not in allowlist_lower
        }

    matches = []
    # The counter should start from the current size of the entity map
    # to ensure new placeholders are unique.
    counter = len(entity_map)

    for entity in entities:
        # If the entity value is already in our map, use the existing
        # placeholder
        if entity.value in reversed_entity_map:
            placeholder = reversed_entity_map[entity.value]
        else:
            # Otherwise, create a new placeholder
            placeholder = wrap_entity(
                entity.type,
                counter,
                start_delimiter,
                end_delimiter,
            )
            # Add the new entity to the maps
            entity_map[placeholder] = entity.value
            reversed_entity_map[entity.value] = placeholder
            counter += 1

        # Find all occurrences of the entity value in the prompt
        escaped = re.escape(entity.value)
        for match in re.finditer(escaped, prompt):
            matches.append(
                (match.start(), match.end(), placeholder, entity.value)
            )

    # Sort matches in descending order by the match start index to avoid
    # shifting
    matches.sort(key=lambda m: m[0], reverse=True)

    cloaked_prompt = prompt
    # We don't need to build the entity map here again, just replace the text
    for start, end, placeholder, _ in matches:
        cloaked_prompt = (
            cloaked_prompt[:start] + placeholder + cloaked_prompt[end:]
        )

    return cloaked_prompt, entity_map
