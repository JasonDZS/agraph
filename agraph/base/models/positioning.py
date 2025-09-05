"""
Entity positioning system for knowledge graph elements.

This module provides positioning capabilities for entities and relations,
enabling precise location tracking within source documents. Based on
LangExtract's positioning analysis, it implements dual-level positioning
(character and token) with alignment status tracking.
"""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class AlignmentStatus(Enum):
    """Alignment status enumeration for entity positioning."""

    MATCH_EXACT = "match_exact"  # Perfect exact match
    MATCH_GREATER = "match_greater"  # Source text contains extraction text
    MATCH_LESSER = "match_lesser"  # Extraction text contains source text
    MATCH_FUZZY = "match_fuzzy"  # Fuzzy match with similarity threshold
    NO_MATCH = "no_match"  # Alignment failed


class CharInterval(BaseModel):
    """Character-level position representation.

    Represents a precise character position range within a document.
    Uses inclusive start and exclusive end positions (Python slice convention).

    Attributes:
        start_pos: Starting character position (inclusive)
        end_pos: Ending character position (exclusive)
    """

    start_pos: int = Field(ge=0, description="Starting character position (inclusive)")
    end_pos: int = Field(ge=0, description="Ending character position (exclusive)")

    @field_validator("end_pos")
    @classmethod
    def validate_end_pos(cls, v: int, info: ValidationInfo) -> int:
        """Validate end position is after start position."""
        if "start_pos" in info.data and v <= info.data["start_pos"]:
            raise ValueError("end_pos must be greater than start_pos")
        return v

    @property
    def length(self) -> int:
        """Get the length of the character interval."""
        return self.end_pos - self.start_pos

    def contains(self, pos: int) -> bool:
        """Check if a character position is within this interval."""
        return self.start_pos <= pos < self.end_pos

    def overlaps(self, other: "CharInterval") -> bool:
        """Check if this interval overlaps with another interval."""
        return self.start_pos < other.end_pos and other.start_pos < self.end_pos

    def is_nested_in(self, other: "CharInterval") -> bool:
        """Check if this interval is completely nested within another."""
        return other.start_pos <= self.start_pos and self.end_pos <= other.end_pos

    def __str__(self) -> str:
        """String representation of the character interval."""
        return f"[{self.start_pos}-{self.end_pos}]"


class TokenInterval(BaseModel):
    """Token-level position representation.

    Represents position range at the token level for NLP processing.
    Uses inclusive start and exclusive end indices.

    Attributes:
        start_index: Starting token index (inclusive)
        end_index: Ending token index (exclusive)
    """

    start_index: int = Field(default=0, ge=0, description="Starting token index (inclusive)")
    end_index: int = Field(default=0, ge=0, description="Ending token index (exclusive)")

    @field_validator("end_index")
    @classmethod
    def validate_end_index(cls, v: int, info: ValidationInfo) -> int:
        """Validate end index is after start index."""
        if "start_index" in info.data and v <= info.data["start_index"]:
            raise ValueError("end_index must be greater than start_index")
        return v

    @property
    def length(self) -> int:
        """Get the length of the token interval."""
        return self.end_index - self.start_index

    def contains(self, index: int) -> bool:
        """Check if a token index is within this interval."""
        return self.start_index <= index < self.end_index

    def overlaps(self, other: "TokenInterval") -> bool:
        """Check if this interval overlaps with another interval."""
        return self.start_index < other.end_index and other.start_index < self.end_index

    def __str__(self) -> str:
        """String representation of the token interval."""
        return f"[{self.start_index}:{self.end_index}]"


class Position(BaseModel):
    """Unified position management for entities and relations.

    Combines character-level and token-level positioning with alignment status
    for comprehensive location tracking within source documents.

    Attributes:
        char_interval: Character-level position (for UI highlighting)
        token_interval: Token-level position (for NLP processing)
        alignment_status: Status of the alignment process
        confidence: Confidence score for the positioning accuracy
        source_context: Additional context about the source location
    """

    char_interval: Optional[CharInterval] = Field(default=None, description="Character-level position interval")
    token_interval: Optional[TokenInterval] = Field(default=None, description="Token-level position interval")
    alignment_status: AlignmentStatus = Field(
        default=AlignmentStatus.NO_MATCH, description="Status of the alignment process"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score for positioning accuracy")
    source_context: str = Field(default="", description="Additional context about the source location")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    @property
    def is_positioned(self) -> bool:
        """Check if position information is available."""
        return self.char_interval is not None or self.token_interval is not None

    @property
    def is_precisely_aligned(self) -> bool:
        """Check if the position is precisely aligned."""
        return self.alignment_status == AlignmentStatus.MATCH_EXACT

    @property
    def has_char_position(self) -> bool:
        """Check if character position is available."""
        return self.char_interval is not None

    @property
    def has_token_position(self) -> bool:
        """Check if token position is available."""
        return self.token_interval is not None

    def get_char_range(self) -> tuple[int, int]:
        """Get character position range as tuple.

        Returns:
            Tuple of (start_pos, end_pos) or (0, 0) if no char position
        """
        if self.char_interval:
            return (self.char_interval.start_pos, self.char_interval.end_pos)
        return (0, 0)

    def get_token_range(self) -> tuple[int, int]:
        """Get token position range as tuple.

        Returns:
            Tuple of (start_index, end_index) or (0, 0) if no token position
        """
        if self.token_interval:
            return (self.token_interval.start_index, self.token_interval.end_index)
        return (0, 0)

    def overlaps_with(self, other: "Position") -> bool:
        """Check if this position overlaps with another position.

        Args:
            other: Another Position instance to check against

        Returns:
            True if positions overlap, False otherwise
        """
        if self.char_interval and other.char_interval:
            return self.char_interval.overlaps(other.char_interval)
        if self.token_interval and other.token_interval:
            return self.token_interval.overlaps(other.token_interval)
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary representation."""
        return {
            "char_interval": (
                {
                    "start_pos": self.char_interval.start_pos,
                    "end_pos": self.char_interval.end_pos,
                }
                if self.char_interval
                else None
            ),
            "token_interval": (
                {
                    "start_index": self.token_interval.start_index,
                    "end_index": self.token_interval.end_index,
                }
                if self.token_interval
                else None
            ),
            "alignment_status": self.alignment_status.value,
            "confidence": self.confidence,
            "source_context": self.source_context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        """Create Position from dictionary data."""
        char_interval = None
        if data.get("char_interval"):
            char_data = data["char_interval"]
            char_interval = CharInterval(start_pos=char_data["start_pos"], end_pos=char_data["end_pos"])

        token_interval = None
        if data.get("token_interval"):
            token_data = data["token_interval"]
            token_interval = TokenInterval(start_index=token_data["start_index"], end_index=token_data["end_index"])

        alignment_status = AlignmentStatus(data.get("alignment_status", AlignmentStatus.NO_MATCH.value))

        return cls(
            char_interval=char_interval,
            token_interval=token_interval,
            alignment_status=alignment_status,
            confidence=data.get("confidence", 1.0),
            source_context=data.get("source_context", ""),
        )


class PositionMixin(BaseModel):
    """Mixin for adding positioning capabilities to entities and relations.

    This mixin provides position-related functionality that can be added
    to Entity and Relation classes to enable source document positioning.
    """

    position: Optional[Position] = Field(default=None, description="Position information for the entity or relation")

    def set_char_position(
        self,
        start_pos: int,
        end_pos: int,
        alignment_status: AlignmentStatus = AlignmentStatus.MATCH_EXACT,
        confidence: float = 1.0,
    ) -> None:
        """Set character-level position.

        Args:
            start_pos: Starting character position
            end_pos: Ending character position
            alignment_status: Status of the alignment
            confidence: Confidence score for the positioning
        """
        char_interval = CharInterval(start_pos=start_pos, end_pos=end_pos)

        if self.position is None:
            self.position = Position(
                char_interval=char_interval,
                alignment_status=alignment_status,
                confidence=confidence,
            )
        else:
            self.position.char_interval = char_interval
            self.position.alignment_status = alignment_status
            self.position.confidence = confidence

        if hasattr(self, "touch"):
            self.touch()

    def set_token_position(
        self,
        start_index: int,
        end_index: int,
        alignment_status: AlignmentStatus = AlignmentStatus.MATCH_EXACT,
        confidence: float = 1.0,
    ) -> None:
        """Set token-level position.

        Args:
            start_index: Starting token index
            end_index: Ending token index
            alignment_status: Status of the alignment
            confidence: Confidence score for the positioning
        """
        token_interval = TokenInterval(start_index=start_index, end_index=end_index)

        if self.position is None:
            self.position = Position(
                token_interval=token_interval,
                alignment_status=alignment_status,
                confidence=confidence,
            )
        else:
            self.position.token_interval = token_interval
            self.position.alignment_status = alignment_status
            self.position.confidence = confidence

        if hasattr(self, "touch"):
            self.touch()

    def set_position(self, position: Position) -> None:
        """Set complete position information.

        Args:
            position: Position instance with all positioning data
        """
        self.position = position
        if hasattr(self, "touch"):
            self.touch()

    def get_char_position(self) -> Optional[tuple[int, int]]:
        """Get character position as tuple.

        Returns:
            Tuple of (start_pos, end_pos) or None if no position
        """
        if self.position and self.position.char_interval:
            return self.position.get_char_range()
        return None

    def get_token_position(self) -> Optional[tuple[int, int]]:
        """Get token position as tuple.

        Returns:
            Tuple of (start_index, end_index) or None if no position
        """
        if self.position and self.position.token_interval:
            return self.position.get_token_range()
        return None

    def has_position(self) -> bool:
        """Check if position information is available."""
        return self.position is not None and self.position.is_positioned

    def has_precise_position(self) -> bool:
        """Check if position is precisely aligned."""
        return self.position is not None and self.position.is_precisely_aligned

    def get_alignment_status(self) -> AlignmentStatus:
        """Get the alignment status."""
        if self.position:
            return self.position.alignment_status
        return AlignmentStatus.NO_MATCH

    def get_position_confidence(self) -> float:
        """Get the positioning confidence score."""
        if self.position:
            return self.position.confidence
        return 0.0

    def overlaps_with(self, other: "PositionMixin") -> bool:
        """Check if this object's position overlaps with another positioned object.

        Args:
            other: Another object with PositionMixin capabilities

        Returns:
            True if positions overlap, False otherwise
        """
        if not (self.has_position() and hasattr(other, "has_position") and other.has_position()):
            return False

        # Add None check for position
        if self.position is None or other.position is None:
            return False

        return self.position.overlaps_with(other.position)

    def clear_position(self) -> None:
        """Clear all position information."""
        self.position = None
        if hasattr(self, "touch"):
            self.touch()
