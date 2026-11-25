"""P2 adapter placeholder — PRD §5. P2 artifacts are unavailable at build time."""

from __future__ import annotations

from askback.extract.interface import Field, Value


class P2Extractor:
    """Wraps P2's extractor plus isotonic calibration once P2 exists."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("P2 artifacts unavailable at build time")

    def extract(self, nl_text: str) -> dict[Field, tuple[Value | None, float]]:
        raise NotImplementedError("P2 artifacts unavailable at build time")


if __name__ == "__main__":
    try:
        P2Extractor()
    except NotImplementedError:
        pass
    else:
        raise AssertionError("P2 adapter must refuse construction")
    print("p2 adapter ok")