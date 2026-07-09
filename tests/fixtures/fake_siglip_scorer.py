class FakeSiglipScorer:
    """Scripted SigLIP-like scorer: returns `next_score` from every call (or
    None, to simulate an unavailable/failed model), recording each call."""

    def __init__(self, next_score: float | None):
        self.next_score = next_score
        self.calls: list[dict] = []

    def score(self, image_bytes: bytes, prompt: str) -> float | None:
        self.calls.append({"image_bytes": image_bytes, "prompt": prompt})
        return self.next_score
