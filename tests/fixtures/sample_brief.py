from pondercanvas.schemas.brief import GenerationBrief


def sample_brief(**overrides) -> GenerationBrief:
    defaults: dict = dict(
        subject="a red bicycle",
        style="watercolor",
        composition="centered, three-quarter view",
        mood="cheerful",
        palette="warm reds and oranges",
        constraints=["no text", "square crop"],
        notes_from_references=None,
        search_queries=["watercolor bicycle illustration"],
        aspect_ratio="1:1",
        raw_user_prompt="draw me a red bicycle",
    )
    defaults.update(overrides)
    return GenerationBrief(**defaults)
