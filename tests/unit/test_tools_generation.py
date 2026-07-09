from pondercanvas.agent import state_keys as sk
from pondercanvas.agent.tools.generation_tool import make_generate_image_tool
from pondercanvas.providers.image.base import ImageResult
from tests.fixtures.fake_image_provider import FakeImageProvider
from tests.fixtures.fake_tool_context import FakeToolContext
from tests.fixtures.sample_brief import sample_brief


def _brief_dict(**overrides):
    return sample_brief(**overrides).model_dump()


class TestGenerateImageTool:
    def test_writes_last_image_path_to_state(self, tmp_path):
        provider = FakeImageProvider()
        tool = make_generate_image_tool(provider, tmp_path)
        ctx = FakeToolContext({sk.BRIEF: _brief_dict()})

        result = tool(ctx)

        assert result["status"] == "ok"
        assert ctx.state[sk.LAST_IMAGE_PATH] == result["image_path"]
        assert (tmp_path / "iteration_0.png").exists()

    def test_appends_to_iterations_list(self, tmp_path):
        provider = FakeImageProvider()
        tool = make_generate_image_tool(provider, tmp_path)
        ctx = FakeToolContext({sk.BRIEF: _brief_dict()})

        tool(ctx)

        assert len(ctx.state[sk.ITERATIONS]) == 1
        assert ctx.state[sk.ITERATIONS][0]["iteration_index"] == 0

    def test_second_call_uses_iteration_index_one(self, tmp_path):
        provider = FakeImageProvider()
        tool = make_generate_image_tool(provider, tmp_path)
        ctx = FakeToolContext({sk.BRIEF: _brief_dict()})

        tool(ctx)
        tool(ctx)

        assert len(ctx.state[sk.ITERATIONS]) == 2
        assert ctx.state[sk.ITERATIONS][1]["iteration_index"] == 1
        assert (tmp_path / "iteration_1.png").exists()

    def test_prompt_includes_feedback_from_prior_evaluation(self, tmp_path):
        provider = FakeImageProvider()
        tool = make_generate_image_tool(provider, tmp_path)
        ctx = FakeToolContext(
            {
                sk.BRIEF: _brief_dict(),
                sk.LAST_EVALUATION: {"feedback": "make the bicycle bigger"},
            }
        )

        tool(ctx)

        assert "make the bicycle bigger" in provider.calls[0]["prompt"]

    def test_prompt_includes_grounding_context(self, tmp_path):
        provider = FakeImageProvider()
        tool = make_generate_image_tool(provider, tmp_path)
        ctx = FakeToolContext(
            {
                sk.BRIEF: _brief_dict(),
                sk.GROUNDING_RESULT: {"summary_text": "vintage bicycles have curved frames"},
            }
        )

        tool(ctx)

        assert "vintage bicycles have curved frames" in provider.calls[0]["prompt"]

    def test_passes_reference_images_to_provider(self, tmp_path):
        provider = FakeImageProvider()
        tool = make_generate_image_tool(provider, tmp_path)
        ctx = FakeToolContext(
            {sk.BRIEF: _brief_dict(), sk.REFERENCE_IMAGE_BYTES: [b"ref1", b"ref2"]}
        )

        tool(ctx)

        assert provider.calls[0]["reference_images"] == [b"ref1", b"ref2"]

    def test_no_reference_images_defaults_to_empty_list(self, tmp_path):
        provider = FakeImageProvider()
        tool = make_generate_image_tool(provider, tmp_path)
        ctx = FakeToolContext({sk.BRIEF: _brief_dict()})

        tool(ctx)

        assert provider.calls[0]["reference_images"] == []

    def test_file_extension_derived_from_mime_type(self, tmp_path):
        provider = FakeImageProvider(
            results=[
                ImageResult(
                    image_bytes=b"x", mime_type="image/jpeg", provider="fake", model_id="m"
                )
            ]
        )
        tool = make_generate_image_tool(provider, tmp_path)
        ctx = FakeToolContext({sk.BRIEF: _brief_dict()})

        result = tool(ctx)

        assert result["image_path"].endswith(".jpeg")

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested_dir = tmp_path / "does" / "not" / "exist"
        provider = FakeImageProvider()
        tool = make_generate_image_tool(provider, nested_dir)
        ctx = FakeToolContext({sk.BRIEF: _brief_dict()})

        tool(ctx)

        assert nested_dir.exists()
