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

    def test_writes_interaction_id_to_state(self, tmp_path):
        provider = FakeImageProvider()
        tool = make_generate_image_tool(provider, tmp_path)
        ctx = FakeToolContext({sk.BRIEF: _brief_dict()})

        tool(ctx)

        assert ctx.state[sk.LAST_INTERACTION_ID] == "fake-interaction-id"
        assert ctx.state[sk.ITERATIONS][0]["interaction_id"] == "fake-interaction-id"

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

    def test_revision_passes_previous_interaction_id_instead_of_image_bytes(self, tmp_path):
        # The core of iterative refinement: once there's a prior attempt and
        # feedback, the model continues that generation via
        # previous_interaction_id (so it can edit its own last image) rather
        # than regenerating blind and re-rolling the same flaw -- no bytes
        # need to be re-uploaded for it to do so.
        provider = FakeImageProvider()
        tool = make_generate_image_tool(provider, tmp_path)
        ctx = FakeToolContext({sk.BRIEF: _brief_dict()})

        tool(ctx)  # iteration 0
        ctx.state[sk.LAST_EVALUATION] = {"feedback": "add a strap to the camera"}
        tool(ctx)  # iteration 1: revision

        assert provider.calls[1]["reference_images"] == []
        assert provider.calls[1]["params"]["previous_interaction_id"] == "fake-interaction-id"

    def test_revision_drops_user_references_and_sends_no_reference_images(self, tmp_path):
        # User-supplied references are for the first generation only; later
        # passes refine the model's own previous output (implicit via
        # previous_interaction_id), not the references.
        provider = FakeImageProvider()
        tool = make_generate_image_tool(provider, tmp_path)
        ctx = FakeToolContext(
            {sk.BRIEF: _brief_dict(), sk.REFERENCE_IMAGE_BYTES: [b"ref1", b"ref2"]}
        )

        tool(ctx)
        assert provider.calls[0]["reference_images"] == [b"ref1", b"ref2"]

        ctx.state[sk.LAST_EVALUATION] = {"feedback": "fix the camera mount"}
        tool(ctx)
        assert provider.calls[1]["reference_images"] == []

    def test_revision_prompt_frames_feedback_as_corrections_to_previous_image(self, tmp_path):
        provider = FakeImageProvider()
        tool = make_generate_image_tool(provider, tmp_path)
        ctx = FakeToolContext({sk.BRIEF: _brief_dict()})

        tool(ctx)
        ctx.state[sk.LAST_EVALUATION] = {"feedback": "add a strap to the camera"}
        tool(ctx)

        prompt = provider.calls[1]["prompt"]
        assert "Corrections to apply" in prompt
        assert "previous attempt" in prompt
        assert "add a strap to the camera" in prompt

    def test_first_iteration_does_not_pass_a_previous_image(self, tmp_path):
        # No prior attempt yet: nothing to revise, only real references flow.
        provider = FakeImageProvider()
        tool = make_generate_image_tool(provider, tmp_path)
        ctx = FakeToolContext(
            {sk.BRIEF: _brief_dict(), sk.REFERENCE_IMAGE_BYTES: [b"ref1"]}
        )

        tool(ctx)

        assert provider.calls[0]["reference_images"] == [b"ref1"]
        assert provider.calls[0]["params"]["previous_interaction_id"] is None
        assert "Corrections to apply" not in provider.calls[0]["prompt"]

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

    def test_explicit_prompt_is_used_verbatim_instead_of_the_template(self, tmp_path):
        provider = FakeImageProvider()
        tool = make_generate_image_tool(provider, tmp_path)
        ctx = FakeToolContext({sk.BRIEF: _brief_dict()})

        result = tool(ctx, prompt="a hand-authored scene description")

        assert provider.calls[0]["prompt"] == "a hand-authored scene description"
        assert ctx.state[sk.ITERATIONS][0]["prompt_used"] == "a hand-authored scene description"
        assert result["status"] == "ok"

    def test_extra_reference_images_are_appended_and_then_cleared(self, tmp_path):
        provider = FakeImageProvider()
        tool = make_generate_image_tool(provider, tmp_path)
        ctx = FakeToolContext(
            {
                sk.BRIEF: _brief_dict(),
                sk.REFERENCE_IMAGE_BYTES: [b"ref1"],
                sk.EXTRA_REFERENCE_IMAGE_BYTES: [b"searched1", b"searched2"],
            }
        )

        tool(ctx)
        assert provider.calls[0]["reference_images"] == [b"ref1", b"searched1", b"searched2"]
        assert ctx.state[sk.EXTRA_REFERENCE_IMAGE_BYTES] == []

        # A later call with nothing newly searched shouldn't reuse the stale batch.
        tool(ctx)
        assert provider.calls[1]["reference_images"] == [b"ref1"]

    def test_extra_reference_images_are_passed_alone_on_revision(self, tmp_path):
        # The previous image is implicit via previous_interaction_id on
        # revision, so only freshly-searched extras go in reference_images.
        provider = FakeImageProvider()
        tool = make_generate_image_tool(provider, tmp_path)
        ctx = FakeToolContext({sk.BRIEF: _brief_dict()})

        tool(ctx)
        ctx.state[sk.LAST_EVALUATION] = {"feedback": "make the cat wet"}
        ctx.state[sk.EXTRA_REFERENCE_IMAGE_BYTES] = [b"wet-cat-ref"]
        tool(ctx)

        assert provider.calls[1]["reference_images"] == [b"wet-cat-ref"]
