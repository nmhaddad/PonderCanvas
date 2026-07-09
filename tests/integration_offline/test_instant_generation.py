from pondercanvas.agent import state_keys as sk
from pondercanvas.agent.refinement import run_instant_generation
from pondercanvas.agent.tools.generation_tool import make_generate_image_tool
from tests.fixtures.fake_image_provider import FakeImageProvider
from tests.fixtures.sample_brief import sample_brief


def _generation_tool(tmp_path):
    image_provider = FakeImageProvider()
    return image_provider, make_generate_image_tool(image_provider, tmp_path)


class TestInstantGenerationRunsGenerationOnce:
    def test_generation_called_exactly_once(self, tmp_path):
        image_provider, gen = _generation_tool(tmp_path)

        final_state = run_instant_generation(gen, {sk.BRIEF: sample_brief().model_dump()})

        assert len(image_provider.calls) == 1
        assert len(final_state[sk.ITERATIONS]) == 1

    def test_no_evaluation_is_recorded(self, tmp_path):
        _, gen = _generation_tool(tmp_path)

        final_state = run_instant_generation(gen, {sk.BRIEF: sample_brief().model_dump()})

        assert sk.LAST_EVALUATION not in final_state
        assert final_state[sk.LAST_IMAGE_PATH]

    def test_does_not_mutate_the_provided_initial_state(self, tmp_path):
        _, gen = _generation_tool(tmp_path)
        initial_state = {sk.BRIEF: sample_brief().model_dump()}

        run_instant_generation(gen, initial_state)

        assert sk.ITERATIONS not in initial_state
        assert sk.LAST_IMAGE_PATH not in initial_state
