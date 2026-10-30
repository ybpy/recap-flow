import json
import numpy as np
import pytest

from openpi.e0.research import labels, prepare
from openpi_client.e0.artifacts import write_json


def test_analysis_requires_frame_alignment_and_excludes_experts(tmp_path):
    paths = []
    for name, values in (("a", [999, 0, 1, 2, 3]), ("b", [-999, 3, 2, 1, 0])):
        p = tmp_path / name
        p.mkdir()
        write_json(p / "e0-manifest.json", {"episodes": [{"episode_uid": "x", "length": 5}]})
        np.savez(p / "advantage-diagnostics.npz", advantage=values, task_index=[0]*5,
                 expert=[True, False, False, False, False])
        paths.append(str(p))
    output = tmp_path / "report.json"
    labels(paths, output)
    result = json.loads(output.read_text())["pairs"][0]["tasks"]["0"]
    assert result["frames"] == 4
    assert result["spearman"] == -1
    assert result["label_flip_fraction"] == 0.5
    write_json(tmp_path / "b/e0-manifest.json", {"episodes": [{"episode_uid": "other", "length": 5}]})
    with pytest.raises(ValueError, match="identical ordered"):
        labels(paths, output)


def test_research_never_silently_substitutes_missing_episodes(tmp_path):
    with pytest.raises(ValueError, match="predetermined 100"):
        prepare(tmp_path, "research/unit", tmp_path / "plan.json")
    with pytest.raises(ValueError, match="research/"):
        prepare(tmp_path, "final/unit", tmp_path / "plan.json")


def test_unconditional_prompt_has_dropout_semantics():
    from openpi.transforms import TokenizePrompt

    class Recorder:
        def tokenize(self, prompt, state, adv_ind, adv_ind_dropout):
            assert adv_ind is None
            assert adv_ind_dropout is False
            return np.array([1]), np.array([True])

    transform = TokenizePrompt(Recorder(), adv_ind_input=True, adv_ind_dropout=False)
    result = transform({"prompt": "task", "adv_ind": "unconditional"})
    assert result["tokenized_prompt"][0] == 1
