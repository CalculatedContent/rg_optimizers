from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_movie_renderer_is_headless_and_single_matrix() -> None:
    source = (
        ROOT
        / "src"
        / "rg_nanogpt_one_head"
        / "muonclip_esd_movie.py"
    ).read_text()

    assert 'os.environ["MPLBACKEND"] = "Agg"' in source
    assert 'matplotlib.use("Agg", force=True)' in source
    assert 'plt.show = lambda *args, **kwargs: None' in source
    assert "class OneMatrixModel" in source
    assert "savefig=savedir" in source
    assert "randomize=False" in source
    assert "ERG=False" in source
    assert "libx264" in source
    assert "native_esd_frames" in source
