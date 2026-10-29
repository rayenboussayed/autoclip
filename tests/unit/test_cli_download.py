"""The downloader command line modes: ``--formats``, ``--platforms``, ``--download-only`` and mixed input.

Everything is offline: the ``Downloader`` is replaced by a scripted fake and ``process_video`` by a
recorder, so these tests only check how the command line wires the pieces together.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from autoclip import cli
from autoclip.cli import main
from autoclip.core.config import Config, parse
from autoclip.core.download import FormatReport
from autoclip.core.errors import ConfigError, DownloadError
from autoclip.core.models import Downloaded
from autoclip.core.resources import RESOURCES

from ..support import Media, write_config

URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
OTHER = "https://vimeo.com/12345"
BAD = "https://example.com/broken"


@pytest.fixture(autouse=True)
def _reset_resources() -> Iterator[None]:
    yield
    RESOURCES.configure(Config())


def install_fake_downloader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    init_error: Exception | None = None,
    downloads: Mapping[str, list[Downloaded] | Exception] | None = None,
    formats: Mapping[str, Exception] | None = None,
) -> list[Any]:
    """Replace ``cli.Downloader`` with a scripted class and return the instances it created."""
    instances: list[Any] = []

    class Fake:
        def __init__(self, cfg: Config) -> None:
            if init_error is not None:
                raise init_error
            self.cfg = cfg
            instances.append(self)

        def formats(self, url: str) -> FormatReport:
            if formats and url in formats:
                raise formats[url]
            return FormatReport(url=url, title="A video", platform="youtube", chosen="v", rows=(), warnings=("one warning",))

        def policy_text(self) -> str:
            return "allow: @popular\ndeny: -\npermitted: youtube"

        def describe(self, url: str) -> str:
            return f"{url}: youtube"

        def download(self, url: str) -> list[Downloaded]:
            value = (downloads or {})[url]
            if isinstance(value, Exception):
                raise value
            return value

    monkeypatch.setattr(cli, "Downloader", Fake)
    return instances


def new_downloader() -> Any:
    """Instantiate the class ``install_fake_downloader`` put in place of ``cli.Downloader``."""
    # Through ``vars`` on purpose: the name is imported into ``cli`` (not re-exported) and replaced by the fixture.
    downloader_class = vars(cli)["Downloader"]
    return downloader_class(parse(Config, {}))


def record_pipeline(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the encoder probe and ``process_video``; every call counts as one rendered clip."""
    calls: list[dict[str, Any]] = []

    def process_video(
        path: Path,
        cfg: Config,
        log_file: Path | None,
        encoder: Any,
        *,
        name: str | None = None,
        subtitle: Path | None = None,
        subtitle_manual: bool = False,
        report: Any = None,
        cancel: Any = None,
    ) -> tuple[int, int]:
        calls.append({"path": Path(path), "name": name, "subtitle": subtitle, "manual": subtitle_manual})
        return 1, 0
        calls.append({"path": Path(path), "name": name, "subtitle": subtitle, "manual": subtitle_manual})
        return 1, 0

    import autoclip.core.batch as batch_mod

    monkeypatch.setattr(batch_mod, "process_video", process_video)
    monkeypatch.setattr(cli, "pick_encoder", lambda *args, **kwargs: None)
    return calls


def test_csv_merges_repeated_and_comma_separated_values() -> None:
    assert cli._csv(None) is None
    assert cli._csv([]) == []
    assert cli._csv(["a,b", " c , ,d "]) == ["a", "b", "c", "d"]


def test_with_policy_replaces_only_the_lists_that_were_given() -> None:
    cfg = parse(Config, {"download": {"platforms": {"allow": ["youtube"], "deny": ["tiktok"], "groups": {"@mine": ["vimeo"]}}}})
    cli._with_policy(cfg, ["@mine"], None)
    assert cfg.download.platforms.allow == ["@mine"] and cfg.download.platforms.deny == ["tiktok"]
    assert cfg.download.platforms.groups == {"@mine": ["vimeo"]}
    cli._with_policy(cfg, None, ["vimeo"])
    assert cfg.download.platforms.allow == ["@mine"] and cfg.download.platforms.deny == ["vimeo"]
    cli._with_policy(cfg, None, None)  # nothing to do
    assert cfg.download.platforms.deny == ["vimeo"]


def test_target_urls_uses_the_given_url_or_every_url_of_the_input() -> None:
    cfg = parse(Config, {"input": ["a.mp4", URL, OTHER]})
    assert cli._target_urls(cfg, URL) == [URL]
    assert cli._target_urls(cfg, True) == [URL, OTHER]


def test_unique_name_counts_collisions_case_insensitively() -> None:
    used = {"talk"}
    assert cli._unique_name("Talk", used) == "Talk (2)"
    assert cli._unique_name("TALK", used) == "TALK (3)"
    assert cli._unique_name("other", used) == "other"
    assert used == {"talk", "talk (2)", "talk (3)", "other"}


def test_show_formats_prints_every_table_and_keeps_going_after_an_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_fake_downloader(monkeypatch, formats={BAD: DownloadError("no formats")})
    downloader = new_downloader()
    assert cli._show_formats(downloader, [URL, BAD]) == 1
    out = capsys.readouterr()
    assert "title:    A video" in out.out and "! one warning" in out.out
    assert f"error: {BAD}: no formats" in out.err
    assert cli._show_formats(downloader, [BAD]) == 1  # every URL failed: no table at all
    assert capsys.readouterr().out == ""


def test_show_platforms_prints_the_policy_and_each_url(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    install_fake_downloader(monkeypatch)
    downloader = new_downloader()
    assert cli._show_platforms(downloader, []) == 0
    assert "allow: @popular" in capsys.readouterr().out
    assert cli._show_platforms(downloader, [URL, OTHER]) == 0
    out = capsys.readouterr().out
    assert f"{URL}\n  {URL}: youtube" in out and f"{OTHER}\n  {OTHER}: youtube" in out


def test_download_only_saves_every_url_and_counts_failures(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    entries: dict[str, list[Downloaded] | Exception] = {
        URL: [Downloaded(Path("a.mp4"), "A", "youtube")],
        OTHER: DownloadError("blocked"),
    }
    install_fake_downloader(monkeypatch, downloads=entries)
    downloader = new_downloader()
    cfg = parse(Config, {"input": [URL, OTHER]})
    with caplog.at_level(logging.INFO, logger="autoclip"):
        assert cli._download_only(cfg, downloader) == 1
    assert "saved a.mp4 (youtube)" in caplog.text
    assert f"{OTHER} failed: blocked" in caplog.text
    assert "finished: 1 original(s) downloaded, 1 URL(s) failed" in caplog.text


def test_download_only_needs_a_url(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    install_fake_downloader(monkeypatch)
    downloader = new_downloader()
    assert cli._download_only(parse(Config, {"input": ["a.mp4"]}), downloader) == 2
    assert "--download-only needs at least one URL" in capsys.readouterr().err


def test_inspection_runs_without_a_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    install_fake_downloader(monkeypatch)
    assert main(["--formats", URL]) == 0
    assert "chosen:   v" in capsys.readouterr().out
    assert main(["--platforms", URL]) == 0
    assert "allow: @popular" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []  # inspecting a URL creates nothing


def test_inspection_needs_a_url_when_the_default_config_has_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    install_fake_downloader(monkeypatch)
    assert main(["--formats"]) == 2


def test_the_two_inspection_modes_are_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["--formats", URL, "--platforms", URL]) == 2
    assert "choose one of --formats and --platforms" in capsys.readouterr().err


def test_inspection_reads_the_default_config_when_it_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(f'{{"input": ["{URL}"]}}', encoding="utf-8")
    install_fake_downloader(monkeypatch)
    assert main(["--platforms"]) == 0
    assert f"{URL}\n  {URL}: youtube" in capsys.readouterr().out


def test_no_config_and_no_input_file_is_a_friendly_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main([]) == 2
    error = capsys.readouterr().err
    assert "no config file and no input.mp4" in error and "hint:" in error
    assert list(tmp_path.iterdir()) == []  # starting without a config writes nothing


def test_no_config_runs_input_mp4_with_the_built_in_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a config file, ./input.mp4 is cut with the defaults and no config is invented."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input.mp4").write_bytes(b"x")
    calls = record_pipeline(monkeypatch)
    assert main([]) == 0
    assert [call["path"] for call in calls] == [Path("input.mp4")]
    assert not (tmp_path / "config.json").exists()


def test_no_config_download_only_needs_a_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["--download-only"]) == 2
    assert "--download-only needs at least one URL" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(("error", "status"), [(ConfigError("bad policy"), 2), (DownloadError("yt-dlp is missing"), 1)])
def test_inspection_reports_downloader_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception, status: int
) -> None:
    monkeypatch.chdir(tmp_path)
    install_fake_downloader(monkeypatch, init_error=error)
    assert main(["--formats", URL]) == status
    assert str(error) in capsys.readouterr().err


def test_allow_and_deny_override_the_loaded_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    instances = install_fake_downloader(monkeypatch)
    assert main(["--formats", URL, "--allow", "youtube,tiktok", "--deny", "@all", "--allow", "vimeo"]) == 0
    policy = instances[0].cfg.download.platforms
    assert policy.allow == ["youtube", "tiktok", "vimeo"] and policy.deny == ["@all"]


def test_download_only_runs_before_the_pipeline(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entries = {URL: [Downloaded(tmp_path / "a.mp4", "A", "youtube")]}
    install_fake_downloader(monkeypatch, downloads=entries)
    calls = record_pipeline(monkeypatch)
    cfg = write_config(tmp_path, media, input=[URL], logging={"enabled": False})
    assert main(["--download-only", str(cfg)]) == 0
    assert calls == []


def test_download_only_reports_a_failed_url(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_downloader(monkeypatch, downloads={URL: DownloadError("blocked")})
    cfg = write_config(tmp_path, media, input=[URL], logging={"enabled": False})
    assert main(["--download-only", str(cfg)]) == 1


def test_download_only_without_urls_is_a_config_error(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_downloader(monkeypatch)  # must not even be constructed
    cfg = write_config(tmp_path, media, input=[str(media.video)], logging={"enabled": False})
    assert main(["--download-only", str(cfg)]) == 2


@pytest.mark.parametrize(("error", "status"), [(ConfigError("bad policy"), 2), (DownloadError("yt-dlp is missing"), 1)])
def test_downloader_init_errors_stop_a_run(
    media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, status: int
) -> None:
    install_fake_downloader(monkeypatch, init_error=error)
    cfg = write_config(tmp_path, media, input=[URL], logging={"enabled": False})
    assert main([str(cfg)]) == status


def test_urls_and_local_files_share_one_batch(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    subtitle = tmp_path / "downloaded.srt"
    subtitle.write_text(media.srt.read_text(encoding="utf-8"), encoding="utf-8")
    entries = {
        URL: [
            Downloaded(tmp_path / "a.mp4", "TALK", "youtube", subtitle, True),
            Downloaded(tmp_path / "b.mp4", "Talk", "youtube"),
        ]
    }
    install_fake_downloader(monkeypatch, downloads=entries)
    calls = record_pipeline(monkeypatch)
    cfg = write_config(tmp_path, media, input=[str(media.video), URL], logging={"enabled": False})
    assert main([str(cfg)]) == 0
    assert [(c["path"].name, c["name"], c["subtitle"], c["manual"]) for c in calls] == [
        ("talk.mp4", None, None, False),
        ("a.mp4", "TALK (2)", subtitle, True),
        ("b.mp4", "Talk (3)", None, False),
    ]


def test_a_failed_download_does_not_stop_the_local_video(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_downloader(monkeypatch, downloads={URL: DownloadError("blocked")})
    calls = record_pipeline(monkeypatch)
    cfg = write_config(tmp_path, media, input=[str(media.video), URL], logging={"enabled": False})
    assert main([str(cfg)]) == 1
    assert [c["path"].name for c in calls] == ["talk.mp4"]


def test_a_local_only_run_never_imports_yt_dlp(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "yt_dlp", None)
    calls = record_pipeline(monkeypatch)
    cfg = write_config(tmp_path, media, input=[str(media.video)], logging={"enabled": False})
    assert main([str(cfg)]) == 0
    assert [c["path"].name for c in calls] == ["talk.mp4"]
