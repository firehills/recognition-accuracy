import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from recognition_accuracy.download import download, extract, safe_path
from recognition_accuracy.runtime import sha256


@pytest.mark.parametrize(
    "name", ["/etc/escape", "../escape", "lfw/../../escape", "C:/escape", "lfw\\escape"]
)
def test_rejects_unsafe_paths(name):
    with pytest.raises(ValueError):
        safe_path(name)


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_local_archive_download_and_hash(tmp_path, kind):
    archive = tmp_path / f"archive.{kind}"
    if kind == "zip":
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("lfw/Person/Person_0001.jpg", b"image bytes")
    else:
        with tarfile.open(archive, "w") as stream:
            info = tarfile.TarInfo("lfw/Person/Person_0001.jpg")
            info.size = 11
            stream.addfile(info, io.BytesIO(b"image bytes"))
    output = tmp_path / "extracted"
    download(output, archive, sha256(archive))
    assert (output / "lfw/Person/Person_0001.jpg").read_bytes() == b"image bytes"
    with pytest.raises(FileExistsError):
        download(output, archive, None)
    with pytest.raises(ValueError, match="SHA-256"):
        download(tmp_path / "bad-hash", archive, "0" * 64)
    assert not (tmp_path / "bad-hash").exists()


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_extract_rejects_symlink(tmp_path, kind):
    archive = tmp_path / f"archive.{kind}"
    if kind == "zip":
        with zipfile.ZipFile(archive, "w") as stream:
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = 0o120777 << 16
            stream.writestr(info, "/etc/passwd")
    else:
        with tarfile.open(archive, "w") as stream:
            info = tarfile.TarInfo("link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            stream.addfile(info)
    with pytest.raises(ValueError, match="Unsupported"):
        extract(archive, tmp_path / "output")


def test_gdown_default_file_id_and_atomic_failure(tmp_path, monkeypatch):
    from recognition_accuracy import download as module

    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        with zipfile.ZipFile(kwargs["output"], "w") as stream:
            stream.writestr("../escape.jpg", b"bad")
        return kwargs["output"]

    monkeypatch.setattr(module.gdown, "download", fake_download)
    output = tmp_path / "lfw"
    with pytest.raises(ValueError, match="Unsafe"):
        download(output, None, None)
    assert calls[0]["id"] == module.LFW_FILE_ID
    assert not output.exists()
    assert not (tmp_path / "escape.jpg").exists()
    assert not list(tmp_path.glob("lfw-download-*"))


def test_no_image_archive_is_not_published(tmp_path):
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("readme.txt", b"text")
    with pytest.raises(ValueError, match="no supported images"):
        download(tmp_path / "lfw", archive, None)


def test_non_lfw_archive_is_rejected(tmp_path):
    archive = tmp_path / "wider.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("WIDER_val/images/6--Funeral/6_Funeral_Funeral_6_572.jpg", b"x")
    with pytest.raises(ValueError, match="not an LFW dataset"):
        download(tmp_path / "lfw", archive, None)
    assert not (tmp_path / "lfw").exists()


def test_custom_url_is_passed_to_gdown(tmp_path, monkeypatch):
    from recognition_accuracy import download as module

    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        with zipfile.ZipFile(kwargs["output"], "w") as stream:
            stream.writestr("lfw/Person/Person_0001.jpg", b"x")
        return kwargs["output"]

    monkeypatch.setattr(module.gdown, "download", fake_download)
    download(tmp_path / "lfw", None, None, "https://example.com/lfw.zip")
    assert calls[0]["url"] == "https://example.com/lfw.zip"
    assert (tmp_path / "lfw/lfw/Person/Person_0001.jpg").is_file()


def test_safe_path():
    assert safe_path("./lfw/Person_0001.jpg") == Path("lfw/Person_0001.jpg")
