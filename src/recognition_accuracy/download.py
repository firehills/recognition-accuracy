import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import gdown

from .dataset import discover_images
from .runtime import sha256

LFW_FILE_ID = "1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q"


def safe_path(name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive entry: {name!r}")
    return Path(*path.parts)


def extract(archive: Path, target: Path) -> None:
    """Extract only ordinary files/directories; never follow archive links."""
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                destination = target / safe_path(member.filename)
                mode = member.external_attr >> 16
                if mode & 0o170000 not in (0, 0o040000, 0o100000):
                    raise ValueError(f"Unsupported archive entry: {member.filename}")
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(member) as src, destination.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(archive, "r:*") as source:
            for member in source:
                destination = target / safe_path(member.name)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    stream = source.extractfile(member)
                    if stream is None:
                        raise ValueError(f"Cannot read archive member: {member.name}")
                    with stream, destination.open("xb") as dst:
                        shutil.copyfileobj(stream, dst)
                else:
                    raise ValueError(f"Unsupported archive entry: {member.name}")


def download(
    output: Path,
    archive: Path | None,
    expected_sha256: str | None,
    url: str | None = None,
) -> None:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lfw-download-", dir=output.parent) as temporary:
        work = Path(temporary)
        if archive is None:
            archive = work / "lfw.archive"
            if url:
                result = gdown.download(url=url, output=str(archive), quiet=False, fuzzy=True)
            else:
                result = gdown.download(id=LFW_FILE_ID, output=str(archive), quiet=False)
            if result is None or not archive.is_file():
                raise RuntimeError("Google Drive download did not produce an archive")
        digest = sha256(archive)
        if expected_sha256 and digest.lower() != expected_sha256.lower():
            raise ValueError(f"Archive SHA-256 mismatch: received {digest}")
        extracted = work / "extracted"
        extracted.mkdir()
        extract(archive, extracted)
        if not any(p.suffix.lower() in (".jpg", ".jpeg", ".png") for p in extracted.rglob("*")):
            raise ValueError("Archive contains no supported images")
        try:
            groups = discover_images(extracted)
        except ValueError as error:
            raise ValueError(
                f"Archive (SHA-256 {digest}) is not an LFW dataset: {error} "
                "Supply a genuine LFW archive with --url or --archive."
            ) from error
        extracted.rename(output)
    images = sum(map(len, groups.values()))
    print(f"Extracted {images} images of {len(groups)} identities to {output}")
    print(f"Archive SHA-256: {digest}")
