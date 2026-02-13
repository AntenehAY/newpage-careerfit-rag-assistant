"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Path to tests/fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_resume_pdf(fixtures_dir: Path) -> Path:
    """Ensure sample_resume.pdf exists; create if missing."""
    path = fixtures_dir / "sample_resume.pdf"
    if not path.exists():
        from tests.create_fixtures import create_sample_pdf
        create_sample_pdf()
    return path


@pytest.fixture(scope="session")
def sample_jd_docx(fixtures_dir: Path) -> Path:
    """Ensure sample_jd.docx exists; create if missing."""
    path = fixtures_dir / "sample_jd.docx"
    if not path.exists():
        from tests.create_fixtures import create_sample_docx
        create_sample_docx()
    return path


@pytest.fixture
def sample_resume_txt(fixtures_dir: Path) -> Path:
    """Path to sample_resume.txt."""
    return fixtures_dir / "sample_resume.txt"
