from io import BytesIO

import pytest
from PIL import Image, ImageChops

from src.pdf_preview import MAX_PIXELS, render_page
from tests.helpers import make_pdf


def test_renderer_opens_exact_page_and_bounds_image_size(tmp_path):
    path = tmp_path / "two-pages.pdf"
    path.write_bytes(make_pdf(["FIRST PAGE", "SECOND PAGE"]))
    info = path.stat()
    first, count = render_page(str(path), info.st_mtime_ns, info.st_size, 1)
    second, _ = render_page(str(path), info.st_mtime_ns, info.st_size, 2)
    assert count == 2
    with Image.open(BytesIO(first)) as image1, Image.open(BytesIO(second)) as image2:
        assert image1.width * image1.height <= MAX_PIXELS
        assert image1.width <= 1400
        assert (
            ImageChops.difference(image1.convert("RGB"), image2.convert("RGB")).getbbox()
            is not None
        )
    for page in [0, 3]:
        with pytest.raises(ValueError, match="not available"):
            render_page(str(path), info.st_mtime_ns, info.st_size, page)


def test_renderer_does_not_reuse_stale_page_after_file_change(tmp_path):
    path = tmp_path / "source.pdf"
    path.write_bytes(make_pdf(["OLD PAGE"]))
    old = path.stat()
    first, _ = render_page(str(path), old.st_mtime_ns, old.st_size, 1)
    path.write_bytes(make_pdf(["NEW PAGE WITH DIFFERENT TEXT"]))
    current = path.stat()
    second, _ = render_page(str(path), current.st_mtime_ns, current.st_size, 1)
    assert first != second
