import pathlib
import sys

import pytest

# Allow running from the repo root without installing the package
_pkg_root = pathlib.Path(__file__).resolve().parents[2] / "python"
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))


@pytest.fixture(scope="session")
def ctx():
    import dressi.torch as dr

    return dr.RasterizeVulkanContext()
