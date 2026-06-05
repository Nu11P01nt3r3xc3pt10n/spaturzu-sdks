import pytest
import spaturzu as sp
from spaturzu import configure, get_default_spaturzu, __reset_default_for_tests


@pytest.fixture(autouse=True)
def _reset():
    __reset_default_for_tests()
    yield
    __reset_default_for_tests()


def test_singleton_identity():
    assert get_default_spaturzu() is get_default_spaturzu()


def test_configure_forwards_options():
    configure(tags={"env": "test"})
    inst = get_default_spaturzu()
    assert inst._process_tags == {"env": "test"}


def test_configure_after_construction_raises():
    get_default_spaturzu()
    with pytest.raises(RuntimeError, match="before constructing"):
        configure(tags={"env": "x"})
