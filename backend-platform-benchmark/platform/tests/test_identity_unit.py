from types import SimpleNamespace

import pytest

from platform_app.errors import PlatformError
from platform_app.identity import _prevent_last_admin_demotion


def test_cannot_demote_last_admin():
    admin = SimpleNamespace(role="admin")
    with pytest.raises(PlatformError) as error:
        _prevent_last_admin_demotion(admin, "user", [admin])
    assert error.value.status_code == 409
    assert error.value.code == "last_admin_required"


def test_can_demote_admin_when_another_admin_remains():
    target = SimpleNamespace(role="admin")
    other = SimpleNamespace(role="admin")
    _prevent_last_admin_demotion(target, "user", [target, other])


def test_role_change_that_keeps_admin_role_is_not_blocked():
    admin = SimpleNamespace(role="admin")
    _prevent_last_admin_demotion(admin, "admin", [admin])
