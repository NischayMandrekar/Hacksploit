from platform_app.auth import hash_password, normalize_email, token_hash, verify_password


def test_password_hash_uses_non_reversible_argon2id_format():
    password = "correct horse battery staple"
    encoded = hash_password(password)
    assert encoded.startswith("$argon2id$")
    assert encoded != password
    assert verify_password(password, encoded)
    assert not verify_password("wrong password", encoded)


def test_email_normalization_and_refresh_hashing():
    assert normalize_email("  Person@Example.COM ") == "person@example.com"
    assert token_hash("refresh-token") == token_hash("refresh-token")
    assert token_hash("refresh-token") != token_hash("other-token")
