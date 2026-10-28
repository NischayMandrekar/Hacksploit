from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoints():
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200


def test_product_lifecycle():
    # CREATE
    response = client.post(
        "/products",
        json={"name": "Test Keyboard", "price": "2499.00"},
    )
    assert response.status_code == 201

    product = response.json()
    product_id = product["id"]

    assert product["name"] == "Test Keyboard"
    assert product["price"] == "2499.00"

    # READ ONE
    response = client.get(f"/products/{product_id}")
    assert response.status_code == 200
    assert response.json()["id"] == product_id

    # READ ALL
    response = client.get("/products")
    assert response.status_code == 200
    assert any(p["id"] == product_id for p in response.json())

    # DELETE
    response = client.delete(f"/products/{product_id}")
    assert response.status_code == 204

    # CONFIRM DELETION
    response = client.get(f"/products/{product_id}")
    assert response.status_code == 404


def test_invalid_product_is_rejected():
    response = client.post(
        "/products",
        json={"name": "", "price": "2499.00"},
    )
    assert response.status_code == 422

    response = client.post(
        "/products",
        json={"name": "Invalid Price", "price": "-10.00"},
    )
    assert response.status_code == 422