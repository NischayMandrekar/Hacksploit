from decimal import Decimal

from sqlalchemy.orm import Session

from app.modules.products import repository
from app.modules.products.models import Product


def create_product(db: Session, name: str, price: Decimal) -> Product:
    return repository.create_product(db, name.strip(), price)


def list_products(db: Session) -> list[Product]:
    return repository.list_products(db)


def get_product(db: Session, product_id: int) -> Product | None:
    return repository.get_product(db, product_id)


def delete_product(db: Session, product: Product) -> None:
    repository.delete_product(db, product)