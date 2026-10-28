from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.products.models import Product


def create_product(
    db: Session, name: str, price
) -> Product:
    product = Product(name=name, price=price)
    db.add(product)
    db.commit()
    db.refresh(product)
    return product


def list_products(db: Session) -> list[Product]:
    statement = select(Product).order_by(Product.id)
    return list(db.scalars(statement).all())


def get_product(db: Session, product_id: int) -> Product | None:
    return db.get(Product, product_id)


def delete_product(db: Session, product: Product) -> None:
    db.delete(product)
    db.commit()