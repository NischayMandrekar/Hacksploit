from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.infrastructure.database import get_db
from app.modules.products import service
from app.modules.products.schemas import ProductCreate, ProductRead

router = APIRouter(prefix="/products", tags=["products"])

Database = Annotated[Session, Depends(get_db)]


@router.post(
    "",
    response_model=ProductRead,
    status_code=status.HTTP_201_CREATED,
)
def create_product(payload: ProductCreate, db: Database) -> ProductRead:
    if not payload.name.strip():
        raise HTTPException(status_code=422, detail="Name cannot be blank")

    return service.create_product(db, payload.name, payload.price)


@router.get("", response_model=list[ProductRead])
def list_products(db: Database) -> list[ProductRead]:
    return service.list_products(db)


@router.get("/{product_id}", response_model=ProductRead)
def get_product(product_id: int, db: Database) -> ProductRead:
    product = service.get_product(db, product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_product(product_id: int, db: Database) -> None:
    product = service.get_product(db, product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")

    service.delete_product(db, product)