from typing import List, Optional, Dict, Any
from datetime import datetime

from sqlmodel import SQLModel, Field, create_engine, select
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, AsyncSession
from sqlalchemy import Column, JSON, Text
from sqlalchemy.orm import sessionmaker

from config import settings

engine: Optional[AsyncEngine] = None
async_session: Optional[sessionmaker] = None


class DeviceSQL(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    brand: Optional[str] = Field(default=None, index=True)
    model_name: Optional[str] = Field(default=None, index=True)
    model_image: Optional[str] = None
    device_type: Optional[str] = Field(default=None, index=True)
    # specs and variants stored as JSON
    specs: Optional[Dict[str, Any]] = Field(sa_column=Column(JSON))
    variants: Optional[List[Dict[str, Any]]] = Field(sa_column=Column(JSON))
    scraped_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


async def init_mysql():
    """Initialize async engine for MySQL.
    Note: Table creation via sync create_all() is not called in async context.
    Ensure tables exist in MySQL before running, or they will be created on first write.
    """
    global engine, async_session
    if engine:
        return

    engine = create_async_engine(settings.mysql_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Tables must be pre-created in MySQL or created outside async context.
    # Avoid calling sync create_all() in async context (causes greenlet error).
    print("MySQL engine initialized successfully.")


async def get_devices(
    device_type: Optional[List[str]] = None,
    brand: Optional[List[str]] = None,
    search: Optional[str] = None,
    processor: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    """Return list of devices and total count matching filters"""
    assert async_session is not None, "MySQL not initialized"
    async with async_session() as session:
        stmt = select(DeviceSQL)

        # Basic filters
        if device_type:
            stmt = stmt.where(DeviceSQL.device_type.in_(device_type))
        if brand:
            stmt = stmt.where(DeviceSQL.brand.in_(brand))
        if search:
            like = f"%{search}%"
            stmt = stmt.where((DeviceSQL.model_name.ilike(like)) | (DeviceSQL.brand.ilike(like)))

        # total
        offset = (page - 1) * page_size
        results = await session.execute(stmt.offset(offset).limit(page_size))
        devices = results.scalars().all()

        # total count: run count separately
        count_stmt = select(DeviceSQL)
        if device_type:
            count_stmt = count_stmt.where(DeviceSQL.device_type.in_(device_type))
        if brand:
            count_stmt = count_stmt.where(DeviceSQL.brand.in_(brand))
        if search:
            like = f"%{search}%"
            count_stmt = count_stmt.where((DeviceSQL.model_name.ilike(like)) | (DeviceSQL.brand.ilike(like)))
        total_res = await session.execute(count_stmt)
        total_count = len(total_res.scalars().all())

        return devices, total_count


async def get_brands() -> List[str]:
    assert async_session is not None, "MySQL not initialized"
    async with async_session() as session:
        res = await session.execute(select(DeviceSQL.brand).distinct())
        brands = [r[0] for r in res.all() if r[0]]
        return brands


async def get_device_by_id(device_id: int) -> Optional[DeviceSQL]:
    assert async_session is not None, "MySQL not initialized"
    async with async_session() as session:
        result = await session.get(DeviceSQL, device_id)
        return result


async def delete_all_devices() -> None:
    assert async_session is not None, "MySQL not initialized"
    async with async_session() as session:
        await session.execute("DELETE FROM device_sql")
        await session.commit()


async def insert_devices(devices: List[Dict[str, Any]]) -> int:
    """Insert list of device dicts. Returns number inserted."""
    assert async_session is not None, "MySQL not initialized"
    objs = []
    for d in devices:
        objs.append(DeviceSQL(
            brand=d.get('brand'),
            model_name=d.get('model_name'),
            model_image=d.get('model_image'),
            device_type=d.get('device_type'),
            specs=d.get('specs'),
            variants=d.get('variants'),
            scraped_at=d.get('scraped_at'),
            updated_at=d.get('updated_at')
        ))

    async with async_session() as session:
        session.add_all(objs)
        await session.commit()
        return len(objs)
