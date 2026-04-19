from fastapi import APIRouter, HTTPException, Depends, status
from typing import List
from beanie import PydanticObjectId
from datetime import datetime

from schemas.user import (
    FavoriteCreate,
    FavoriteResponse,
    SavedSearchCreate,
    SavedSearchResponse,
    SavedSearchUpdate
)
from models.user import User
from models.favorite import Favorite
from models.saved_search import SavedSearch
from models.device import Device
from utils.auth import get_current_user

router = APIRouter()


# ==================== FAVORITES ====================

@router.get("/favorites", response_model=List[FavoriteResponse])
async def get_favorites(current_user: User = Depends(get_current_user)):
    """Get all favorites for current user"""
    
    favorites = await Favorite.find(
        Favorite.user_id == current_user.id
    ).sort("-added_at").to_list()
    
    # Populate device information
    result = []
    for fav in favorites:
        device = await Device.get(fav.device_id)
        device_data = None
        if device:
            device_data = {
                "id": str(device.id),
                "brand": device.brand,
                "model_name": device.model_name,
                "model_image": device.model_image,
                "device_type": device.device_type,
                "specs": device.specs
            }
        
        result.append(FavoriteResponse(
            id=str(fav.id),
            device_id=str(fav.device_id),
            user_id=str(fav.user_id),
            added_at=fav.added_at,
            note=fav.note or "",
            device=device_data
        ))
    
    return result


@router.post("/favorites", response_model=FavoriteResponse, status_code=status.HTTP_201_CREATED)
async def add_favorite(
    favorite_data: FavoriteCreate,
    current_user: User = Depends(get_current_user)
):
    """Add device to favorites"""
    
    try:
        device_id = PydanticObjectId(favorite_data.device_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid device ID")
    
    # Check if device exists
    device = await Device.get(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    # Check if already favorited
    existing = await Favorite.find_one(
        Favorite.user_id == current_user.id,
        Favorite.device_id == device_id
    )
    
    if existing:
        raise HTTPException(status_code=400, detail="Device already in favorites")
    
    # Create favorite
    favorite = Favorite(
        user_id=current_user.id,
        device_id=device_id,
        note=favorite_data.note or ""
    )
    
    await favorite.insert()  # type: ignore
    
    device_data = {
        "id": str(device.id),
        "brand": device.brand,
        "model_name": device.model_name,
        "model_image": device.model_image,
        "device_type": device.device_type,
        "specs": device.specs
    }
    
    return FavoriteResponse(
        id=str(favorite.id),
        device_id=str(favorite.device_id),
        user_id=str(favorite.user_id),
        added_at=favorite.added_at,
        note=favorite.note or "",
        device=device_data
    )


@router.delete("/favorites/{device_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_favorite(
    device_id: str,
    current_user: User = Depends(get_current_user)
):
    """Remove device from favorites"""
    
    try:
        device_obj_id = PydanticObjectId(device_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid device ID")
    
    favorite = await Favorite.find_one(
        Favorite.user_id == current_user.id,
        Favorite.device_id == device_obj_id
    )
    
    if not favorite:
        raise HTTPException(status_code=404, detail="Favorite not found")
    
    await favorite.delete()  # type: ignore
    
    return None


# ==================== SAVED SEARCHES ====================

@router.get("/searches", response_model=List[SavedSearchResponse])
async def get_saved_searches(current_user: User = Depends(get_current_user)):
    """Get all saved searches for current user"""
    
    searches = await SavedSearch.find(
        SavedSearch.user_id == current_user.id
    ).sort("-last_used").to_list()
    
    return [
        SavedSearchResponse(
            id=str(search.id),
            user_id=str(search.user_id),
            name=search.name,
            filters=search.filters,
            created_at=search.created_at,
            last_used=search.last_used,
            use_count=search.use_count
        )
        for search in searches
    ]


@router.post("/searches", response_model=SavedSearchResponse, status_code=status.HTTP_201_CREATED)
async def create_saved_search(
    search_data: SavedSearchCreate,
    current_user: User = Depends(get_current_user)
):
    """Create a new saved search"""
    
    search = SavedSearch(
        user_id=current_user.id,
        name=search_data.name,
        filters=search_data.filters,
        use_count=0
    )
    
    await search.insert()  # type: ignore
    
    return SavedSearchResponse(
        id=str(search.id),
        user_id=str(search.user_id),
        name=search.name,
        filters=search.filters,
        created_at=search.created_at,
        last_used=search.last_used,
        use_count=search.use_count
    )


@router.put("/searches/{search_id}", response_model=SavedSearchResponse)
async def update_saved_search(
    search_id: str,
    update_data: SavedSearchUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update a saved search"""
    
    try:
        search_obj_id = PydanticObjectId(search_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid search ID")
    
    search = await SavedSearch.find_one(
        SavedSearch.id == search_obj_id,
        SavedSearch.user_id == current_user.id
    )
    
    if not search:
        raise HTTPException(status_code=404, detail="Saved search not found")
    
    if update_data.name is not None:
        search.name = update_data.name
    
    if update_data.filters is not None:
        search.filters = update_data.filters
    
    search.last_used = datetime.utcnow()
    search.use_count += 1
    
    await search.save()  # type: ignore
    
    return SavedSearchResponse(
        id=str(search.id),
        user_id=str(search.user_id),
        name=search.name,
        filters=search.filters,
        created_at=search.created_at,
        last_used=search.last_used,
        use_count=search.use_count
    )


@router.delete("/searches/{search_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_saved_search(
    search_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a saved search"""
    
    try:
        search_obj_id = PydanticObjectId(search_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid search ID")
    
    search = await SavedSearch.find_one(
        SavedSearch.id == search_obj_id,
        SavedSearch.user_id == current_user.id
    )
    
    if not search:
        raise HTTPException(status_code=404, detail="Saved search not found")
    
    await search.delete()  # type: ignore
    
    return None
