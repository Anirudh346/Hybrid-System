from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Dict, Any
from beanie import PydanticObjectId

from schemas.comparison import (
    ComparisonRequest,
    ComparisonResponse,
    ComparisonDeviceResponse,
    ComparisonListResponse
)
from models.comparison import Comparison
from models.device import Device
from models.user import User
from utils.auth import get_current_user

router = APIRouter()


@router.post("", response_model=ComparisonResponse, status_code=status.HTTP_201_CREATED)
async def create_comparison(
    request: ComparisonRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new device comparison"""
    
    if len(request.device_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 devices required for comparison")
    
    if len(request.device_ids) > 4:
        raise HTTPException(status_code=400, detail="Maximum 4 devices can be compared")
    
    # Convert to ObjectIds and verify devices exist
    device_obj_ids = []
    devices = []
    
    for device_id in request.device_ids:
        try:
            obj_id = PydanticObjectId(device_id)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid device ID: {device_id}")
        
        device = await Device.get(obj_id)
        if not device:
            raise HTTPException(status_code=404, detail=f"Device not found: {device_id}")
        
        device_obj_ids.append(obj_id)
        devices.append(device)
    
    # Create comparison
    comparison = Comparison(
        user_id=current_user.id,
        device_ids=device_obj_ids,
        name=request.name or f"Comparison - {devices[0].brand} vs {devices[1].brand}"
    )
    
    await comparison.insert()
    
    # Build comparison table
    comparison_table = _build_comparison_table(devices)
    
    device_responses = [
        ComparisonDeviceResponse(
            id=str(device.id),
            brand=device.brand,
            model_name=device.model_name,
            model_image=device.model_image,
            device_type=device.device_type,
            specs=device.specs
        )
        for device in devices
    ]
    
    return ComparisonResponse(
        id=str(comparison.id),
        devices=device_responses,
        name=comparison.name,
        created_at=comparison.created_at.isoformat(),
        comparison_table=comparison_table
    )


@router.get("", response_model=ComparisonListResponse)
async def get_comparisons(current_user: User = Depends(get_current_user)):
    """Get all comparisons for current user"""
    
    comparisons = await Comparison.find(
        Comparison.user_id == current_user.id
    ).sort(-Comparison.created_at).to_list()
    
    result = []
    
    for comp in comparisons:
        # Fetch devices
        devices = []
        for device_id in comp.device_ids:
            device = await Device.get(device_id)
            if device:
                devices.append(device)
        
        if devices:
            comparison_table = _build_comparison_table(devices)
            
            device_responses = [
                ComparisonDeviceResponse(
                    id=str(device.id),
                    brand=device.brand,
                    model_name=device.model_name,
                    model_image=device.model_image,
                    device_type=device.device_type,
                    specs=device.specs
                )
                for device in devices
            ]
            
            result.append(
                ComparisonResponse(
                    id=str(comp.id),
                    devices=device_responses,
                    name=comp.name,
                    created_at=comp.created_at.isoformat(),
                    comparison_table=comparison_table
                )
            )
    
    return ComparisonListResponse(comparisons=result)


@router.get("/{comparison_id}", response_model=ComparisonResponse)
async def get_comparison(
    comparison_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get a specific comparison"""
    
    try:
        comp_obj_id = PydanticObjectId(comparison_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid comparison ID")
    
    comparison = await Comparison.find_one(
        Comparison.id == comp_obj_id,
        Comparison.user_id == current_user.id
    )
    
    if not comparison:
        raise HTTPException(status_code=404, detail="Comparison not found")
    
    # Fetch devices
    devices = []
    for device_id in comparison.device_ids:
        device = await Device.get(device_id)
        if device:
            devices.append(device)
    
    if not devices:
        raise HTTPException(status_code=404, detail="No devices found in comparison")
    
    comparison_table = _build_comparison_table(devices)
    
    device_responses = [
        ComparisonDeviceResponse(
            id=str(device.id),
            brand=device.brand,
            model_name=device.model_name,
            model_image=device.model_image,
            device_type=device.device_type,
            specs=device.specs
        )
        for device in devices
    ]
    
    return ComparisonResponse(
        id=str(comparison.id),
        devices=device_responses,
        name=comparison.name,
        created_at=comparison.created_at.isoformat(),
        comparison_table=comparison_table
    )


@router.delete("/{comparison_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_comparison(
    comparison_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a comparison"""
    
    try:
        comp_obj_id = PydanticObjectId(comparison_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid comparison ID")
    
    comparison = await Comparison.find_one(
        Comparison.id == comp_obj_id,
        Comparison.user_id == current_user.id
    )
    
    if not comparison:
        raise HTTPException(status_code=404, detail="Comparison not found")
    
    await comparison.delete()
    
    return None


def _build_comparison_table(devices: List[Device]) -> Dict[str, List[Any]]:
    """Build side-by-side comparison table"""
    
    # Collect all unique spec keys
    all_spec_keys = set()
    for device in devices:
        all_spec_keys.update(device.specs.keys())
    
    # Important specs to show first
    priority_specs = [
        'OS', 'Chipset', 'CPU', 'GPU',
        'Display', 'Size', 'Resolution',
        'Internal', 'RAM',
        'Main Camera', 'Selfie camera',
        'Battery', 'Charging',
        'Price', 'Announced', 'Status'
    ]
    
    # Build comparison table
    comparison_table = {}
    
    # Add priority specs first
    for spec_key in priority_specs:
        if spec_key in all_spec_keys:
            comparison_table[spec_key] = [
                device.specs.get(spec_key, 'N/A') for device in devices
            ]
    
    # Add remaining specs
    for spec_key in sorted(all_spec_keys):
        if spec_key not in comparison_table:
            comparison_table[spec_key] = [
                device.specs.get(spec_key, 'N/A') for device in devices
            ]
    
    return comparison_table
