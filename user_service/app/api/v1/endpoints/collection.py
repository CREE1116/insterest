from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete
from app.db.session import get_db
from app.api.deps import get_current_user_id
from app.models.models import Collection, CollectionItem
from app.schemas.schemas import CollectionCreate, CollectionRead, CollectionItemCreate
import uuid
from typing import List, Dict, Any
import httpx
from app.core.config import settings

router = APIRouter()

@router.get("/ids")
async def get_all_saved_post_ids(
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id)
):
    # 유저의 모든 컬렉션에서 post_id들만 추출
    result = await db.execute(
        select(CollectionItem.post_id)
        .join(Collection, Collection.id == CollectionItem.collection_id)
        .where(Collection.user_id == user_id)
    )
    post_ids = result.scalars().all()
    return list(set(post_ids))

@router.post("/", response_model=CollectionRead)
async def create_collection(
    col_in: CollectionCreate,
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id)
):
    # 동일한 이름의 컬렉션이 있는지 확인
    exist_check = await db.execute(
        select(Collection).where(Collection.user_id == user_id, Collection.name == col_in.name)
    )
    if exist_check.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="이미 존재하는 이름의 폴더입니다.")

    new_col = Collection(user_id=user_id, name=col_in.name)
    db.add(new_col)
    await db.commit()
    await db.refresh(new_col)
    
    # Return enriched with default values for new collection
    return CollectionRead(
        id=new_col.id,
        name=new_col.name,
        created_at=new_col.created_at,
        item_count=0,
        collection_images=[]
    )

@router.get("/", response_model=List[CollectionRead])
async def get_my_collections(
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id)
):
    # 1. Fetch collections
    result = await db.execute(select(Collection).where(Collection.user_id == user_id))
    collections = result.scalars().all()
    
    if not collections:
        return []

    # 2. For each collection, get item count and recent 4 post_ids
    enriched_cols = []
    all_post_ids = set()
    col_data_map = {}

    for col in collections:
        # Count items
        count_res = await db.execute(
            select(func.count(CollectionItem.id)).where(CollectionItem.collection_id == col.id)
        )
        item_count = count_res.scalar() or 0
        
        # Get up to 4 recent post_ids for collage
        items_res = await db.execute(
            select(CollectionItem.post_id)
            .where(CollectionItem.collection_id == col.id)
            .order_by(CollectionItem.created_at.desc())
            .limit(4)
        )
        p_ids = items_res.scalars().all()
        for pid in p_ids:
            all_post_ids.add(pid)
        
        col_data_map[col.id] = {
            "item_count": item_count,
            "post_ids": p_ids
        }

    # 3. Fetch image URLs from upload-service for these post_ids
    image_map = {}
    if all_post_ids:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client_httpx:
                # Internal URL must match upload-service's full prefix path
                res = await client_httpx.post(
                    f"{settings.UPLOAD_SERVICE_URL}/api/v1/upload/content/batch",
                    json={"post_ids": [str(pid) for pid in list(all_post_ids)]}
                )
                if res.status_code == 200:
                    posts_data = res.json()
                    for p in posts_data:
                        # Find first image media
                        media_list = p.get("content", {}).get("media_list", [])
                        img_url = next((m["url"] for m in media_list if m["type"] == "image"), None)
                        if img_url:
                            image_map[uuid.UUID(p["id"])] = img_url
        except Exception as e:
            print(f"Failed to fetch images from upload-service: {e}")

    # 4. Final assembly
    for col in collections:
        data = col_data_map[col.id]
        collage_images = [image_map[pid] for pid in data["post_ids"] if pid in image_map]
        
        enriched_cols.append(CollectionRead(
            id=col.id,
            name=col.name,
            created_at=col.created_at,
            item_count=data["item_count"],
            collection_images=collage_images
        ))

    return enriched_cols

@router.get("/{collection_id}/items")
async def get_collection_items(
    collection_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id)
):
    # 권한 확인
    col_result = await db.execute(select(Collection).where(Collection.id == collection_id, Collection.user_id == user_id))
    if not col_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Collection not found")
        
    result = await db.execute(select(CollectionItem).where(CollectionItem.collection_id == collection_id))
    return result.scalars().all()

@router.post("/{collection_id}/items")
async def add_item_to_collection(
    collection_id: uuid.UUID,
    item_in: CollectionItemCreate,
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id)
):
    # 권한 확인
    result = await db.execute(select(Collection).where(Collection.id == collection_id, Collection.user_id == user_id))
    col = result.scalar_one_or_none()
    if not col:
        raise HTTPException(status_code=404, detail="Collection not found or access denied")
        
    # 이미 있는지 확인
    exist_result = await db.execute(select(CollectionItem).where(
        CollectionItem.collection_id == collection_id, 
        CollectionItem.post_id == item_in.post_id
    ))
    if exist_result.scalar_one_or_none():
        return {"status": "already_exists"}
        
    new_item = CollectionItem(collection_id=collection_id, post_id=item_in.post_id)
    db.add(new_item)
    await db.commit()
    return {"status": "added"}

@router.delete("/items/{post_id}")
async def remove_item_from_all_collections(
    post_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id)
):
    # 1. 유저의 컬렉션 ID 목록 조회
    col_query = select(Collection.id).where(Collection.user_id == user_id)
    col_result = await db.execute(col_query)
    col_ids = col_result.scalars().all()
    
    if not col_ids:
        return {"status": "no_collections"}

    # 2. 해당 컬렉션들에 속한 아이템 삭제
    del_query = delete(CollectionItem).where(
        CollectionItem.collection_id.in_(col_ids),
        CollectionItem.post_id == post_id
    )
    await db.execute(del_query)
    await db.commit()
    
    return {"status": "removed"}
    
@router.patch("/{collection_id}", response_model=CollectionRead)
async def update_collection(
    collection_id: uuid.UUID,
    col_in: CollectionCreate,
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id)
):
    result = await db.execute(select(Collection).where(Collection.id == collection_id, Collection.user_id == user_id))
    col = result.scalar_one_or_none()
    if not col:
        raise HTTPException(status_code=404, detail="Collection not found")
        
    col.name = col_in.name
    await db.commit()
    await db.refresh(col)
    
    # Return enriched
    return await get_my_collections_single(db, user_id, col.id)

@router.delete("/{collection_id}")
async def delete_collection(
    collection_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID = Depends(get_current_user_id)
):
    result = await db.execute(select(Collection).where(Collection.id == collection_id, Collection.user_id == user_id))
    col = result.scalar_one_or_none()
    if not col:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    await db.execute(delete(CollectionItem).where(CollectionItem.collection_id == collection_id))
    await db.delete(col)
    await db.commit()
    
    return {"status": "deleted"}

async def get_my_collections_single(db: AsyncSession, user_id: uuid.UUID, collection_id: uuid.UUID):
    # Helper for patch to return enriched data
    result = await db.execute(select(Collection).where(Collection.id == collection_id))
    col = result.scalar_one()
    
    count_res = await db.execute(select(func.count(CollectionItem.id)).where(CollectionItem.collection_id == col.id))
    item_count = count_res.scalar() or 0
    
    return CollectionRead(
        id=col.id,
        name=col.name,
        created_at=col.created_at,
        item_count=item_count,
        collection_images=[] # Simple fallback for patch
    )
