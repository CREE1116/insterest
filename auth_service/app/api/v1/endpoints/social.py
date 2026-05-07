from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.config import settings
from app.core import security
from app.db.session import get_db
from app.models.user import User

router = APIRouter()
oauth = OAuth()

oauth.register(
    name='google',
    client_id=settings.GOOGLE_CLIENT_ID,
    client_secret=settings.GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

@router.get("/login/google")
async def google_login(request: Request):
    """Google 로그인 페이지로 리다이렉트합니다."""
    redirect_uri = request.url_for('auth_google')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@router.get("/auth/google")
async def auth_google(request: Request, db: AsyncSession = Depends(get_db)):
    """Google 로그인 콜백을 처리합니다."""
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get('userinfo')
    
    if not user_info:
        raise HTTPException(status_code=400, detail="Google 인증에 실패했습니다.")
    
    email = user_info.get('email')
    google_id = user_info.get('sub')
    
    # 기존 유저 확인 또는 생성
    result = await db.execute(select(User).filter(User.email == email))
    user = result.scalars().first()
    
    if not user:
        user = User(
            email=email,
            full_name=user_info.get('name'),
            is_active=True,
            is_verified=True,  # 소셜은 기본적으로 인증된 것으로 간주
            social_provider='google',
            social_id=google_id
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    
    access_token = security.create_access_token(subject=str(user.id))
    
    # 리다이렉트 응답 생성
    response = RedirectResponse(url="/")
    
    # 쿠키 설정
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=False,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        path="/",
    )
    
    return response
