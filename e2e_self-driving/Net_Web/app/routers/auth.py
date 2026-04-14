"""注册 / 登录 / 当前用户。"""

from fastapi import APIRouter, HTTPException, status

from app import database as db
from app.deps import CurrentUser
from app.schemas import LoginBody, RegisterBody, TokenUserResponse, UserOut
from app.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenUserResponse)
def register(body: RegisterBody):
    if db.get_user_by_username(body.username):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户名已存在")
    pwd_hash = hash_password(body.password)
    user = db.create_user(body.username, pwd_hash, body.email)
    token = create_access_token(subject=user["id"], extra={"username": user["username"]})
    return TokenUserResponse(
        token=token,
        user=UserOut(id=user["id"], username=user["username"], email=user.get("email")),
    )


@router.post("/login", response_model=TokenUserResponse)
def login(body: LoginBody):
    row = db.get_user_by_username(body.username)
    if row is None or not verify_password(body.password, row["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    token = create_access_token(subject=row["id"], extra={"username": row["username"]})
    return TokenUserResponse(
        token=token,
        user=UserOut(id=row["id"], username=row["username"], email=row["email"]),
    )


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser):
    return UserOut(id=user["id"], username=user["username"], email=user.get("email"))
