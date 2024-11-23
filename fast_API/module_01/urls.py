from fastapi import APIRouter

module01_router = APIRouter()

@module01_router.post('/login')
def user_login():
    return {'message': 'User login'}

@module01_router.post('/logout')
def user_logout():
    return {'message': 'User logout'}