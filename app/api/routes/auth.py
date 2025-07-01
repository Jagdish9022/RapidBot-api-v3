from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from app.db.models import UserCreate, UserLogin, User, Token
from app.auth.auth import (
    get_password_hash, verify_password, create_access_token,
    get_current_active_user, ACCESS_TOKEN_EXPIRE_MINUTES
)
from app.db.mysql import get_db
import logging
from datetime import datetime, timedelta
import uuid
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["authentication"])

@router.post("/signup", response_model=User)
async def signup(user: UserCreate, db = Depends(get_db)):
    """Create a new user account."""
    try:
        logger.info(f"Signup attempt for email: {user.email}")
        
        # Check if user already exists
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (user.email,))
        existing_user = cursor.fetchone()
        cursor.close()
        
        if existing_user:
            logger.warning(f"Signup failed: Email already registered: {user.email}")
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create new user
        user_id = str(uuid.uuid4())
        hashed_password = get_password_hash(user.password)
        
        # Insert new user
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO users (id, email, username, password_hash)
            VALUES (%s, %s, %s, %s)
        """, (
            user_id,
            user.email,
            user.username,
            hashed_password
        ))
        db.commit()
        cursor.close()
        
        logger.info(f"User created successfully: {user.email}")
        return {
            'id': user_id,
            'email': user.email,
            'username': user.username,
            'created_at': datetime.now(),
            'is_active': True
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in signup: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="An error occurred during signup. Please try again."
        )

@router.post("/login", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db = Depends(get_db)
):
    """Login endpoint for OAuth2 password flow"""
    try:
        logger.info(f"Login attempt for username: {form_data.username}")
        
        # Find user
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (form_data.username,))
        user = cursor.fetchone()
        cursor.close()
        
        if not user:
            logger.warning(f"Login failed: User not found with email {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify password
        if not verify_password(form_data.password, user['password_hash']):
            logger.warning(f"Login failed: Invalid password for user {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user['id']}, expires_delta=access_token_expires
        )
        
        logger.info(f"Login successful for user {form_data.username}")
        return {"access_token": access_token, "token_type": "bearer"}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in login: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during login. Please try again."
        )

@router.post("/login/json", response_model=Token)
async def login_json(user_data: UserLogin, db = Depends(get_db)):
    """Login endpoint for JSON data"""
    try:
        logger.info(f"Login attempt for email: {user_data.email}")
        
        # Find user
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (user_data.email,))
        user = cursor.fetchone()
        cursor.close()
        
        if not user:
            logger.warning(f"Login failed: User not found with email {user_data.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify password
        if not verify_password(user_data.password, user['password_hash']):
            logger.warning(f"Login failed: Invalid password for user {user_data.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user['id']}, expires_delta=access_token_expires
        )
        
        logger.info(f"Login successful for user {user_data.email}")
        return {"access_token": access_token, "token_type": "bearer"}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in login: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during login. Please try again."
        )
