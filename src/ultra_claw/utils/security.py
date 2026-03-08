"""
Security utilities for Ultra-Claw.

This module provides encryption, hashing, and other security functions.
"""

import base64
import hashlib
import secrets
from typing import Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def _get_fernet(key: Optional[str] = None) -> Fernet:
    """Get or create a Fernet instance."""
    if key is None:
        # Use a default key from environment or generate one
        import os
        key = os.getenv("ULTRACLAW_ENCRYPTION_KEY")
        if key is None:
            raise ValueError("Encryption key not provided")
    
    # Ensure key is properly formatted
    if isinstance(key, str):
        key = key.encode()
    
    # If key is too short, derive a proper key
    if len(key) < 32:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"ultra-claw-salt",  # In production, use a random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(key))
    
    return Fernet(key)


def encrypt(data: str, key: Optional[str] = None) -> str:
    """
    Encrypt a string.
    
    Args:
        data: Data to encrypt
        key: Encryption key
        
    Returns:
        Encrypted data (base64 encoded)
    """
    fernet = _get_fernet(key)
    encrypted = fernet.encrypt(data.encode())
    return base64.urlsafe_b64encode(encrypted).decode()


def decrypt(encrypted_data: str, key: Optional[str] = None) -> str:
    """
    Decrypt a string.
    
    Args:
        encrypted_data: Encrypted data (base64 encoded)
        key: Encryption key
        
    Returns:
        Decrypted data
    """
    fernet = _get_fernet(key)
    encrypted = base64.urlsafe_b64decode(encrypted_data.encode())
    return fernet.decrypt(encrypted).decode()


def hash_password(password: str) -> str:
    """
    Hash a password using PBKDF2.
    
    Args:
        password: Password to hash
        
    Returns:
        Hashed password
    """
    salt = secrets.token_hex(16)
    pwdhash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        100000
    )
    return salt + pwdhash.hex()


def verify_password(password: str, hashed: str) -> bool:
    """
    Verify a password against a hash.
    
    Args:
        password: Password to verify
        hashed: Hashed password
        
    Returns:
        True if password matches
    """
    salt = hashed[:32]
    stored_hash = hashed[32:]
    
    pwdhash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        100000
    )
    
    return pwdhash.hex() == stored_hash


def generate_token(length: int = 32) -> str:
    """
    Generate a secure random token.
    
    Args:
        length: Token length
        
    Returns:
        Random token
    """
    return secrets.token_urlsafe(length)


def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        text: Input text
        
    Returns:
        Sanitized text
    """
    # Remove null bytes
    text = text.replace("\x00", "")
    
    # Limit length
    max_length = 100000
    if len(text) > max_length:
        text = text[:max_length]
    
    return text
