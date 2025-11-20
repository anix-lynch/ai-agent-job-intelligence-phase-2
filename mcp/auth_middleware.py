"""
Simple auth middleware for MCP server
For public demo, all requests are allowed
"""

from fastapi import Request, HTTPException
from typing import Callable

async def require_auth(request: Request):
    """No-op auth for public demo"""
    return True

async def is_owner(request: Request):
    """Check if request is from owner (always True for demo)"""
    return True

async def require_owner(request: Request):
    """Require owner access (no-op for demo)"""
    return True
