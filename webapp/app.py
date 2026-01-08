# Main app module for webapp - imports from __init__.py
from . import app, server

# This file maintains compatibility with existing import patterns
# while the actual app initialization is in __init__.py following Flask best practices
