# File: src/data/__init__.py
"""
Data management module for handling checkpoints, buffers, and potentially logs.
Uses Pydantic schemas for data structure definition.
"""
from .data_manager import DataManager
from .schemas import CheckpointData, BufferData, LoadedTrainingState # Export schemas

__all__ = [
    "DataManager",
    "CheckpointData", # Export schema
    "BufferData",     # Export schema
    "LoadedTrainingState", # Export schema
]