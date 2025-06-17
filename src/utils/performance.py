"""Performance monitoring utilities for pipeline components."""

import time
from typing import Dict, List

import numpy as np


class PerformanceMonitor:
    """
    Monitor and report performance metrics for pipeline stages.

    Tracks execution times for different pipeline stages and provides
    methods to calculate FPS and average execution times.
    """

    def __init__(self, window_size: int = 30) -> None:
        """
        Initialize the performance monitor.

        Args:
            window_size: Number of frames for moving average calculation.
        """
        self.stages = {}
        self.window_size = window_size

    def start_timer(self) -> float:
        """Start timing a stage."""
        return time.perf_counter()

    def end_timer(self, stage_name: str, start_time: float) -> float:
        """
        End timing for a stage and record the duration.

        Args:
            stage_name: Name of the pipeline stage.
            start_time: Start time returned by start_timer.

        Returns:
            float: Elapsed time in seconds.
        """
        elapsed = time.perf_counter() - start_time
        if stage_name not in self.stages:
            self.stages[stage_name] = []

        self.stages[stage_name].append(elapsed)

        # Keep only the most recent window_size measurements
        if len(self.stages[stage_name]) > self.window_size:
            self.stages[stage_name] = self.stages[stage_name][-self.window_size :]

        return elapsed

    def get_fps(self, stage_name: str) -> int:
        """Get the FPS for a specific stage."""
        if stage_name not in self.stages or not self.stages[stage_name]:
            return 0

        avg_time = np.mean(self.stages[stage_name])
        return int(1 / avg_time) if avg_time > 0 else 0

    def get_average_time(self, stage_name: str) -> float:
        """Get the average execution time for a stage in milliseconds."""
        if stage_name not in self.stages or not self.stages[stage_name]:
            return 0.0

        avg_time = np.mean(self.stages[stage_name])
        return avg_time * 1000  # Convert to milliseconds

    def get_all_fps(self) -> Dict[str, int]:
        """Get FPS for all monitored stages."""
        return {stage: self.get_fps(stage) for stage in self.stages}

    def get_all_times(self) -> Dict[str, float]:
        """Get average execution times for all monitored stages in ms."""
        return {stage: self.get_average_time(stage) for stage in self.stages}

    def reset(self) -> None:
        """Reset all performance data."""
        self.stages = {}
