"""Stable exports for the frozen GL-RBF core and flow-matching wrapper."""

from Model import ConditionalPointHybridLocalGlobalRBF, PointCloudFFM

GLRbfCore = ConditionalPointHybridLocalGlobalRBF

__all__ = ["ConditionalPointHybridLocalGlobalRBF", "GLRbfCore", "PointCloudFFM"]
