"""
Author: B. Chen

Model implementations module.
Auto-registration trigger for all model implementations.
"""


def register_all_models():
    """Import all implementations to trigger @register_model decorators."""
    from .sam_vit_b import wrapper
    from .spatial_attention import wrapper


# Auto-trigger registration on module import
register_all_models()