"""Site heat model v2: predicts where it is hot (Landsat LST) from land cover plus Sentinel-2 indices.

It describes sites; it is not a what-if model for interventions (indices cannot be edited by a redesign).
"""

from .predict import SiteModel, load_site_model

__all__ = ["SiteModel", "load_site_model"]
