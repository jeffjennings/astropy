# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module defines custom errors and exceptions used in astropy.coordinates."""

__all__ = [
    "ConvertError",
    "NonRotationTransformationError",
    "NonRotationTransformationWarning",
    "UnknownSiteException",
]

from typing import TYPE_CHECKING

from astropy.utils.exceptions import AstropyUserWarning

if TYPE_CHECKING:
    from astropy.coordinates import BaseCoordinateFrame


def _frame_repr_without_data(frame) -> str:
    """Return a repr of *frame* without coordinate data.

    Handles both `~astropy.coordinates.BaseCoordinateFrame` instances (which have
    ``replicate_without_data``) and data-less ``BaseFrame`` subclasses (which do
    not).  For the latter, the corresponding BaseCoordinateFrame subclass is
    and used so that the repr format is consistent (e.g. ``<GCRS Frame ...>`` rather than
    ``<GCRSFrame Frame ...>``).
    """
    if hasattr(frame, "replicate_without_data"):
        return str(frame.replicate_without_data())
    # TODO: APE23: simplify when BaseCoordinateFrame deprecated
    from .baseframe import BaseCoordinateFrame  # local import to avoid circularity

    fa = {k: getattr(frame, k) for k in type(frame).frame_attributes}
    for sub in type(frame).__subclasses__():
        if issubclass(sub, BaseCoordinateFrame):
            try:
                return str(sub(**fa).replicate_without_data())
            except Exception:
                break
    # Ultimate fallback: just use the frame's own repr.
    return repr(frame)


# TODO: consider if this should be used to `units`?
class UnitsError(ValueError):
    """
    Raised if units are missing or invalid.
    """


class ConvertError(Exception):
    """
    Raised if a coordinate system cannot be converted to another.
    """


class NonRotationTransformationError(ValueError):
    """
    Raised for transformations that are not simple rotations. Such
    transformations can change the angular separation between coordinates
    depending on its direction.
    """

    def __init__(
        self, frame_to: "BaseCoordinateFrame", frame_from: "BaseCoordinateFrame"
    ) -> None:
        self.frame_to = frame_to
        self.frame_from = frame_from

    def __str__(self) -> str:
        return (
            "refusing to transform other coordinates from "
            f"{_frame_repr_without_data(self.frame_from)} to "
            f"{_frame_repr_without_data(self.frame_to)} because angular separation "
            "can depend on the direction of the transformation"
        )


class UnknownSiteException(KeyError):
    def __init__(self, site, attribute, close_names=None):
        self.site = site
        self.attribute = attribute
        self.close_names = close_names

    def __str__(self) -> str:
        msg = (
            f"Site {self.site!r} not in database. Use {self.attribute} to see "
            f"available sites. If {self.site!r} exists in the online astropy-data "
            "repository, use the 'refresh_cache=True' option to download the latest "
            "version."
        )
        if self.close_names:
            msg += f" Did you mean one of: {', '.join(map(repr, self.close_names))}?"
        return msg


class NonRotationTransformationWarning(AstropyUserWarning):
    """
    Emitted for transformations that are not simple rotations. Such
    transformations can change the angular separation between coordinates
    depending on its direction.
    """

    def __init__(
        self, frame_to: "BaseCoordinateFrame", frame_from: "BaseCoordinateFrame"
    ) -> None:
        self.frame_to = frame_to
        self.frame_from = frame_from

    def __str__(self) -> str:
        return (
            "transforming other coordinates from "
            f"{_frame_repr_without_data(self.frame_from)} to "
            f"{_frame_repr_without_data(self.frame_to)}. Angular separation can depend "
            "on the direction of the transformation."
        )
