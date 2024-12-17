
# TODO APE
__all__ = ["AbstractCoordinate", "AbstractReferenceFrame", "_get_repr_cls", "_get_repr_classes"]

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from astropy import units as u
from astropy.utils.decorators import lazyproperty

from . import representation as r
from .angles import Angle, position_angle
from .errors import NonRotationTransformationError, NonRotationTransformationWarning
from .transformations import (
    DynamicMatrixTransform,
    StaticMatrixTransform,
    TransformGraph,
)

if TYPE_CHECKING:
    from typing import Literal

    from astropy.coordinates import BaseCoordinateFrame, Latitude, Longitude, SkyCoord
    from astropy.units import Unit


def _get_repr_cls(value):
    """
    Return a valid representation class from ``value`` or raise exception.
    """
    if value in r.REPRESENTATION_CLASSES:
        value = r.REPRESENTATION_CLASSES[value]
    elif not isinstance(value, type) or not issubclass(value, r.BaseRepresentation):
        raise ValueError(
            f"Representation is {value!r} but must be a BaseRepresentation class "
            f"or one of the string aliases {list(r.REPRESENTATION_CLASSES)}"
        )
    return value

def _get_repr_classes(base, **differentials):
    """Get valid representation and differential classes.

    Parameters
    ----------
    base : str or `~astropy.coordinates.BaseRepresentation` subclass
        class for the representation of the base coordinates.  If a string,
        it is looked up among the known representation classes.
    **differentials : dict of str or `~astropy.coordinates.BaseDifferentials`
        Keys are like for normal differentials, i.e., 's' for a first
        derivative in time, etc.  If an item is set to `None`, it will be
        guessed from the base class.

    Returns
    -------
    repr_classes : dict of subclasses
        The base class is keyed by 'base'; the others by the keys of
        ``diffferentials``.
    """
    base = _get_repr_cls(base)
    repr_classes = {"base": base}

    for name, differential_type in differentials.items():
        if differential_type == "base":
            # We don't want to fail for this case.
            differential_type = r.DIFFERENTIAL_CLASSES.get(base.get_name(), None)

        elif differential_type in r.DIFFERENTIAL_CLASSES:
            differential_type = r.DIFFERENTIAL_CLASSES[differential_type]

        elif differential_type is not None and (
            not isinstance(differential_type, type)
            or not issubclass(differential_type, r.BaseDifferential)
        ):
            raise ValueError(
                "Differential is {differential_type!r} but must be a BaseDifferential"
                f" class or one of the string aliases {list(r.DIFFERENTIAL_CLASSES)}"
            )
        repr_classes[name] = differential_type
    return repr_classes