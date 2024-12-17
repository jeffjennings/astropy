
# TODO APE: update
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

class AbstractCoordinate(MaskableShapedLikeNDArray):
    # TODO APE: docstring 

    default_representation = None
    default_differential = None

    # Specifies special names and units for representation and differential
    # attributes.
    frame_specific_representation_info = {}

    def __init_subclass__(cls, **kwargs):
        # We first check for explicitly set values for these:
        default_repr = getattr(cls, "default_representation", None)
        default_diff = getattr(cls, "default_differential", None)
        repr_info = getattr(cls, "frame_specific_representation_info", None)
        # Then, to make sure this works for subclasses-of-subclasses, we also
        # have to check for cases where the attribute names have already been
        # replaced by underscore-prefaced equivalents by the logic below:
        if default_repr is None or isinstance(default_repr, property):
            default_repr = getattr(cls, "_default_representation", None)

        if default_diff is None or isinstance(default_diff, property):
            default_diff = getattr(cls, "_default_differential", None)

        if repr_info is None or isinstance(repr_info, property):
            repr_info = getattr(cls, "_frame_specific_representation_info", None)

        repr_info = cls._infer_repr_info(repr_info)

        # Make read-only properties for the frame class attributes that should
        # be read-only to make them immutable after creation.
        # We copy attributes instead of linking to make sure there's no
        # accidental cross-talk between classes
        cls._create_readonly_property(
            "default_representation",
            default_repr,
            "Default representation for position data",
        )
        cls._create_readonly_property(
            "default_differential",
            default_diff,
            "Default representation for differential data (e.g., velocity)",
        )
        cls._create_readonly_property(
            "frame_specific_representation_info",
            copy.deepcopy(repr_info),
            "Mapping for frame-specific component names",
        )

        # TODO APE: keep?
        cls._frame_class_cache = {} 


    # TODO APE: update args?
    def __init__(
        self,
        *args,
        copy=True,
        representation_type=None,
        differential_type=None,
        **kwargs,
    ):

        self._representation = self._infer_representation(
            representation_type, differential_type
        )
        # TODO APE: can we simplify this?
        data = self._infer_data(args, copy, kwargs)  # possibly None. 

        if copy:
            data = data.copy()
        self._data = data

        # The logic of this block is not related to the previous one
        if self.has_data:
            # This makes the cache keys backwards-compatible, but also adds
            # support for having differentials attached to the frame data
            # representation object.
            if "s" in self._data.differentials:
                # TODO: assumes a velocity unit differential
                key = (
                    self._data.__class__.__name__,
                    self._data.differentials["s"].__class__.__name__,
                    False,
                )
            else:
                key = (self._data.__class__.__name__, False)

            # Set up representation cache.
            self.cache["representation"][key] = self._data


