
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

    def _infer_representation(self, representation_type, differential_type):
        if representation_type is None and differential_type is None:
            return {"base": self.default_representation, "s": self.default_differential}

        if representation_type is None:
            representation_type = self.default_representation

        if isinstance(differential_type, type) and issubclass(
            differential_type, r.BaseDifferential
        ):
            # TODO: assumes the differential class is for the velocity
            # differential
            differential_type = {"s": differential_type}

        elif isinstance(differential_type, str):
            # TODO: assumes the differential class is for the velocity
            # differential
            diff_cls = r.DIFFERENTIAL_CLASSES[differential_type]
            differential_type = {"s": diff_cls}

        elif differential_type is None:
            if representation_type == self.default_representation:
                differential_type = {"s": self.default_differential}
            else:
                differential_type = {"s": "base"}  # see set_representation_cls()

        return _get_repr_classes(representation_type, **differential_type)

    def _infer_data(self, args, copy, kwargs):
        # if not set below, this is a frame with no data
        representation_data = None
        differential_data = None

        args = list(args)  # need to be able to pop them
        if args and (isinstance(args[0], r.BaseRepresentation) or args[0] is None):
            representation_data = args.pop(0)  # This can still be None
            if len(args) > 0:
                raise TypeError(
                    "Cannot create a frame with both a representation object "
                    "and other positional arguments"
                )

            if representation_data is not None:
                diffs = representation_data.differentials
                differential_data = diffs.get("s", None)
                if (differential_data is None and len(diffs) > 0) or (
                    differential_data is not None and len(diffs) > 1
                ):
                    raise ValueError(
                        "Multiple differentials are associated with the representation"
                        " object passed in to the frame initializer. Only a single"
                        f" velocity differential is supported. Got: {diffs}"
                    )

        else:
            representation_cls = self.get_representation_cls()
            # Get any representation data passed in to the frame initializer
            # using keyword or positional arguments for the component names
            repr_kwargs = {}
            for nmkw, nmrep in self.representation_component_names.items():
                if len(args) > 0:
                    # first gather up positional args
                    repr_kwargs[nmrep] = args.pop(0)
                elif nmkw in kwargs:
                    repr_kwargs[nmrep] = kwargs.pop(nmkw)

            # special-case the Spherical->UnitSpherical if no `distance`

            if repr_kwargs:
                # TODO: determine how to get rid of the part before the "try" -
                # currently removing it has a performance regression for
                # unitspherical because of the try-related overhead.
                # Also frames have no way to indicate what the "distance" is
                if repr_kwargs.get("distance", True) is None:
                    del repr_kwargs["distance"]

                if (
                    issubclass(representation_cls, r.SphericalRepresentation)
                    and "distance" not in repr_kwargs
                ):
                    representation_cls = representation_cls._unit_representation

                try:
                    representation_data = representation_cls(copy=copy, **repr_kwargs)
                except TypeError as e:
                    # this except clause is here to make the names of the
                    # attributes more human-readable.  Without this the names
                    # come from the representation instead of the frame's
                    # attribute names.
                    try:
                        representation_data = representation_cls._unit_representation(
                            copy=copy, **repr_kwargs
                        )
                    except Exception:
                        msg = str(e)
                        names = self.get_representation_component_names()
                        for frame_name, repr_name in names.items():
                            msg = msg.replace(repr_name, frame_name)
                        msg = msg.replace("__init__()", f"{self.__class__.__name__}()")
                        e.args = (msg,)
                        raise e

            # Now we handle the Differential data:
            # Get any differential data passed in to the frame initializer
            # using keyword or positional arguments for the component names
            differential_cls = self.get_representation_cls("s")
            diff_component_names = self.get_representation_component_names("s")
            diff_kwargs = {}
            for nmkw, nmrep in diff_component_names.items():
                if len(args) > 0:
                    # first gather up positional args
                    diff_kwargs[nmrep] = args.pop(0)
                elif nmkw in kwargs:
                    diff_kwargs[nmrep] = kwargs.pop(nmkw)

            if diff_kwargs:
                if (
                    hasattr(differential_cls, "_unit_differential")
                    and "d_distance" not in diff_kwargs
                ):
                    differential_cls = differential_cls._unit_differential

                elif len(diff_kwargs) == 1 and "d_distance" in diff_kwargs:
                    differential_cls = r.RadialDifferential

                try:
                    differential_data = differential_cls(copy=copy, **diff_kwargs)
                except TypeError as e:
                    # this except clause is here to make the names of the
                    # attributes more human-readable.  Without this the names
                    # come from the representation instead of the frame's
                    # attribute names.
                    msg = str(e)
                    names = self.get_representation_component_names("s")
                    for frame_name, repr_name in names.items():
                        msg = msg.replace(repr_name, frame_name)
                    msg = msg.replace("__init__()", f"{self.__class__.__name__}()")
                    e.args = (msg,)
                    raise

        if len(args) > 0:
            raise TypeError(
                f"{type(self).__name__}.__init__ had {len(args)} remaining "
                "unhandled arguments"
            )

        if representation_data is None and differential_data is not None:
            raise ValueError(
                "Cannot pass in differential component data "
                "without positional (representation) data."
            )

        if differential_data:
            # Check that differential data provided has units compatible
            # with time-derivative of representation data.
            # NOTE: there is no dimensionless time while lengths can be
            # dimensionless (u.dimensionless_unscaled).
            for comp in representation_data.components:
                if (diff_comp := f"d_{comp}") in differential_data.components:
                    current_repr_unit = representation_data._units[comp]
                    current_diff_unit = differential_data._units[diff_comp]
                    expected_unit = current_repr_unit / u.s
                    if not current_diff_unit.is_equivalent(expected_unit):
                        for (
                            key,
                            val,
                        ) in self.get_representation_component_names().items():
                            if val == comp:
                                current_repr_name = key
                                break
                        for key, val in self.get_representation_component_names(
                            "s"
                        ).items():
                            if val == diff_comp:
                                current_diff_name = key
                                break
                        raise ValueError(
                            f'{current_repr_name} has unit "{current_repr_unit}" with'
                            f' physical type "{current_repr_unit.physical_type}", but'
                            f" {current_diff_name} has incompatible unit"
                            f' "{current_diff_unit}" with physical type'
                            f' "{current_diff_unit.physical_type}" instead of the'
                            f' expected "{(expected_unit).physical_type}".'
                        )

            representation_data = representation_data.with_differentials(
                {"s": differential_data}
            )

        return representation_data
