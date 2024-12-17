
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


    @classmethod
    def _infer_repr_info(cls, repr_info):
        # Unless overridden via `frame_specific_representation_info`, velocity
        # name defaults are (see also docstring for BaseCoordinateFrame):
        #   * ``pm_{lon}_cos{lat}``, ``pm_{lat}`` for
        #     `SphericalCosLatDifferential` proper motion components
        #   * ``pm_{lon}``, ``pm_{lat}`` for `SphericalDifferential` proper
        #     motion components
        #   * ``radial_velocity`` for any `d_distance` component
        #   * ``v_{x,y,z}`` for `CartesianDifferential` velocity components
        # where `{lon}` and `{lat}` are the frame names of the angular
        # components.
        if repr_info is None:
            repr_info = {}

        # the tuple() call below is necessary because if it is not there,
        # the iteration proceeds in a difficult-to-predict manner in the
        # case that one of the class objects hash is such that it gets
        # revisited by the iteration.  The tuple() call prevents this by
        # making the items iterated over fixed regardless of how the dict
        # changes
        for cls_or_name in tuple(repr_info.keys()):
            if isinstance(cls_or_name, str):
                # TODO: this provides a layer of backwards compatibility in
                # case the key is a string, but now we want explicit classes.
                repr_info[_get_repr_cls(cls_or_name)] = repr_info.pop(cls_or_name)

        # The default spherical names are 'lon' and 'lat'
        sph_repr = repr_info.setdefault(
            r.SphericalRepresentation,
            [RepresentationMapping("lon", "lon"), RepresentationMapping("lat", "lat")],
        )

        sph_component_map = {m.reprname: m.framename for m in sph_repr}
        lon = sph_component_map["lon"]
        lat = sph_component_map["lat"]

        ang_v_unit = u.mas / u.yr
        lin_v_unit = u.km / u.s

        sph_coslat_diff = repr_info.setdefault(
            r.SphericalCosLatDifferential,
            [
                RepresentationMapping("d_lon_coslat", f"pm_{lon}_cos{lat}", ang_v_unit),
                RepresentationMapping("d_lat", f"pm_{lat}", ang_v_unit),
                RepresentationMapping("d_distance", "radial_velocity", lin_v_unit),
            ],
        )
        sph_diff = repr_info.setdefault(
            r.SphericalDifferential,
            [
                RepresentationMapping("d_lon", f"pm_{lon}", ang_v_unit),
                RepresentationMapping("d_lat", f"pm_{lat}", ang_v_unit),
                RepresentationMapping("d_distance", "radial_velocity", lin_v_unit),
            ],
        )
        repr_info.setdefault(
            r.RadialDifferential,
            [RepresentationMapping("d_distance", "radial_velocity", lin_v_unit)],
        )
        repr_info.setdefault(
            r.CartesianDifferential,
            [RepresentationMapping(f"d_{c}", f"v_{c}", lin_v_unit) for c in "xyz"],
        )

        # Unit* classes should follow the same naming conventions
        # TODO: this adds some unnecessary mappings for the Unit classes, so
        # this could be cleaned up, but in practice doesn't seem to have any
        # negative side effects
        repr_info.setdefault(r.UnitSphericalRepresentation, sph_repr)
        repr_info.setdefault(r.UnitSphericalCosLatDifferential, sph_coslat_diff)
        repr_info.setdefault(r.UnitSphericalDifferential, sph_diff)

        return repr_info

    @lazyproperty
    def cache(self):
        """Cache for this frame, a dict.

        It stores anything that should be computed from the coordinate data (*not* from
        the frame attributes). This can be used in functions to store anything that
        might be expensive to compute but might be re-used by some other function.
        E.g.::

            if 'user_data' in myframe.cache:
                data = myframe.cache['user_data']
            else:
                myframe.cache['user_data'] = data = expensive_func(myframe.lat)

        If in-place modifications are made to the frame data, the cache should
        be cleared::

            myframe.cache.clear()

        """
        return defaultdict(dict)
    
    @property
    def data(self):
        """
        The coordinate data for this object.  If this frame has no data, an
        `ValueError` will be raised.  Use `has_data` to
        check if data is present on this frame object.
        """
        if self._data is None:
            raise ValueError(
                f'The frame object "{self!r}" does not have associated data'
            )
        return self._data

    @property
    def has_data(self):
        """
        True if this frame has `data`, False otherwise.
        """
        return self._data is not None
    
    @property
    def size(self):
        return self.data.size

    def __bool__(self):
        return self.has_data and self.size > 0

    @property
    def shape(self):
        return self._shape

    def get_representation_cls(self, which="base"):
        """The class used for part of this frame's data.

        Parameters
        ----------
        which : ('base', 's', `None`)
            The class of which part to return.  'base' means the class used to
            represent the coordinates; 's' the first derivative to time, i.e.,
            the class representing the proper motion and/or radial velocity.
            If `None`, return a dict with both.

        Returns
        -------
        representation : `~astropy.coordinates.BaseRepresentation` or `~astropy.coordinates.BaseDifferential`.
        """
        return self._representation if which is None else self._representation[which]

    def set_representation_cls(self, base=None, s="base"):
        """Set representation and/or differential class for this frame's data.

        Parameters
        ----------
        base : str, `~astropy.coordinates.BaseRepresentation` subclass, optional
            The name or subclass to use to represent the coordinate data.
        s : `~astropy.coordinates.BaseDifferential` subclass, optional
            The differential subclass to use to represent any velocities,
            such as proper motion and radial velocity.  If equal to 'base',
            which is the default, it will be inferred from the representation.
            If `None`, the representation will drop any differentials.
        """
        if base is None:
            base = self._representation["base"]
        self._representation = _get_repr_classes(base=base, s=s)

    representation_type = property(
        fget=get_representation_cls,
        fset=set_representation_cls,
        doc="""The representation class used for this frame's data.

        This will be a subclass from `~astropy.coordinates.BaseRepresentation`.
        Can also be *set* using the string name of the representation. If you
        wish to set an explicit differential class (rather than have it be
        inferred), use the ``set_representation_cls`` method.
        """,
    )

    @property
    def differential_type(self):
        """
        The differential used for this frame's data.

        This will be a subclass from `~astropy.coordinates.BaseDifferential`.
        For simultaneous setting of representation and differentials, see the
        ``set_representation_cls`` method.
        """
        return self.get_representation_cls("s")

    @differential_type.setter
    def differential_type(self, value):
        self.set_representation_cls(s=value)

    @classmethod
    def _get_representation_info(cls):
        # This exists as a class method only to support handling frame inputs
        # without units, which are deprecated and will be removed.  This can be
        # moved into the representation_info property at that time.
        # note that if so moved, the cache should be acceessed as
        # self.__class__._frame_class_cache

        if (
            cls._frame_class_cache.get("last_reprdiff_hash", None)
            != r.get_reprdiff_cls_hash()
        ):
            repr_attrs = {}
            for repr_diff_cls in list(r.REPRESENTATION_CLASSES.values()) + list(
                r.DIFFERENTIAL_CLASSES.values()
            ):
                repr_attrs[repr_diff_cls] = {"names": [], "units": []}
                for c, c_cls in repr_diff_cls.attr_classes.items():
                    repr_attrs[repr_diff_cls]["names"].append(c)
                    rec_unit = u.deg if issubclass(c_cls, Angle) else None
                    repr_attrs[repr_diff_cls]["units"].append(rec_unit)

            for (
                repr_diff_cls,
                mappings,
            ) in cls._frame_specific_representation_info.items():
                # take the 'names' and 'units' tuples from repr_attrs,
                # and then use the RepresentationMapping objects
                # to update as needed for this frame.
                nms = repr_attrs[repr_diff_cls]["names"]
                uns = repr_attrs[repr_diff_cls]["units"]
                comptomap = {m.reprname: m for m in mappings}
                for i, c in enumerate(repr_diff_cls.attr_classes.keys()):
                    if (mapping := comptomap.get(c)) is not None:
                        nms[i] = mapping.framename
                        defaultunit = mapping.defaultunit

                        # need the isinstance because otherwise if it's a unit it
                        # will try to compare to the unit string representation
                        if not (
                            isinstance(defaultunit, str)
                            and defaultunit == "recommended"
                        ):
                            uns[i] = defaultunit
                            # else we just leave it as recommended_units says above

                # Convert to tuples so that this can't mess with frame internals
                repr_attrs[repr_diff_cls]["names"] = tuple(nms)
                repr_attrs[repr_diff_cls]["units"] = tuple(uns)

            cls._frame_class_cache["representation_info"] = repr_attrs
            cls._frame_class_cache["last_reprdiff_hash"] = r.get_reprdiff_cls_hash()
        return cls._frame_class_cache["representation_info"]

    @lazyproperty
    def representation_info(self):
        """
        A dictionary with the information of what attribute names for this frame
        apply to particular representations.
        """
        return self._get_representation_info()

    def get_representation_component_names(self, which="base"):
        cls = self.get_representation_cls(which)
        if cls is None:
            return {}
        return dict(zip(self.representation_info[cls]["names"], cls.attr_classes))

    def get_representation_component_units(self, which="base"):
        repr_or_diff_cls = self.get_representation_cls(which)
        if repr_or_diff_cls is None:
            return {}
        repr_attrs = self.representation_info[repr_or_diff_cls]
        return {k: v for k, v in zip(repr_attrs["names"], repr_attrs["units"]) if v}

    representation_component_names = property(get_representation_component_names)

    representation_component_units = property(get_representation_component_units)
