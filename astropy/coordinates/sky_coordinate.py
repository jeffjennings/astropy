import copy
import functools
import re
import warnings
from collections.abc import Callable
from typing import Union

import erfa
import numpy as np

from astropy import units as u
from astropy.constants import c as speed_of_light
from astropy.table import QTable
from astropy.time import Time
from astropy.utils import ShapedLikeNDArray
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils.masked import MaskableShapedLikeNDArray

from .angles import Angle, offset_by
from .baseframe import (
    BaseCoordinateFrame,
    BaseFrame,
    CoordinateFrameInfo,
    GenericFrame,
    frame_transform_graph,
)
from .builtin_frames import SkyOffsetFrame
from .coordinate import BaseCoordinate, Coordinate
from .distances import Distance
from .errors import ConvertError
from .representation import (
    SphericalDifferential,
    SphericalRepresentation,
    UnitSphericalRepresentation,
)
from .sky_coordinate_parsers import (
    _get_frame_class,
    _get_frame_without_data,
    _parse_coordinate_data,
)

__all__ = ["SkyCoord", "SkyCoordInfo"]


# TODO: APE23: remove _split_bcf when BaseCoordinateFrame is deprecated
def _split_bcf(bcf, copy=True):
    """Split a `~astropy.coordinates.BaseCoordinateFrame` into
    ``(dataless_frame, representation)``.

    Finds the corresponding dataless ``BaseFrame`` subclass
    (e.g., ``ICRSFrame`` from ``ICRS``), instantiates it with the
    BaseCoordinateFrame's frame attributes, and returns it along with the
    BaseCoordinateFrame's data.
    """
    for cls in type(bcf).__mro__:
        if (
            cls is not BaseFrame
            and issubclass(cls, BaseFrame)
            and not issubclass(cls, BaseCoordinateFrame)
        ):
            frame_attrs = {
                k: getattr(bcf, k)
                for k in type(bcf).frame_attributes
                if not bcf.is_frame_attr_default(k)
            }

            dataless = cls(
                representation_type=bcf.representation_type,
                differential_type=bcf.differential_type,
                **frame_attrs,
            )
            break
    else:
        frame_attrs = {
            k: getattr(bcf, k)
            for k in type(bcf).frame_attributes
            if not bcf.is_frame_attr_default(k)
        }
        dataless = type(bcf)(
            representation_type=bcf.representation_type,
            differential_type=bcf.differential_type,
            **frame_attrs,
        )
    data = bcf.data.copy() if copy else bcf.data
    return dataless, data


class SkyCoordInfo(CoordinateFrameInfo):
    # Information for a SkyCoord is almost identical to that of a frame;
    # we only need to add the name of the frame used underneath.
    def _represent_as_dict(self):
        sc = self._parent
        out = super()._represent_as_dict()
        out["frame"] = sc.frame.name
        return out


class SkyCoord(BaseCoordinate, MaskableShapedLikeNDArray):
    """High-level object providing a flexible interface for celestial coordinate
    representation, manipulation, and transformation between systems.

    The |SkyCoord| class accepts a wide variety of inputs for initialization. At
    a minimum these must provide one or more celestial coordinate values with
    unambiguous units.  Inputs may be scalars or lists/tuples/arrays, yielding
    scalar or array coordinates (can be checked via ``SkyCoord.isscalar``).
    Typically one also specifies the coordinate frame, though this is not
    required. The general pattern for spherical representations is::

      SkyCoord(COORD, [FRAME], keyword_args ...)
      SkyCoord(LON, LAT, [FRAME], keyword_args ...)
      SkyCoord(LON, LAT, [DISTANCE], frame=FRAME, unit=UNIT, keyword_args ...)
      SkyCoord([FRAME], <lon_attr>=LON, <lat_attr>=LAT, keyword_args ...)

    It is also possible to input coordinate values in other representations
    such as cartesian or cylindrical.  In this case one includes the keyword
    argument ``representation_type='cartesian'`` (for example) along with data
    in ``x``, ``y``, and ``z``.

    See also: https://docs.astropy.org/en/stable/coordinates/

    Examples
    --------
    The examples below illustrate common ways of initializing a |SkyCoord|
    object.  For a complete description of the allowed syntax see the
    full coordinates documentation.  First some imports::

      >>> from astropy.coordinates import SkyCoord  # High-level coordinates
      >>> from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
      >>> from astropy.coordinates import Angle, Latitude, Longitude  # Angles
      >>> import astropy.units as u

    The coordinate values and frame specification can now be provided using
    positional and keyword arguments::

      >>> c = SkyCoord(10, 20, unit="deg")  # defaults to ICRS frame
      >>> c = SkyCoord([1, 2, 3], [-30, 45, 8], frame="icrs", unit="deg")  # 3 coords

      >>> coords = ["1:12:43.2 +31:12:43", "1 12 43.2 +31 12 43"]
      >>> c = SkyCoord(coords, frame=FK4, unit=(u.hourangle, u.deg), obstime="J1992.21")

      >>> c = SkyCoord("1h12m43.2s +1d12m43s", frame=Galactic)  # Units from string
      >>> c = SkyCoord(frame="galactic", l="1h12m43.2s", b="+1d12m43s")

      >>> ra = Longitude([1, 2, 3], unit=u.deg)  # Could also use Angle
      >>> dec = np.array([4.5, 5.2, 6.3]) * u.deg  # Astropy Quantity
      >>> c = SkyCoord(ra, dec, frame='icrs')
      >>> c = SkyCoord(frame=ICRS, ra=ra, dec=dec, obstime='2001-01-02T12:34:56')

      >>> c = FK4(1 * u.deg, 2 * u.deg)  # Uses defaults for obstime, equinox
      >>> c = SkyCoord(c, obstime='J2010.11', equinox='B1965')  # Override defaults

      >>> c = SkyCoord(w=0, u=1, v=2, unit='kpc', frame='galactic',
      ...              representation_type='cartesian')

      >>> c = SkyCoord([ICRS(ra=1*u.deg, dec=2*u.deg), ICRS(ra=3*u.deg, dec=4*u.deg)])

    Velocity components (proper motions or radial velocities) can also be
    provided in a similar manner::

      >>> c = SkyCoord(ra=1*u.deg, dec=2*u.deg, radial_velocity=10*u.km/u.s)

      >>> c = SkyCoord(ra=1*u.deg, dec=2*u.deg, pm_ra_cosdec=2*u.mas/u.yr, pm_dec=1*u.mas/u.yr)

    As shown, the frame can be a `~astropy.coordinates.BaseCoordinateFrame`
    class or the corresponding string alias -- lower-case versions of the
    class name that allow for creating a |SkyCoord| object and transforming
    frames without explicitly importing the frame classes.

    Parameters
    ----------
    frame : `~astropy.coordinates.BaseCoordinateFrame` class or string, optional
        Type of coordinate frame this |SkyCoord| should represent. Defaults to
        to ICRS if not given or given as None.
    unit : `~astropy.units.Unit`, string, or tuple of :class:`~astropy.units.Unit` or str, optional
        Units for supplied coordinate values.
        If only one unit is supplied then it applies to all values.
        Note that passing only one unit might lead to unit conversion errors
        if the coordinate values are expected to have mixed physical meanings
        (e.g., angles and distances).
    obstime : time-like, optional
        Time(s) of observation.
    equinox : time-like, optional
        Coordinate frame equinox time.
    representation_type : str or Representation class
        Specifies the representation, e.g. 'spherical', 'cartesian', or
        'cylindrical'.  This affects the positional args and other keyword args
        which must correspond to the given representation.
    copy : bool, optional
        If `True` (default), a copy of any coordinate data is made.  This
        argument can only be passed in as a keyword argument.
    **keyword_args
        Other keyword arguments as applicable for user-defined coordinate frames.
        Common options include:

        ra, dec : angle-like, optional
            RA and Dec for frames where ``ra`` and ``dec`` are keys in the
            frame's ``representation_component_names``, including ``ICRS``,
            ``FK5``, ``FK4``, and ``FK4NoETerms``.
        pm_ra_cosdec, pm_dec  : `~astropy.units.Quantity` ['angular speed'], optional
            Proper motion components, in angle per time units.
        l, b : angle-like, optional
            Galactic ``l`` and ``b`` for for frames where ``l`` and ``b`` are
            keys in the frame's ``representation_component_names``, including
            the ``Galactic`` frame.
        pm_l_cosb, pm_b : `~astropy.units.Quantity` ['angular speed'], optional
            Proper motion components in the `~astropy.coordinates.Galactic` frame,
            in angle per time units.
        x, y, z : float or `~astropy.units.Quantity` ['length'], optional
            Cartesian coordinates values
        u, v, w : float or `~astropy.units.Quantity` ['length'], optional
            Cartesian coordinates values for the Galactic frame.
        radial_velocity : `~astropy.units.Quantity` ['speed'], optional
            The component of the velocity along the line-of-sight (i.e., the
            radial direction), in velocity units.
    """

    # Declare that SkyCoord can be used as a Table column by defining the
    # info property.
    info = SkyCoordInfo()

    # TODO: APE23: remove has_data and just check if self._frame is None
    # when BaseCoordinateFrame is deprecated
    has_data = True

    # Methods implemented by the underlying frame
    position_angle: Callable[[Union[BaseCoordinateFrame, "SkyCoord"]], Angle]
    separation: Callable[[Union[BaseCoordinateFrame, "SkyCoord"]], Angle]
    separation_3d: Callable[[Union[BaseCoordinateFrame, "SkyCoord"]], Distance]

    def __init__(self, *args, copy=True, **kwargs):
        # these are frame attributes set on this SkyCoord but *not* a part of
        # the frame object this SkyCoord contains
        self._extra_frameattr_names = set()

        if len(args) == 1 and isinstance(args[0], Coordinate):
            coord = args[0]
            self._frame = coord.frame
            self._data = coord.data.copy() if copy else coord.data
            for attr, val in kwargs.items():
                setattr(self, attr, val)
            return

        # If all that is passed in is a frame instance that already has data,
        # we should bypass all of the parsing and logic below. This is here
        # to make this the fastest way to create a SkyCoord instance. Many of
        # the classmethods implemented for performance enhancements will use
        # this as the initialization path
        # TODO: APE23: simplify when BaseCoordinateFrame deprecated
        if (
            len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], (BaseCoordinateFrame, SkyCoord))
        ):
            coords = args[0]
            if isinstance(coords, SkyCoord):
                self._extra_frameattr_names = coords._extra_frameattr_names
                self.info = coords.info

                # Copy over any extra frame attributes
                for attr_name in self._extra_frameattr_names:
                    # Setting it will also validate it.
                    setattr(self, attr_name, getattr(coords, attr_name))
                self._frame = coords._frame
                self._data = coords._data.copy() if copy else coords._data
                return

            if not coords.has_data:
                raise ValueError(
                    "Cannot initialize from a coordinate frame "
                    "instance without coordinate data"
                )
            self._frame, self._data = _split_bcf(coords, copy=copy)

        else:
            # TODO: APE23: simplify once BaseCoordinateFrame is deprecated
            _dataless_frame_input = None
            frame_arg = kwargs.get("frame")
            if isinstance(frame_arg, BaseFrame) and not isinstance(
                frame_arg, BaseCoordinateFrame
            ):
                _dataless_frame_input = frame_arg
                for sub in type(_dataless_frame_input).__subclasses__():
                    if issubclass(sub, BaseCoordinateFrame):
                        kwargs["frame"] = sub
                        break
                else:
                    raise ValueError(
                        "No legacy frame class found for "
                        f"{type(_dataless_frame_input).__name__}"
                    )
                for attr_name in type(_dataless_frame_input).frame_attributes:
                    if _dataless_frame_input.is_frame_attr_default(attr_name):
                        continue
                    frame_val = getattr(_dataless_frame_input, attr_name)
                    if attr_name not in kwargs:
                        kwargs[attr_name] = frame_val
                    elif np.any(frame_val != kwargs[attr_name]):
                        raise ValueError(
                            f"Frame attribute '{attr_name}' has conflicting values"
                            " between the input coordinate data and either keyword"
                            " arguments or the frame specification (frame=...):"
                            f" {frame_val} =/= {kwargs[attr_name]}"
                        )

            frame_cls, frame_kwargs = _get_frame_without_data(args, kwargs)

            args = list(args)  # Make it mutable
            skycoord_kwargs, components, info = _parse_coordinate_data(
                frame_cls(**frame_kwargs), args, kwargs
            )

            for attr in skycoord_kwargs:
                setattr(self, attr, skycoord_kwargs[attr])

            if info is not None:
                self.info = info

            frame_kwargs.update(components)
            bcf = frame_cls(copy=copy, **frame_kwargs)

            if not bcf.has_data:
                raise ValueError("Cannot create a SkyCoord without data")

            if _dataless_frame_input is not None:
                self._frame = _dataless_frame_input
                self._data = bcf.data.copy() if copy else bcf.data
            else:
                self._frame, self._data = _split_bcf(bcf, copy=copy)

    @functools.cached_property
    def cache(self):
        """Cache for this SkyCoord, a dict.

        It stores anything that should be computed from the coordinate data (*not* from
        the frame attributes). This can be used in functions to store anything that
        might be expensive to compute but might be reused by some other function.
        E.g.::

            if 'user_data' in mycoord.cache:
                data = mycoord.cache['user_data']
            else:
                mycoord.cache['user_data'] = data = expensive_func(mycoord.lat)

        If in-place modifications are made to the coordinate data, the cache should
        be cleared::

            mycoord.cache.clear()
        """
        if "_cache" not in self.__dict__:
            from collections import defaultdict

            self.__dict__["_cache"] = defaultdict(dict)
        return self.__dict__["_cache"]

    @property
    def frame(self):
        """The coordinate frame as a `~astropy.coordinates.BaseCoordinateFrame`
        instance that includes the coordinate data.

        During the deprecation cycle, this always returns a BaseCoordinateFrame instance
        with data (e.g. ``ICRS(ra=..., dec=...)``, not the internal
        ``ICRSFrame()``).  Internal code that needs the data-less
        frame should use ``self._frame``.
        """
        # TODO: APE23: replace function with:
        # def frame(self):
        #     return self._frame
        # when BaseCoordinateFrame is deprecated
        bcf_frame = self.cache.get("frame", {}).get("bcf")
        if bcf_frame is not None:
            return bcf_frame

        if isinstance(self._frame, BaseCoordinateFrame):
            bcf_frame = self._frame.realize_frame(self._data, copy=False)
        else:
            for sub in type(self._frame).__subclasses__():
                if issubclass(sub, BaseCoordinateFrame):
                    fa = {
                        k: getattr(self._frame, k)
                        for k in type(self._frame).frame_attributes
                        if not self._frame.is_frame_attr_default(k)
                    }
                    try:
                        bcf_frame = sub(
                            self._data,
                            copy=False,
                            representation_type=self._frame.representation_type,
                            differential_type=self._frame.differential_type,
                            **fa,
                        )
                        break
                    except Exception:
                        pass
            if bcf_frame is None:
                bcf_frame = self._frame

        self.cache.setdefault("frame", {})["bcf"] = bcf_frame
        return bcf_frame

    @property
    def data(self):
        return self._data

    def __replace__(self, **changes):
        """Return a copy with specified fields replaced.

        Parameters
        ----------
        data : `~astropy.coordinates.BaseRepresentation`, optional
            New representation data.
        frame : frame instance, optional
            New frame (replaces the entire frame; mutually exclusive with
            per-attribute frame overrides).
        **changes
            Frame attribute overrides (e.g. ``obstime=``) or extra frame
            attributes stored on this |SkyCoord|.
        """
        data = changes.pop("data", self._data)
        explicit_frame = changes.pop("frame", None)

        # TODO: APE23: remove this first branch when BaseCoordinateFrame is deprecated
        if isinstance(self._frame, BaseCoordinateFrame):
            frame_attr_names = set(type(self._frame).frame_attributes)
            frame_changes = {k: v for k, v in changes.items() if k in frame_attr_names}
            extra_changes = {
                k: v for k, v in changes.items() if k not in frame_attr_names
            }
            if explicit_frame is not None:
                base_frame = explicit_frame
            elif frame_changes:
                base_frame = self._frame.replicate_without_data(**frame_changes)
            else:
                base_frame = self._frame
            sc_extra = {a: getattr(self, a) for a in self._extra_frameattr_names}
            sc_extra.update(extra_changes)
            return self.__class__(base_frame.realize_frame(data), **sc_extra)
        else:
            # _frame is a data-less BaseFrame subclass.
            frame_attr_names = set(type(self._frame).frame_attributes)
            frame_changes = {k: v for k, v in changes.items() if k in frame_attr_names}
            extra_changes = {
                k: v for k, v in changes.items() if k not in frame_attr_names
            }
            if explicit_frame is not None:
                new_frame = explicit_frame
            elif frame_changes:
                fa = {
                    k: getattr(self._frame, k)
                    for k in type(self._frame).frame_attributes
                }
                fa.update(frame_changes)
                new_frame = type(self._frame)(**fa)
            else:
                new_frame = self._frame
            sc_extra = {a: getattr(self, a) for a in self._extra_frameattr_names}
            sc_extra.update(extra_changes)
            return self.__class__(Coordinate(frame=new_frame, data=data), **sc_extra)

    @property
    def representation_type(self):
        return self._frame.representation_type

    @representation_type.setter
    def representation_type(self, value):
        self._frame.representation_type = value
        self.cache.clear()

    @property
    def differential_type(self):
        return self._frame.differential_type

    @differential_type.setter
    def differential_type(self, value):
        self._frame.differential_type = value
        self.cache.clear()

    @property
    def shape(self):
        return self._data.shape

    def __eq__(self, value):
        """Equality operator for SkyCoord.

        This implements strict equality and requires that the frames are
        equivalent, extra frame attributes are equivalent, and that the
        representation data are exactly equal.
        """
        # TODO: APE23: simplify when BaseCoordinateFrame is deprecated
        if isinstance(value, BaseCoordinateFrame):
            if value._data is None:
                raise ValueError("Can only compare SkyCoord to Frame with data")
            frame_equiv = (
                type(self._frame) == type(value)
                or issubclass(type(value), type(self._frame))
            ) and all(
                BaseCoordinateFrame._frameattr_equiv(
                    getattr(self._frame, a), getattr(value, a)
                )
                for a in type(self._frame).frame_attributes
            )
            if not frame_equiv:
                raise TypeError(
                    "Cannot compare: objects must have equivalent frames: "
                    f"{self._frame!r} vs. {value.replicate_without_data()!r}"
                )
            return self._data == value.data

        if not isinstance(value, SkyCoord):
            return NotImplemented

        # Make sure that any extra frame attribute names are equivalent.
        for attr in self._extra_frameattr_names | value._extra_frameattr_names:
            if not BaseCoordinateFrame._frameattr_equiv(
                getattr(self, attr), getattr(value, attr)
            ):
                raise ValueError(
                    f"cannot compare: extra frame attribute '{attr}' is not equivalent"
                    " (perhaps compare the frames directly to avoid this exception)"
                )

        return (
            self._frame.is_equivalent_frame(value._frame) and self._data == value._data
        )

    def _apply(self, method, *args, **kwargs):
        """Create a new instance, applying a method to the underlying data.

        In typical usage, the method is any of the shape-changing methods for
        `~numpy.ndarray` (``reshape``, ``swapaxes``, etc.), as well as those
        picking particular elements (``__getitem__``, ``take``, etc.), which
        are all defined in `~astropy.utils.shapes.ShapedLikeNDArray`. It will be
        applied to the underlying arrays in the representation (e.g., ``x``,
        ``y``, and ``z`` for `~astropy.coordinates.CartesianRepresentation`),
        as well as to any frame attributes that have a shape, with the results
        used to create a new instance.

        Internally, it is also used to apply functions to the above parts
        (in particular, `~numpy.broadcast_to`).

        Parameters
        ----------
        method : str or callable
            If str, it is the name of a method that is applied to the internal
            ``components``. If callable, the function is applied.
        *args
            Any positional arguments for ``method``.
        **kwargs : dict
            Any keyword arguments for ``method``.
        """

        def apply_method(value):
            if isinstance(value, ShapedLikeNDArray):
                return value._apply(method, *args, **kwargs)
            else:
                if callable(method):
                    return method(value, *args, **kwargs)
                else:
                    return getattr(value, method)(*args, **kwargs)

        # create a new but empty instance, and copy over stuff
        new = super().__new__(self.__class__)
        new._frame = self._apply_to_frame(self._frame, method, *args, **kwargs)
        new._data = self._data._apply(method, *args, **kwargs)
        new._extra_frameattr_names = self._extra_frameattr_names.copy()
        for attr in self._extra_frameattr_names:
            value = getattr(self, attr)
            if getattr(value, "shape", ()):
                value = apply_method(value)
            elif method == "copy" or method == "flatten":
                # flatten should copy also for a single element array, but
                # we cannot use it directly for array scalars, since it
                # always returns a one-dimensional array. So, just copy.
                value = copy.copy(value)
            setattr(new, "_" + attr, value)

        # Copy other 'info' attr only if it has actually been defined.
        # See PR #3898 for further explanation and justification, along
        # with Quantity.__array_finalize__
        if "info" in self.__dict__:
            new.info = self.info

        return new

    def __setitem__(self, item, value):
        """Implement self[item] = value for SkyCoord.

        The right hand ``value`` must be strictly consistent with self:
        - Identical class
        - Equivalent frames
        - Identical representation_types
        - Identical representation differentials keys
        - Identical frame attributes
        - Identical "extra" frame attributes (e.g. obstime for an ICRS coord)

        With these caveats the setitem ends up as effectively a setitem on
        the representation data.

          self.frame.data[item] = value.frame.data
        """
        if value is np.ma.masked or value is np.ma.nomask:
            self._data.__setitem__(item, value)
            self.cache.clear()
            return

        if self.__class__ is not value.__class__:
            raise TypeError(
                "can only set from object of same class: "
                f"{self.__class__.__name__} vs. {value.__class__.__name__}"
            )

        if not self._frame.is_equivalent_frame(value._frame):
            raise ValueError("cannot set: frames are not equivalent")

        # Make sure that any extra frame attribute names are equivalent.
        for attr in self._extra_frameattr_names | value._extra_frameattr_names:
            # TODO: APE23: simplify when BaseCoordinateFrame is deprecated
            if not BaseCoordinateFrame._frameattr_equiv(
                getattr(self, attr), getattr(value, attr)
            ):
                raise ValueError(f"attribute {attr} is not equivalent")

        if self._data.__class__ is not value._data.__class__:
            raise TypeError(
                "can only set from object of same class: "
                f"{self._data.__class__.__name__} vs. {value._data.__class__.__name__}"
            )

        if self._data._differentials:
            if self._data._differentials.keys() != value._data._differentials.keys():
                raise ValueError("setitem value must have same differentials")
            for key, self_diff in self._data._differentials.items():
                if self_diff.__class__ is not value._data._differentials[key].__class__:
                    raise TypeError(
                        "can only set from object of same class: "
                        f"{self_diff.__class__.__name__} vs. "
                        f"{value._data._differentials[key].__class__.__name__}"
                    )

        if self._data.shape == ():
            clsnm = type(self._frame).__name__
            clsnm = clsnm.removesuffix("Frame")
            raise TypeError(
                f"scalar '{clsnm}' frame object does not support item assignment"
            )

        # Set on the representation and clear the cache
        self._data[item] = value._data
        self.cache.clear()

    def insert(self, obj, values, axis=0):
        return self.info._insert(obj, values, axis)

    insert.__doc__ = SkyCoordInfo._insert.__doc__

    def transform_to(self, frame, merge_attributes=True):
        """Transform this coordinate to a new frame.

        The precise frame transformed to depends on ``merge_attributes``.
        If `False`, the destination frame is used exactly as passed in.
        But this is often not quite what one wants.  E.g., suppose one wants to
        transform an ICRS coordinate that has an obstime attribute to FK4; in
        this case, one likely would want to use this information. Thus, the
        default for ``merge_attributes`` is `True`, in which the precedence is
        as follows: (1) explicitly set (i.e., non-default) values in the
        destination frame; (2) explicitly set values in the source; (3) default
        value in the destination frame.

        Note that in either case, any explicitly set attributes on the source
        |SkyCoord| that are not part of the destination frame's definition are
        kept (stored on the resulting |SkyCoord|), and thus one can round-trip
        (e.g., from FK4 to ICRS to FK4 without losing obstime).

        Parameters
        ----------
        frame : str, `~astropy.coordinates.BaseCoordinateFrame` class or instance, or |SkyCoord| instance
            The frame to transform this coordinate into.  If a |SkyCoord|, the
            underlying frame is extracted, and all other information ignored.
        merge_attributes : bool, optional
            Whether the default attributes in the destination frame are allowed
            to be overridden by explicitly set attributes in the source
            (see note above; default: `True`).

        Returns
        -------
        coord : |SkyCoord|
            A new object with this coordinate represented in the `frame` frame.

        Raises
        ------
        ValueError
            If there is no possible transformation route.

        """
        frame_kwargs = {}

        # Frame name (string) or frame class?  Coerce into an instance.
        try:
            frame = _get_frame_class(frame)()
        except Exception:
            pass

        if isinstance(frame, SkyCoord):
            frame = frame.frame  # Change to underlying coord frame instance

        # TODO: APE23: simplify when BaseCoordinateFrame deprecated
        if isinstance(frame, BaseCoordinateFrame):
            new_frame_cls = frame.__class__
            # Get frame attributes, allowing defaults to be overridden by
            # explicitly set attributes of the source if ``merge_attributes``.
            for attr in frame_transform_graph.frame_attributes:
                self_val = getattr(self, attr, None)
                frame_val = getattr(frame, attr, None)
                if frame_val is not None and not (
                    merge_attributes and frame.is_frame_attr_default(attr)
                ):
                    frame_kwargs[attr] = frame_val
                elif self_val is not None and not self.is_frame_attr_default(attr):
                    frame_kwargs[attr] = self_val
                elif frame_val is not None:
                    frame_kwargs[attr] = frame_val
        elif isinstance(frame, BaseFrame):
            new_frame_cls = type(frame)
            for attr in frame_transform_graph.frame_attributes:
                self_val = getattr(self, attr, None)
                frame_val = getattr(frame, attr, None)
                if frame_val is not None and not (
                    merge_attributes and frame.is_frame_attr_default(attr)
                ):
                    frame_kwargs[attr] = frame_val
                elif self_val is not None and not self.is_frame_attr_default(attr):
                    frame_kwargs[attr] = self_val
                elif frame_val is not None:
                    frame_kwargs[attr] = frame_val
        else:
            raise ValueError(
                "Transform `frame` must be a frame name, class, or instance"
            )

        # Get the composite transform to the new frame
        trans = frame_transform_graph.get_transform(self.frame.__class__, new_frame_cls)

        # TODO: APE23: simplify when BaseCoordinateFrame is deprecated.
        if (
            trans is not None
            and not trans.transforms
            and self.frame.__class__ is new_frame_cls
        ):
            _bcf_self = next(
                (
                    s
                    for s in type(self._frame).__subclasses__()
                    if issubclass(s, BaseCoordinateFrame)
                ),
                None,
            )
            if _bcf_self is not None:
                _bcf_trans = frame_transform_graph.get_transform(_bcf_self, _bcf_self)
                if _bcf_trans is not None and _bcf_trans.transforms:
                    trans = _bcf_trans

        if trans is None:
            raise ConvertError(
                f"Cannot transform from {self.frame.__class__} to {new_frame_cls}"
            )

        # Make a generic frame which will accept all the frame kwargs that
        # are provided and allow for transforming through intermediate frames
        # which may require one or more of those kwargs.
        generic_frame = GenericFrame(frame_kwargs)

        # TODO: APE23: simplify when BaseCoordinateFrame is deprecated.
        _bcf_cls = next(
            (
                cls
                for cls in type(self._frame).__subclasses__()
                if issubclass(cls, BaseCoordinateFrame)
            ),
            None,
        )
        if _bcf_cls is not None:
            _frame_attrs = {
                k: getattr(self._frame, k) for k in type(self._frame).frame_attributes
            }
            _from_coord = _bcf_cls(self._data, copy=False, **_frame_attrs)
        else:
            _from_coord = Coordinate(frame=self._frame, data=self._data)
        new_coord = trans(_from_coord, generic_frame)

        # new_coord is a Coordinate. Strip frame_kwargs that are
        # already in the result's frame, then build the new SkyCoord.
        result_frame_attrs = set(type(new_coord.frame).frame_attributes)
        for attr in result_frame_attrs & set(frame_kwargs.keys()):
            frame_kwargs.pop(attr)

        # Always remove the origin frame attribute; it only makes sense on a
        # SkyOffsetFrame (where it is stored on the frame itself).  See gh-11277.
        frame_kwargs.pop("origin", None)

        return self.__class__(new_coord, **frame_kwargs)

    def apply_space_motion(self, new_obstime=None, dt=None):
        """Compute the position to a new time using the velocities.

        Compute the position of the source represented by this coordinate object
        to a new time using the velocities stored in this object and assuming
        linear space motion (including relativistic corrections). This is
        sometimes referred to as an "epoch transformation".

        The initial time before the evolution is taken from the ``obstime``
        attribute of this coordinate.  Note that this method currently does not
        support evolving coordinates where the *frame* has an ``obstime`` frame
        attribute, so the ``obstime`` is only used for storing the before and
        after times, not actually as an attribute of the frame. Alternatively,
        if ``dt`` is given, an ``obstime`` need not be provided at all.

        Parameters
        ----------
        new_obstime : `~astropy.time.Time`, optional
            The time at which to evolve the position to. Requires that the
            ``obstime`` attribute be present on this frame.
        dt : `~astropy.units.Quantity`, `~astropy.time.TimeDelta`, optional
            An amount of time to evolve the position of the source. Cannot be
            given at the same time as ``new_obstime``.

        Returns
        -------
        new_coord : |SkyCoord|
            A new coordinate object with the evolved location of this coordinate
            at the new time.  ``obstime`` will be set on this object to the new
            time only if ``self`` also has ``obstime``.
        """
        from .builtin_frames.icrs import ICRS

        if (new_obstime is None) == (dt is None):
            raise ValueError(
                "You must specify one of `new_obstime` or `dt`, but not both."
            )

        # Validate that we have velocity info
        if "s" not in self.data.differentials:
            raise ValueError("SkyCoord requires velocity data to evolve the position.")

        if "obstime" in self.frame.frame_attributes:
            raise NotImplementedError(
                "Updating the coordinates in a frame with explicit time dependence is"
                " currently not supported. If you would like this functionality, please"
                " open an issue on github:\nhttps://github.com/astropy/astropy"
            )

        if new_obstime is not None and self.obstime is None:
            # If no obstime is already on this object, raise an error if a new
            # obstime is passed: we need to know the time / epoch at which the
            # the position / velocity were measured initially
            raise ValueError(
                "This object has no associated `obstime`. apply_space_motion() must"
                " receive a time difference, `dt`, and not a new obstime."
            )

        # Compute t1 and t2, the times used in the starpm call, which *only*
        # uses them to compute a delta-time
        t1 = self.obstime
        if dt is None:
            # self.obstime is not None and new_obstime is not None b/c of above
            # checks
            t2 = new_obstime
        else:
            # new_obstime is definitely None b/c of the above checks
            if t1 is None:
                # MAGIC NUMBER: if the current SkyCoord object has no obstime,
                # assume J2000 to do the dt offset. This is not actually used
                # for anything except a delta-t in starpm, so it's OK that it's
                # not necessarily the "real" obstime
                t1 = Time("J2000")
                new_obstime = None  # we don't actually know the initial obstime
                t2 = t1 + dt
            else:
                t2 = t1 + dt
                new_obstime = t2
        # starpm wants tdb time
        t1 = t1.tdb
        t2 = t2.tdb

        # proper motion in RA should not include the cos(dec) term, see the
        # erfa function eraStarpv, comment (4).  So we convert to the regular
        # spherical differentials.
        icrsrep = self.icrs.represent_as(SphericalRepresentation, SphericalDifferential)
        icrsvel = icrsrep.differentials["s"]

        parallax_zero = False
        try:
            plx = icrsrep.distance.to_value(u.arcsecond, u.parallax())
        except u.UnitConversionError:  # No distance: set to 0 by convention
            plx = 0.0
            parallax_zero = True

        try:
            rv = icrsvel.d_distance.to_value(u.km / u.s)
        except u.UnitConversionError:  # No RV
            rv = 0.0

        starpm = erfa.pmsafe(
            icrsrep.lon.radian,
            icrsrep.lat.radian,
            icrsvel.d_lon.to_value(u.radian / u.yr),
            icrsvel.d_lat.to_value(u.radian / u.yr),
            plx,
            rv,
            t1.jd1,
            t1.jd2,
            t2.jd1,
            t2.jd2,
        )

        if parallax_zero:
            new_distance = None
        else:
            new_distance = Distance(parallax=starpm[4] << u.arcsec)

        icrs2 = ICRS(
            ra=u.Quantity(starpm[0], u.radian, copy=None),
            dec=u.Quantity(starpm[1], u.radian, copy=None),
            pm_ra=u.Quantity(starpm[2], u.radian / u.yr, copy=None),
            pm_dec=u.Quantity(starpm[3], u.radian / u.yr, copy=None),
            distance=new_distance,
            radial_velocity=u.Quantity(starpm[5], u.km / u.s, copy=None),
            differential_type=SphericalDifferential,
        )

        # Update the obstime of the returned SkyCoord, and need to carry along
        # the frame attributes
        frattrs = {
            attrnm: getattr(self, attrnm) for attrnm in self._extra_frameattr_names
        }
        frattrs["obstime"] = new_obstime
        result = self.__class__(icrs2, **frattrs).transform_to(self.frame)

        # Without this the output might not have the right differential type.
        # Not sure if this fixes the problem or just hides it.  See #11932
        result.differential_type = self.differential_type

        return result

    def realize_frame(self, data, **kwargs):
        """Return a new |SkyCoord| with ``data`` substituted for the current data.

        The frame and extra frame attributes are preserved.

        Parameters
        ----------
        data : `~astropy.coordinates.BaseRepresentation`
            The new representation data.
        **kwargs
            Frame attributes to override.

        Returns
        -------
        |SkyCoord|
        """
        return self.__replace__(data=data, **kwargs)

    def replicate(self, data=None, **frame_attrs):
        """Return a new |SkyCoord| with optionally substituted data and/or frame attributes."""
        if data is not None:
            return self.__replace__(data=data, **frame_attrs)
        return self.__replace__(**frame_attrs)

    def replicate_without_data(self, **frame_attrs):
        """|SkyCoord| always requires data; this method is not supported.

        To get the underlying frame without data, use
        ``sc.frame.replicate_without_data(**frame_attrs)``.
        """
        raise NotImplementedError(
            "SkyCoord always requires data. "
            "Use sc.frame.replicate_without_data() to get the underlying frame without data."
        )

    def _is_name(self, string):
        """
        Returns whether a string is one of the aliases for the frame.
        """
        return self._frame.name == string or (
            isinstance(self._frame.name, list) and string in self._frame.name
        )

    def is_frame_attr_default(self, attrnm):
        """Return True if ``attrnm`` is using its default value.

        For attributes on the current frame, this delegates to the frame.
        Attributes in ``_extra_frameattr_names`` were explicitly set and are
        not considered defaults.
        """
        if attrnm in type(self._frame).frame_attributes:
            return self._frame.is_frame_attr_default(attrnm)
        if attrnm in self._extra_frameattr_names:
            return False
        return True

    def __getattr__(self, attr):  # noqa: PLR0911
        """
        Overrides getattr to return coordinates that this can be transformed
        to, based on the alias attr in the primary transform graph.
        """
        if "_frame" in self.__dict__:
            if self._is_name(attr):
                return self

            # Frame attributes: prefer current frame, then carry-along extras
            if attr in frame_transform_graph.frame_attributes:
                if attr in type(self._frame).frame_attributes:
                    return getattr(self._frame, attr)
                else:
                    return getattr(self, "_" + attr, None)

            # Representation component names (e.g., ra, dec, l, b).
            repr_names = self._frame.representation_component_names
            if attr in repr_names:
                cache_key = (type(self._frame.representation_type), None, True)
                if cache_key not in self.cache["representation"]:
                    self.cache["representation"][cache_key] = self.represent_as(
                        self._frame.representation_type, in_frame_units=True
                    )
                rep = self.cache["representation"][cache_key]
                return getattr(rep, repr_names[attr])

            # Differential component names (e.g., pm_ra_cosdec, radial_velocity).
            diff_names = self._frame.get_representation_component_names("s")
            if attr in diff_names:
                if "s" not in self._data.differentials:
                    raise AttributeError(
                        f"{type(self).__name__!r} object has no associated"
                        f" differentials. The component {attr!r} requires them."
                    )
                rep_cls_dict = self._frame.get_representation_cls(None)
                cache_key = (
                    type(rep_cls_dict["base"]),
                    type(rep_cls_dict.get("s")),
                    True,
                )
                if cache_key not in self.cache["representation"]:
                    self.cache["representation"][cache_key] = self.represent_as(
                        in_frame_units=True, **rep_cls_dict
                    )
                rep = self.cache["representation"][cache_key]
                return getattr(rep.differentials["s"], diff_names[attr])

            # Try to interpret as a new frame for transforming.
            frame_cls = frame_transform_graph.lookup_name(attr)
            if frame_cls is not None and self._frame.is_transformable_to(frame_cls):
                return self.transform_to(attr)

            # Registered frame-specific properties
            from astropy.coordinates.coordinate import BaseCoordinate

            frame_props = BaseCoordinate._get_frame_props(self._frame.name)
            if attr in frame_props:
                return frame_props[attr](self)

            # TODO: APE23: remove when BaseCoordinateFrame is deprecated
            try:
                return getattr(self.frame, attr)
            except AttributeError:
                pass

        # Call __getattribute__; this will give the correct exception.
        return self.__getattribute__(attr)

    def __setattr__(self, attr, val):
        # This is to make anything available through __getattr__ immutable
        if attr != "info" and not attr.startswith("_"):
            if "_frame" in self.__dict__:
                if self._is_name(attr):
                    raise AttributeError(f"'{attr}' is immutable")

                _repr_names = self._frame.get_representation_component_names()
                _diff_names = self._frame.get_representation_component_names("s")
                if attr in _repr_names or attr in _diff_names:
                    raise AttributeError(f"'{attr}' is immutable")

                if attr in type(self._frame).frame_attributes:
                    raise AttributeError(f"'{attr}' is immutable")

            if attr in frame_transform_graph.frame_attributes:
                # Store as a private variable and track in _extra_frameattr_names
                super().__setattr__("_" + attr, val)
                # Validate it
                frame_transform_graph.frame_attributes[attr].__get__(self)
                self._extra_frameattr_names |= {attr}
                return

            if frame_transform_graph.lookup_name(attr) is not None:
                raise AttributeError(f"'{attr}' is immutable")

        # Otherwise, do the standard Python attribute setting
        super().__setattr__(attr, val)

    def __delattr__(self, attr):
        # mirror __setattr__ above
        if "_frame" in self.__dict__:
            if self._is_name(attr):
                raise AttributeError(f"'{attr}' is immutable")

            # Attribute belongs to the current frame: rebuild the frame
            # without this attribute (i.e., revert to its default).
            if attr in type(self._frame).frame_attributes:
                fa = {
                    k: getattr(self._frame, k)
                    for k in type(self._frame).frame_attributes
                    if k != attr
                }
                self.__dict__["_frame"] = type(self._frame)(**fa)
                self.cache.clear()
                return

            frame_cls = frame_transform_graph.lookup_name(attr)
            if frame_cls is not None and self._frame.is_transformable_to(frame_cls):
                raise AttributeError(f"'{attr}' is immutable")

        if attr in frame_transform_graph.frame_attributes:
            # All possible frame attributes can be deleted, but need to remove
            # the corresponding private variable.  See __getattr__ above.
            super().__delattr__("_" + attr)
            # Also remove it from the set of extra attributes
            self._extra_frameattr_names -= {attr}

        else:
            # Otherwise, do the standard Python attribute setting
            super().__delattr__(attr)

    def __dir__(self):
        """Original dir() behavior, plus frame attributes and transforms.

        This dir includes:
        - All attributes of the SkyCoord class
        - Coordinate transforms available by aliases
        - Attribute / methods of the underlying self.frame objects
        """
        dir_values = set(super().__dir__())

        # determine the aliases that this can be transformed to.
        for name in frame_transform_graph.get_names():
            frame_cls = frame_transform_graph.lookup_name(name)
            if self.frame.is_transformable_to(frame_cls):
                dir_values.add(name)

        # Add public attributes of self.frame
        dir_values.update(
            {attr for attr in dir(self.frame) if not attr.startswith("_")}
        )

        # Add all possible frame attributes
        dir_values.update(frame_transform_graph.frame_attributes.keys())

        # Add registered frame-specific properties
        from astropy.coordinates.coordinate import BaseCoordinate

        dir_values.update(BaseCoordinate._get_frame_props(self._frame.name))

        return sorted(dir_values)

    def __repr__(self):
        clsnm = self.__class__.__name__
        # TODO: APE23: simplify when BaseCoordinateFrame is deprecated
        bcf_cls = next(
            (
                cls
                for cls in type(self._frame).__subclasses__()
                if issubclass(cls, BaseCoordinateFrame)
            ),
            None,
        )
        coonm = bcf_cls.__name__ if bcf_cls is not None else type(self._frame).__name__
        frameattrs = self._frame._frame_attrs_repr()
        if frameattrs:
            frameattrs = ": " + frameattrs

        data = self._data_repr()
        if data:
            data = ": " + data

        return f"<{clsnm} ({coonm}{frameattrs}){data}>"

    def to_table(self) -> QTable:
        """
        Convert this |SkyCoord| to a |QTable|.

        Any attributes that have the same length as the |SkyCoord| will be
        converted to columns of the |QTable|. All other attributes will be
        recorded as metadata.

        Returns
        -------
        `~astropy.table.QTable`
            A |QTable| containing the data of this |SkyCoord|.

        Examples
        --------
        >>> sc = SkyCoord(ra=[40, 70]*u.deg, dec=[0, -20]*u.deg,
        ...               obstime=Time([2000, 2010], format='jyear'))
        >>> t =  sc.to_table()
        >>> t
        <QTable length=2>
           ra     dec   obstime
          deg     deg
        float64 float64   Time
        ------- ------- -------
           40.0     0.0  2000.0
           70.0   -20.0  2010.0
        >>> t.meta
        {'representation_type': 'spherical', 'frame': 'icrs'}
        """
        table = super().to_table()
        # Record extra carry-along attributes: array-shaped ones become columns,
        # scalar ones become metadata.
        for key in self._extra_frameattr_names:
            value = getattr(self, key)
            if getattr(value, "shape", ())[:1] == (len(self),):
                table[key] = value
            else:
                table.meta[key] = value
        return table

    def is_equivalent_frame(self, other):
        """
        Checks if this object's frame is the same as that of the ``other``
        object.

        To be the same frame, two objects must be the same frame class and have
        the same frame attributes. For two |SkyCoord| objects, *all* of the
        frame attributes have to match, not just those relevant for the object's
        frame.

        Parameters
        ----------
        other : SkyCoord or BaseCoordinateFrame
            The other object to check.

        Returns
        -------
        isequiv : bool
            True if the frames are the same, False if not.

        Raises
        ------
        TypeError
            If ``other`` isn't a |SkyCoord| or a subclass of
            `~astropy.coordinates.BaseCoordinateFrame`.
        """
        # TODO: APE23: simplify when BaseCoordinateFrame is deprecated
        if isinstance(other, BaseCoordinateFrame):
            other_frame = other
            for cls in type(other).__mro__:
                if (
                    cls is not BaseFrame
                    and issubclass(cls, BaseFrame)
                    and not issubclass(cls, BaseCoordinateFrame)
                ):
                    fa = {
                        k: getattr(other, k)
                        for k in type(other).frame_attributes
                        if not other.is_frame_attr_default(k)
                    }
                    other_frame = cls(**fa)
                    break
            return self.frame.is_equivalent_frame(other_frame)
        elif isinstance(other, SkyCoord):
            if other.frame.name != self.frame.name:
                return False

            for fattrnm in frame_transform_graph.frame_attributes:
                if not BaseCoordinateFrame._frameattr_equiv(
                    getattr(self, fattrnm), getattr(other, fattrnm)
                ):
                    return False
            return True
        else:
            # not a BaseCoordinateFrame nor a SkyCoord object
            raise TypeError(
                "Tried to do is_equivalent_frame on something that isn't frame-like"
            )

    # High-level convenience methods
    def spherical_offsets_to(self, tocoord):
        r"""
        Computes angular offsets to go *from* this coordinate *to* another.

        Parameters
        ----------
        tocoord : `~astropy.coordinates.BaseCoordinateFrame`
            The coordinate to find the offset to.

        Returns
        -------
        lon_offset : `~astropy.coordinates.Angle`
            The angular offset in the longitude direction. The definition of
            "longitude" depends on this coordinate's frame (e.g., RA for
            equatorial coordinates).
        lat_offset : `~astropy.coordinates.Angle`
            The angular offset in the latitude direction. The definition of
            "latitude" depends on this coordinate's frame (e.g., Dec for
            equatorial coordinates).

        Raises
        ------
        ValueError
            If the ``tocoord`` is not in the same frame as this one. This is
            different from the behavior of the
            :meth:`~astropy.coordinates.BaseCoordinateFrame.separation`/:meth:`~astropy.coordinates.BaseCoordinateFrame.separation_3d`
            methods because the offset components depend critically on the
            specific choice of frame.

        Notes
        -----
        This uses the sky offset frame machinery, and hence will produce a new
        sky offset frame if one does not already exist for this object's frame
        class.

        See Also
        --------
        :meth:`~astropy.coordinates.BaseCoordinateFrame.separation` :
            for the *total* angular offset (not broken out into components).
        :meth:`~astropy.coordinates.BaseCoordinateFrame.position_angle` :
            for the direction of the offset.

        """
        if not self.is_equivalent_frame(tocoord):
            raise ValueError(
                "Tried to use spherical_offsets_to with two non-matching frames!"
            )

        aframe = self.skyoffset_frame()
        acoord = tocoord.transform_to(aframe)

        dlon = acoord.spherical.lon.view(Angle)
        dlat = acoord.spherical.lat.view(Angle)
        return dlon, dlat

    def spherical_offsets_by(self, d_lon, d_lat):
        """
        Computes the coordinate that is a specified pair of angular offsets away
        from this coordinate.

        Parameters
        ----------
        d_lon : angle-like
            The angular offset in the longitude direction. The definition of
            "longitude" depends on this coordinate's frame (e.g., RA for
            equatorial coordinates).
        d_lat : angle-like
            The angular offset in the latitude direction. The definition of
            "latitude" depends on this coordinate's frame (e.g., Dec for
            equatorial coordinates).

        Returns
        -------
        newcoord : `~astropy.coordinates.SkyCoord`
            The coordinates for the location that corresponds to offsetting by
            ``d_lat`` in the latitude direction and ``d_lon`` in the longitude
            direction.

        Notes
        -----
        This internally uses `~astropy.coordinates.SkyOffsetFrame` to do the
        transformation. For a more complete set of transform offsets, use
        `~astropy.coordinates.SkyOffsetFrame` or `~astropy.wcs.WCS` manually.
        This specific method can be reproduced by doing
        ``SkyCoord(SkyOffsetFrame(d_lon, d_lat, origin=self).transform_to(self))``.

        See Also
        --------
        spherical_offsets_to : compute the angular offsets to another coordinate
        directional_offset_by : offset a coordinate by an angle in a direction
        """
        from .builtin_frames.skyoffset import SkyOffsetFrame

        return self.__class__(
            SkyOffsetFrame(d_lon, d_lat, origin=self).transform_to(self)
        )

    def directional_offset_by(self, position_angle, separation):
        """
        Computes coordinates at the given offset from this coordinate.

        Parameters
        ----------
        position_angle : `~astropy.coordinates.Angle`
            position_angle of offset
        separation : `~astropy.coordinates.Angle`
            offset angular separation

        Returns
        -------
        newpoints : `~astropy.coordinates.SkyCoord`
            The coordinates for the location that corresponds to offsetting by
            the given ``position_angle`` and ``separation``.

        Notes
        -----
        Returned SkyCoord frame retains only the frame attributes that are for
        the resulting frame type.  (e.g. if the input frame is
        `~astropy.coordinates.ICRS`, an ``equinox`` value will be retained, but
        an ``obstime`` will not.)

        For a more complete set of transform offsets, use `~astropy.wcs.WCS`.
        `~astropy.coordinates.SkyCoord.skyoffset_frame()` can also be used to
        create a spherical frame with (lat=0, lon=0) at a reference point,
        approximating an xy cartesian system for small offsets. This method
        is distinct in that it is accurate on the sphere.

        See Also
        --------
        :meth:`~astropy.coordinates.BaseCoordinateFrame.position_angle` :
            inverse operation for the ``position_angle`` component
        :meth:`~astropy.coordinates.BaseCoordinateFrame.separation` :
            inverse operation for the ``separation`` component

        """
        slat = self.represent_as(UnitSphericalRepresentation).lat
        slon = self.represent_as(UnitSphericalRepresentation).lon

        newlon, newlat = offset_by(
            lon=slon, lat=slat, posang=position_angle, distance=separation
        )

        return SkyCoord(newlon, newlat, frame=self.frame)

    def match_to_catalog_sky(self, catalogcoord, nthneighbor=1):
        """
        Finds the nearest on-sky matches of this coordinate in a set of
        catalog coordinates.

        For more on how to use this (and related) functionality, see the
        examples in :ref:`astropy-coordinates-separations-matching`.

        Parameters
        ----------
        catalogcoord : `~astropy.coordinates.SkyCoord` or `~astropy.coordinates.BaseCoordinateFrame`
            The base catalog in which to search for matches. Typically this
            will be a coordinate object that is an array (i.e.,
            ``catalogcoord.isscalar == False``)
        nthneighbor : int, optional
            Which closest neighbor to search for.  Typically ``1`` is
            desired here, as that is correct for matching one set of
            coordinates to another. The next likely use case is ``2``,
            for matching a coordinate catalog against *itself* (``1``
            is inappropriate because each point will find itself as the
            closest match).

        Returns
        -------
        CoordinateMatchResult
            A `~typing.NamedTuple` with attributes representing for each
            source in this |SkyCoord| the indices and angular and
            physical separations of the match in ``catalogcoord``. If
            either the |SkyCoord| or ``catalogcoord`` don't have
            distances, the physical separation is the 3D distance on the
            unit sphere, rather than a true distance.

        Notes
        -----
        This method requires `SciPy <https://www.scipy.org/>`_ to be
        installed or it will fail.

        See Also
        --------
        astropy.coordinates.match_coordinates_sky
        SkyCoord.match_to_catalog_3d
        """
        from .matching import match_coordinates_sky

        # TODO: APE23: simplify when BaseCoordinateFrame deprecated
        if not (
            isinstance(catalogcoord, (SkyCoord, BaseCoordinateFrame))
            and catalogcoord.has_data
        ):
            raise TypeError(
                "Can only get separation to another SkyCoord or a "
                "coordinate frame with data"
            )

        return match_coordinates_sky(
            self, catalogcoord, nthneighbor=nthneighbor, storekdtree="_kdtree_sky"
        )

    def match_to_catalog_3d(self, catalogcoord, nthneighbor=1):
        """
        Finds the nearest 3-dimensional matches of this coordinate to a set
        of catalog coordinates.

        This finds the 3-dimensional closest neighbor, which is only different
        from the on-sky distance if ``distance`` is set in this object or the
        ``catalogcoord`` object.

        For more on how to use this (and related) functionality, see the
        examples in :ref:`astropy-coordinates-separations-matching`.

        Parameters
        ----------
        catalogcoord : `~astropy.coordinates.SkyCoord` or `~astropy.coordinates.BaseCoordinateFrame`
            The base catalog in which to search for matches. Typically this
            will be a coordinate object that is an array (i.e.,
            ``catalogcoord.isscalar == False``)
        nthneighbor : int, optional
            Which closest neighbor to search for.  Typically ``1`` is
            desired here, as that is correct for matching one set of
            coordinates to another.  The next likely use case is
            ``2``, for matching a coordinate catalog against *itself*
            (``1`` is inappropriate because each point will find
            itself as the closest match).

        Returns
        -------
        CoordinateMatchResult
            A `~typing.NamedTuple` with attributes representing for each
            source in this |SkyCoord| the indices and angular and physical
            separations of the match in ``catalogcoord``.

        Notes
        -----
        This method requires `SciPy <https://www.scipy.org/>`_ to be
        installed or it will fail.

        See Also
        --------
        astropy.coordinates.match_coordinates_3d
        SkyCoord.match_to_catalog_sky
        """
        from .matching import match_coordinates_3d

        # TODO: APE23: simplify when BaseCoordinateFrame is deprecated
        if not (
            isinstance(catalogcoord, (SkyCoord, BaseCoordinateFrame))
            and catalogcoord.has_data
        ):
            raise TypeError(
                "Can only get separation to another SkyCoord or a "
                "coordinate frame with data"
            )

        return match_coordinates_3d(
            self, catalogcoord, nthneighbor=nthneighbor, storekdtree="_kdtree_3d"
        )

    def search_around_sky(self, searcharoundcoords, seplimit):
        """
        Searches for all coordinates in this object around a supplied set of
        points within a given on-sky separation.

        This is intended for use on `~astropy.coordinates.SkyCoord` objects
        with coordinate arrays, rather than a scalar coordinate.  For a scalar
        coordinate, it is better to use
        :meth:`~astropy.coordinates.BaseCoordinateFrame.separation`.

        For more on how to use this (and related) functionality, see the
        examples in :ref:`astropy-coordinates-separations-matching`.

        Parameters
        ----------
        searcharoundcoords : coordinate-like
            The coordinates to search around to try to find matching points in
            this |SkyCoord|. This must be a one-dimensional coordinate array.
        seplimit : `~astropy.units.Quantity` ['angle']
            The on-sky separation to search within. It should be broadcastable to the
            same shape as ``searcharoundcoords``.

        Returns
        -------
        CoordinateSearchResult
            A `~typing.NamedTuple` with attributes representing the
            indices of the elements of found pairs in the other set of
            coordinates and this |SkyCoord| and angular and physical
            separations of the pairs. If either set of sources lack
            distances, the physical separation is the 3D distance on the
            unit sphere, rather than a true distance.

        Notes
        -----
        This method requires `SciPy <https://www.scipy.org/>`_ to be
        installed or it will fail.

        In the current implementation, the return values are always sorted in
        the same order as the ``searcharoundcoords`` (so ``idxsearcharound`` is
        in ascending order).  This is considered an implementation detail,
        though, so it could change in a future release.

        See Also
        --------
        astropy.coordinates.search_around_sky
        SkyCoord.search_around_3d
        """
        from .matching import search_around_sky

        return search_around_sky(
            searcharoundcoords, self, seplimit, storekdtree="_kdtree_sky"
        )

    def search_around_3d(self, searcharoundcoords, distlimit):
        """
        Searches for all coordinates in this object around a supplied set of
        points within a given 3D radius.

        This is intended for use on `~astropy.coordinates.SkyCoord` objects
        with coordinate arrays, rather than a scalar coordinate.  For a scalar
        coordinate, it is better to use
        :meth:`~astropy.coordinates.BaseCoordinateFrame.separation_3d`.

        For more on how to use this (and related) functionality, see the
        examples in :ref:`astropy-coordinates-separations-matching`.

        Parameters
        ----------
        searcharoundcoords : `~astropy.coordinates.SkyCoord` or `~astropy.coordinates.BaseCoordinateFrame`
            The coordinates to search around to try to find matching points in
            this |SkyCoord|. This must be a one-dimensional coordinate array.
        distlimit : `~astropy.units.Quantity` ['length']
            The physical radius to search within. It should be broadcastable to the same
            shape as ``searcharoundcoords``.

        Returns
        -------
        CoordinateSearchResult
            A `~typing.NamedTuple` with attributes representing the
            indices of the elements of found pairs in the other set of
            coordinates and this |SkyCoord| and angular and physical
            separations of the pairs.

        Notes
        -----
        This method requires `SciPy <https://www.scipy.org/>`_ to be
        installed or it will fail.

        In the current implementation, the return values are always sorted in
        the same order as the ``searcharoundcoords`` (so ``idxsearcharound`` is
        in ascending order).  This is considered an implementation detail,
        though, so it could change in a future release.

        See Also
        --------
        astropy.coordinates.search_around_3d
        SkyCoord.search_around_sky
        """
        from .matching import search_around_3d

        return search_around_3d(
            searcharoundcoords, self, distlimit, storekdtree="_kdtree_3d"
        )

    def skyoffset_frame(self, rotation=None):
        """
        Returns the sky offset frame with this SkyCoord at the origin.

        Parameters
        ----------
        rotation : angle-like
            The final rotation of the frame about the ``origin``. The sign of
            the rotation is the left-hand rule. That is, an object at a
            particular position angle in the un-rotated system will be sent to
            the positive latitude (z) direction in the final frame.

        Returns
        -------
        astrframe : `~astropy.coordinates.SkyOffsetFrame`
            A sky offset frame of the same type as this |SkyCoord| (e.g., if
            this object has an ICRS coordinate, the resulting frame is
            SkyOffsetICRS, with the origin set to this object)
        """
        return SkyOffsetFrame(origin=self, rotation=rotation)

    def get_constellation(self, short_name=False, constellation_list="iau"):
        """
        Determines the constellation(s) of the coordinates this SkyCoord contains.

        Parameters
        ----------
        short_name : bool
            If True, the returned names are the IAU-sanctioned abbreviated
            names.  Otherwise, full names for the constellations are used.
        constellation_list : str
            The set of constellations to use.  Currently only ``'iau'`` is
            supported, meaning the 88 "modern" constellations endorsed by the IAU.

        Returns
        -------
        constellation : str or string array
            If this is a scalar coordinate, returns the name of the
            constellation.  If it is an array |SkyCoord|, it returns an array of
            names.

        Notes
        -----
        To determine which constellation a point on the sky is in, this first
        precesses to B1875, and then uses the Delporte boundaries of the 88
        modern constellations, as tabulated by
        `Roman 1987 <https://cdsarc.cds.unistra.fr/viz-bin/Cat?VI/42>`_.

        See Also
        --------
        astropy.coordinates.get_constellation
        """
        from .funcs import get_constellation

        # because of issue #7028, the conversion to a PrecessedGeocentric
        # system fails in some cases.  Work around is to  drop the velocities.
        # they are not needed here since only position information is used
        extra_frameattrs = {nm: getattr(self, nm) for nm in self._extra_frameattr_names}
        novel = SkyCoord(
            self.realize_frame(self.data.without_differentials()), **extra_frameattrs
        )
        return get_constellation(novel, short_name, constellation_list)

        # the simpler version below can be used when gh-issue #7028 is resolved
        # return get_constellation(self, short_name, constellation_list)

    # WCS pixel to/from sky conversions
    def to_pixel(self, wcs, origin=0, mode="all"):
        """
        Convert this coordinate to pixel coordinates using a `~astropy.wcs.WCS`
        object.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The WCS to use for convert
        origin : int
            Whether to return 0 or 1-based pixel coordinates.
        mode : 'all' or 'wcs'
            Whether to do the transformation including distortions (``'all'``) or
            only including only the core WCS transformation (``'wcs'``).

        Returns
        -------
        xp, yp : `numpy.ndarray`
            The pixel coordinates

        See Also
        --------
        astropy.wcs.utils.skycoord_to_pixel : the implementation of this method
        """
        from astropy.wcs.utils import skycoord_to_pixel

        return skycoord_to_pixel(self, wcs=wcs, origin=origin, mode=mode)

    @classmethod
    def from_pixel(cls, xp, yp, wcs, origin=0, mode="all"):
        """
        Create a new SkyCoord from pixel coordinates using a World Coordinate System.

        Parameters
        ----------
        xp, yp : float or ndarray
            The coordinates to convert.
        wcs : `~astropy.wcs.WCS`
            The WCS to use for convert
        origin : int
            Whether to return 0 or 1-based pixel coordinates.
        mode : 'all' or 'wcs'
            Whether to do the transformation including distortions (``'all'``) or
            only including only the core WCS transformation (``'wcs'``).

        Returns
        -------
        coord : `~astropy.coordinates.SkyCoord`
            A new object with sky coordinates corresponding to the input ``xp``
            and ``yp``.

        See Also
        --------
        to_pixel : to do the inverse operation
        astropy.wcs.utils.pixel_to_skycoord : the implementation of this method
        """
        from astropy.wcs.utils import pixel_to_skycoord

        return pixel_to_skycoord(xp, yp, wcs=wcs, origin=origin, mode=mode, cls=cls)

    def contained_by(self, wcs, image=None, **kwargs):
        """
        Determines if the SkyCoord is contained in the given wcs footprint.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The coordinate to check if it is within the wcs coordinate.
        image : array
            Optional.  The image associated with the wcs object that the coordinate
            is being checked against. If not given the naxis keywords will be used
            to determine if the coordinate falls within the wcs footprint.
        **kwargs
            Additional arguments to pass to `~astropy.coordinates.SkyCoord.to_pixel`

        Returns
        -------
        response : bool
            True means the WCS footprint contains the coordinate, False means it does not.
        """
        if image is not None:
            ymax, xmax = image.shape
        else:
            xmax, ymax = wcs._naxis

        with warnings.catch_warnings():
            #  Suppress warnings since they just mean we didn't find the coordinate
            warnings.simplefilter("ignore")
            try:
                x, y = self.to_pixel(wcs, **kwargs)
            except Exception:
                return False

        return (x < xmax) & (x > 0) & (y < ymax) & (y > 0)

    def radial_velocity_correction(
        self, kind="barycentric", obstime=None, location=None
    ):
        """
        Compute the correction required to convert a radial velocity at a given
        time and place on the Earth's Surface to a barycentric or heliocentric
        velocity.

        Parameters
        ----------
        kind : str
            The kind of velocity correction.  Must be 'barycentric' or
            'heliocentric'.
        obstime : `~astropy.time.Time` or None, optional
            The time at which to compute the correction.  If `None`, the
            ``obstime`` frame attribute on the |SkyCoord| will be used.
        location : `~astropy.coordinates.EarthLocation` or None, optional
            The observer location at which to compute the correction.  If
            `None`, the  ``location`` frame attribute on the passed-in
            ``obstime`` will be used, and if that is None, the ``location``
            frame attribute on the |SkyCoord| will be used.

        Raises
        ------
        ValueError
            If either ``obstime`` or ``location`` are passed in (not ``None``)
            when the frame attribute is already set on this |SkyCoord|.
        TypeError
            If ``obstime`` or ``location`` aren't provided, either as arguments
            or as frame attributes.

        Returns
        -------
        vcorr : `~astropy.units.Quantity` ['speed']
            The  correction with a positive sign.  I.e., *add* this
            to an observed radial velocity to get the barycentric (or
            heliocentric) velocity. If m/s precision or better is needed,
            see the notes below.

        Notes
        -----
        The barycentric correction is calculated to higher precision than the
        heliocentric correction and includes additional physics (e.g time dilation).
        Use barycentric corrections if m/s precision is required.

        The algorithm here is sufficient to perform corrections at the mm/s level, but
        care is needed in application. The barycentric correction returned uses the optical
        approximation v = z * c. Strictly speaking, the barycentric correction is
        multiplicative and should be applied as::

          >>> from astropy.time import Time
          >>> from astropy.coordinates import SkyCoord, EarthLocation
          >>> from astropy.constants import c
          >>> t = Time(56370.5, format='mjd', scale='utc')
          >>> loc = EarthLocation('149d33m00.5s','-30d18m46.385s',236.87*u.m)
          >>> sc = SkyCoord(1*u.deg, 2*u.deg)
          >>> vcorr = sc.radial_velocity_correction(kind='barycentric', obstime=t, location=loc)  # doctest: +REMOTE_DATA
          >>> rv = rv + vcorr + rv * vcorr / c  # doctest: +SKIP

        Also note that this method returns the correction velocity in the so-called
        *optical convention*::

          >>> vcorr = zb * c  # doctest: +SKIP

        where ``zb`` is the barycentric correction redshift as defined in section 3
        of Wright & Eastman (2014). The application formula given above follows from their
        equation (11) under assumption that the radial velocity ``rv`` has also been defined
        using the same optical convention. Note, this can be regarded as a matter of
        velocity definition and does not by itself imply any loss of accuracy, provided
        sufficient care has been taken during interpretation of the results. If you need
        the barycentric correction expressed as the full relativistic velocity (e.g., to provide
        it as the input to another software which performs the application), the
        following recipe can be used::

          >>> zb = vcorr / c  # doctest: +REMOTE_DATA
          >>> zb_plus_one_squared = (zb + 1) ** 2  # doctest: +REMOTE_DATA
          >>> vcorr_rel = c * (zb_plus_one_squared - 1) / (zb_plus_one_squared + 1)  # doctest: +REMOTE_DATA

        or alternatively using just equivalencies::

          >>> vcorr_rel = vcorr.to(u.Hz, u.doppler_optical(1*u.Hz)).to(vcorr.unit, u.doppler_relativistic(1*u.Hz))  # doctest: +REMOTE_DATA

        See also `~astropy.units.doppler_optical`,
        `~astropy.units.doppler_radio`, and
        `~astropy.units.doppler_relativistic` for more information on
        the velocity conventions.

        The default is for this method to use the builtin ephemeris for
        computing the sun and earth location.  Other ephemerides can be chosen
        by setting the `~astropy.coordinates.solar_system_ephemeris` variable,
        either directly or via ``with`` statement.  For example, to use the JPL
        ephemeris, do::

          >>> from astropy.coordinates import solar_system_ephemeris
          >>> sc = SkyCoord(1*u.deg, 2*u.deg)
          >>> with solar_system_ephemeris.set('jpl'):  # doctest: +REMOTE_DATA
          ...     rv += sc.radial_velocity_correction(obstime=t, location=loc)  # doctest: +SKIP

        """
        # has to be here to prevent circular imports
        from .solar_system import get_body_barycentric_posvel

        # location validation
        timeloc = getattr(obstime, "location", None)
        if location is None:
            if self.location is not None:
                location = self.location
                if timeloc is not None:
                    raise ValueError(
                        "`location` cannot be in both the passed-in `obstime` and this"
                        " `SkyCoord` because it is ambiguous which is meant for the"
                        " radial_velocity_correction."
                    )
            elif timeloc is not None:
                location = timeloc
            else:
                raise TypeError(
                    "Must provide a `location` to radial_velocity_correction, either as"
                    " a SkyCoord frame attribute, as an attribute on the passed in"
                    " `obstime`, or in the method call."
                )

        elif self.location is not None or timeloc is not None:
            raise ValueError(
                "Cannot compute radial velocity correction if `location` argument is"
                " passed in and there is also a  `location` attribute on this SkyCoord"
                " or the passed-in `obstime`."
            )

        # obstime validation
        coo_at_rv_obstime = self  # assume we need no space motion for now
        if obstime is None:
            obstime = self.obstime
            if obstime is None:
                raise TypeError(
                    "Must provide an `obstime` to radial_velocity_correction, either as"
                    " a SkyCoord frame attribute or in the method call."
                )
        elif self.obstime is not None and self.data.differentials:
            # we do need space motion after all
            coo_at_rv_obstime = self.apply_space_motion(obstime)
        elif self.obstime is None and "s" in self.data.differentials:
            warnings.warn(
                "SkyCoord has space motion, and therefore the specified "
                "position of the SkyCoord may not be the same as "
                "the `obstime` for the radial velocity measurement. "
                "This may affect the rv correction at the order of km/s"
                "for very high proper motions sources. If you wish to "
                "apply space motion of the SkyCoord to correct for this"
                "the `obstime` attribute of the SkyCoord must be set",
                AstropyUserWarning,
            )

        pos_earth, v_origin_to_earth = get_body_barycentric_posvel("earth", obstime)
        if kind == "heliocentric":
            v_origin_to_earth -= get_body_barycentric_posvel("sun", obstime)[1]
        elif kind != "barycentric":
            raise ValueError(
                "`kind` argument to radial_velocity_correction must "
                f"be 'barycentric' or 'heliocentric', but got '{kind}'"
            )

        gcrs_p, gcrs_v = location.get_gcrs_posvel(obstime)
        # transforming to GCRS is not the correct thing to do here, since we don't want to
        # include aberration (or light deflection)? Instead, only apply parallax if necessary
        icrs_cart = coo_at_rv_obstime.icrs.cartesian
        targcart = icrs_cart.without_differentials()
        if self.data.__class__ is not UnitSphericalRepresentation:
            # SkyCoord has distances, so apply parallax by calculating
            # the direction of the target as seen by the observer.
            targcart -= pos_earth + gcrs_p
            targcart /= targcart.norm()

        if kind == "heliocentric":
            # Do a simpler correction than for barycentric ignoring time dilation and
            # gravitational redshift.  This is adequate since heliocentric corrections
            # shouldn't be used if cm/s precision is required.
            return targcart.dot(v_origin_to_earth + gcrs_v)

        beta_obs = (v_origin_to_earth + gcrs_v) / speed_of_light
        gamma_obs = 1 / np.sqrt(1 - beta_obs.norm() ** 2)
        gr = location.gravitational_redshift(obstime)
        # barycentric redshift according to eq 28 in Wright & Eastmann (2014),
        # neglecting Shapiro delay and effects of the star's own motion
        zb = gamma_obs * (1 + beta_obs.dot(targcart)) / (1 + gr / speed_of_light)
        # try and get terms corresponding to stellar motion.
        if icrs_cart.differentials:
            try:
                ro = self.icrs.cartesian
                beta_star = ro.differentials["s"].to_cartesian() / speed_of_light
                # ICRS unit vector at coordinate epoch
                ro = ro.without_differentials()
                ro /= ro.norm()
                zb *= (1 + beta_star.dot(ro)) / (1 + beta_star.dot(targcart))
            except u.UnitConversionError:
                warnings.warn(
                    "SkyCoord contains some velocity information, but not enough to"
                    " calculate the full space motion of the source, and so this"
                    " has been ignored for the purposes of calculating the radial"
                    " velocity correction. This can lead to errors on the order of"
                    " metres/second.",
                    AstropyUserWarning,
                )
        return (zb - 1) * speed_of_light

    # Table interactions
    @classmethod
    def guess_from_table(cls, table, **coord_kwargs):
        r"""
        A convenience method to create and return a new SkyCoord from the data
        in an astropy Table.

        This method matches table columns that start with the case-insensitive
        names of the components of the requested frames (including
        differentials), if they are also followed by a non-alphanumeric
        character. It will also match columns that *end* with the component name
        if a non-alphanumeric character is *before* it.

        For example, the first rule means columns with names like
        ``'RA[J2000]'`` or ``'ra'`` will be interpreted as ``ra`` attributes for
        `~astropy.coordinates.ICRS` frames, but ``'RAJ2000'`` or ``'radius'``
        are *not*. Similarly, the second rule applied to the
        `~astropy.coordinates.Galactic` frame means that a column named
        ``'gal_l'`` will be used as the ``l`` component, but ``gall`` or
        ``'fill'`` will not.

        The definition of alphanumeric here is based on Unicode's definition
        of alphanumeric, except without ``_`` (which is normally considered
        alphanumeric).  So for ASCII, this means the non-alphanumeric characters
        are ``<space>_!"#$%&'()*+,-./\:;<=>?@[]^`{|}~``).

        Parameters
        ----------
        table : `~astropy.table.Table` or subclass
            The table to load data from.
        **coord_kwargs
            Any additional keyword arguments are passed directly to this class's
            constructor.

        Returns
        -------
        newsc : `~astropy.coordinates.SkyCoord` or subclass
            The new instance.

        Raises
        ------
        ValueError
            If more than one match is found in the table for a component,
            unless the additional matches are also valid frame component names.
            If a "coord_kwargs" is provided for a value also found in the table.

        """
        _frame_cls, _frame_kwargs = _get_frame_without_data([], coord_kwargs)
        frame = _frame_cls(**_frame_kwargs)
        coord_kwargs["frame"] = coord_kwargs.get("frame", frame)

        representation_component_names = set(
            frame.get_representation_component_names()
        ).union(set(frame.get_representation_component_names("s")))

        comp_kwargs = {}
        for comp_name in representation_component_names:
            # this matches things like 'ra[...]'' but *not* 'rad'.
            # note that the "_" must be in there explicitly, because
            # "alphanumeric" usually includes underscores.
            starts_with_comp = comp_name + r"(\W|\b|_)"
            # this part matches stuff like 'center_ra', but *not*
            # 'aura'
            ends_with_comp = r".*(\W|\b|_)" + comp_name + r"\b"
            # the final regex ORs together the two patterns
            rex = re.compile(
                rf"({starts_with_comp})|({ends_with_comp})", re.IGNORECASE | re.UNICODE
            )

            # find all matches
            matches = {col_name for col_name in table.colnames if rex.match(col_name)}

            # now need to select among matches, also making sure we don't have
            # an exact match with another component
            if len(matches) == 0:  # no matches
                continue
            elif len(matches) == 1:  # only one match
                col_name = matches.pop()
            else:  # more than 1 match
                # try to sieve out other components
                matches -= representation_component_names - {comp_name}
                # if there's only one remaining match, it worked.
                if len(matches) == 1:
                    col_name = matches.pop()
                else:
                    raise ValueError(
                        f'Found at least two matches for component "{comp_name}":'
                        f' "{matches}". Cannot guess coordinates from a table with this'
                        " ambiguity."
                    )

            comp_kwargs[comp_name] = table[col_name]

        for k, v in comp_kwargs.items():
            if k in coord_kwargs:
                raise ValueError(
                    f'Found column "{v.name}" in table, but it was already provided as'
                    ' "{k}" keyword to guess_from_table function.'
                )
            coord_kwargs[k] = v

        return cls(**coord_kwargs)

    # Name resolve
    @classmethod
    def from_name(cls, name, frame="icrs", parse=False, cache=True):
        """
        Given a name, query the CDS name resolver to attempt to retrieve
        coordinate information for that object. The search database, sesame
        url, and  query timeout can be set through configuration items in
        ``astropy.coordinates.name_resolve`` -- see docstring for
        `~astropy.coordinates.get_icrs_coordinates` for more
        information.

        Parameters
        ----------
        name : str
            The name of the object to get coordinates for, e.g. ``'M42'``.
        frame : str or `BaseCoordinateFrame` class or instance
            The frame to transform the object to.
        parse : bool
            Whether to attempt extracting the coordinates from the name by
            parsing with a regex. For objects catalog names that have
            J-coordinates embedded in their names, e.g.,
            'CRTS SSS100805 J194428-420209', this may be much faster than a
            Sesame query for the same object name. The coordinates extracted
            in this way may differ from the database coordinates by a few
            deci-arcseconds, so only use this option if you do not need
            sub-arcsecond accuracy for coordinates.
        cache : bool, optional
            Determines whether to cache the results or not. To update or
            overwrite an existing value, pass ``cache='update'``.

        Returns
        -------
        coord : SkyCoord
            Instance of the SkyCoord class.
        """
        from .name_resolve import get_icrs_coordinates

        icrs_coord = get_icrs_coordinates(name, parse, cache=cache)
        icrs_sky_coord = cls(icrs_coord)
        if frame in ("icrs", icrs_coord.__class__):
            return icrs_sky_coord
        else:
            return icrs_sky_coord.transform_to(frame)
