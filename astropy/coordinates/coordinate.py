# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Framework and base class for Coordinate objects.
"""

__all__ = ["BaseCoordinate", "Coordinate"]

import copy
import operator
import warnings
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from astropy import units as u
from astropy.table import QTable
from astropy.utils import ShapedLikeNDArray
from astropy.utils.masked import MaskableShapedLikeNDArray, combine_masks

from .angles import Angle
from .representation import SphericalRepresentation


class BaseCoordinate(ABC):
    """
    The abstract base class for coordinate objects (dataless frame + coordinate data).

    This class is intended to be subclassed to create instances of specific
    systems.  Subclasses must expose a `~astropy.coordinates.BaseFrame` instance
    as ``.frame`` and a `~astropy.coordinates.BaseRepresentation` instance as
    ``.data``, and must implement `transform_to` and `__replace__`.
    """

    _frame_properties = {}

    @staticmethod
    def _get_frame_props(frame_name):
        """Look up registered properties for a frame."""
        if isinstance(frame_name, str):
            return BaseCoordinate._frame_properties.get(frame_name, {})
        for name in frame_name:
            props = BaseCoordinate._frame_properties.get(name, {})
            if props:
                return props
        return {}

    @staticmethod
    def register_property(frame_name):
        """Register a frame-specific property.

        Parameters
        ----------
        frame_name : str
            The ``name`` attribute of the frame this property belongs to
            (e.g., ``"altaz"``).

        Examples
        --------
        >>> @BaseCoordinate.register_property("myframe")
        ... def my_property(coord):
        ...     return coord.data
        """

        def decorator(func):
            BaseCoordinate._frame_properties.setdefault(frame_name, {})[
                func.__name__
            ] = func
            return func

        return decorator

    @property
    def frame_attributes(self):
        """Frame attributes for the coordinate frame."""
        return type(self.frame).frame_attributes

    @property
    @abstractmethod
    def frame(self):
        """The coordinate reference frame."""

    @property
    @abstractmethod
    def data(self):
        """The coordinate data as a `~astropy.coordinates.BaseRepresentation`."""

    @abstractmethod
    def transform_to(self, new_frame, merge_attributes=True):
        """
        Transform this coordinate to a new frame.

        Parameters
        ----------
        new_frame : `~astropy.coordinates.BaseFrame` instance
            The target dataless frame.
        merge_attributes : bool, optional
            If ``True``, attributes of the current frame that are
            not in ``new_frame`` (or that ``new_frame`` left at their
            defaults) are propagated. ``False`` ignores source
            attributes and uses only those of ``new_frame``.

        Returns
        -------
        coord : `BaseCoordinate` subclass instance
            The coordinate in the new frame.
        """

    @abstractmethod
    def __replace__(self, **changes):
        """
        Return a copy with specified attributes replaced.

        Parameters
        ----------
        **changes
            Attributes to override. Subclasses define valid attribute names.

        Returns
        -------
        coord : same type as self
            A new instance with the specified attributes replaced.
        """

    @property
    def representation_type(self):
        """
        The representation used for this coordinate's data.
        """
        return self.frame.representation_type

    @property
    def differential_type(self):
        """
        The differential used for this coordinate's velocity data.
        """
        return self.frame.get_representation_cls("s")

    @property
    def representation_component_names(self):
        """
        A dictionary mapping component names (e.g. ``ra``, ``dec``) to the
        corresponding attribute names on the representation class.

        Delegates to the underlying frame.
        """
        return self.frame.representation_component_names

    def get_representation_component_names(self, which="base"):
        """
        Return the component names for the requested representation.

        Delegates to the underlying frame.  ``which`` may be ``'base'``
        (position components) or ``'s'`` (velocity / differential components).
        """
        return self.frame.get_representation_component_names(which)

    @property
    def representation_component_units(self):
        """
        A dictionary mapping component names to their units for the current
        representation.  Delegates to the underlying frame.
        """
        return self.frame.get_representation_component_units()

    def get_representation_component_units(self, which="base"):
        """
        Return the component units for the requested representation.

        Delegates to the underlying frame.  ``which`` may be ``'base'``
        (position components) or ``'s'`` (velocity / differential components).
        """
        return self.frame.get_representation_component_units(which)

    def represent_as(self, base, s="base", in_frame_units=False):
        """
        Generate and return a new representation of this coordinate's `data`
        as a `~astropy.coordinates.BaseRepresentation` object.

        Parameters
        ----------
        base : `~astropy.coordinates.BaseRepresentation` subclass or str
            The type of representation to generate.  Must be a class
            (not an instance), or the string name of the representation class.
        s : `~astropy.coordinates.BaseDifferential` subclass, str, optional
            Class in which any velocities should be represented.  Must be a
            class (not an instance), or the string name of the differential
            class.  If equal to ``'base'``, inferred from the base
            class.  If `None`, all velocity information is dropped.
        in_frame_units : bool, optional
            If `True`, convert the representation to the units preferred by
            this frame (as specified in ``frame_specific_representation_info``).

        Returns
        -------
        newrep : `~astropy.coordinates.BaseRepresentation` subclass instance
            A new representation object of this coordinate's `data`.
        """
        from astropy.coordinates.baseframe import (
            BaseCoordinateFrame,
            _get_repr_classes,
        )

        # TODO: APE23: simplify when BaseCoordinateFrame deprecated
        frame = self.frame
        if (
            isinstance(frame, BaseCoordinateFrame)
            and type(frame).represent_as is not BaseCoordinateFrame.represent_as
        ):
            realized = frame.realize_frame(self.data)
            return realized.represent_as(base, s=s, in_frame_units=in_frame_units)

        repr_classes = _get_repr_classes(base=base, s=s)
        repr_cls = repr_classes["base"]

        # Determine the differential class to use
        diff_cls = None
        if "s" in self.data.differentials:
            if s == "base":
                existing = self.data.differentials["s"].__class__
                if existing in repr_cls._compatible_differentials:
                    diff_cls = existing
                else:
                    diff_cls = repr_classes["s"]
            elif s is not None:
                diff_cls = repr_classes["s"]

        # Convert to the target representation
        rep = (
            self.data.represent_as(repr_cls, diff_cls)
            if diff_cls
            else self.data.represent_as(repr_cls)
        )

        if not in_frame_units:
            return rep

        repr_info = self.frame.representation_info

        # Save differential before rebuilding positional rep (unit conversion strips it)
        diff = rep.differentials.get("s") if diff_cls else None

        # Apply frame units to the positional part
        if pos_info := repr_info.get(repr_cls):
            datakwargs = {comp: getattr(rep, comp) for comp in rep.components}
            for comp, unit in zip(rep.components, pos_info["units"]):
                if unit:
                    datakwargs[comp] = datakwargs[comp].to(unit)
            rep = rep.__class__(copy=False, **datakwargs)

        # Apply frame units to the differential part
        if diff_cls:
            from astropy.coordinates import representation as r

            orig_diff = self.data.differentials["s"]
            if diff_info := repr_info.get(diff_cls):
                diffkwargs = {comp: getattr(diff, comp) for comp in diff.components}
                for comp, unit in zip(diff.components, diff_info["units"]):
                    if (
                        isinstance(
                            orig_diff,
                            (
                                r.UnitSphericalDifferential,
                                r.UnitSphericalCosLatDifferential,
                                r.RadialDifferential,
                            ),
                        )
                        and comp not in orig_diff.__class__.attr_classes
                    ):
                        continue
                    if unit and hasattr(diff, comp):
                        try:
                            diffkwargs[comp] = diffkwargs[comp].to(unit)
                        except Exception:
                            pass
                diff = diff.__class__(copy=False, **diffkwargs)

            rep._differentials.update({"s": diff})

        return rep

    @property
    def cartesian(self):
        """
        A Cartesian representation of the coordinates in this object.
        """
        # TODO: if representations are updated to use a full transform graph,
        #       the representation aliases should not be hard-coded like this
        return self.represent_as("cartesian", in_frame_units=True)

    @property
    def cylindrical(self):
        """
        A cylindrical representation of the coordinates in this object.
        """
        # TODO: if representations are updated to use a full transform graph,
        #       the representation aliases should not be hard-coded like this
        return self.represent_as("cylindrical", in_frame_units=True)

    @property
    def spherical(self):
        """
        A spherical representation of the coordinates in this object.
        """
        # TODO: if representations are updated to use a full transform graph,
        #       the representation aliases should not be hard-coded like this
        return self.represent_as("spherical", in_frame_units=True)

    @property
    def sphericalcoslat(self):
        """
        A spherical representation of the positional data and a
        `~astropy.coordinates.SphericalCosLatDifferential` for the velocity
        data in this object.
        """
        # TODO: if representations are updated to use a full transform graph,
        #       the representation aliases should not be hard-coded like this
        return self.represent_as("spherical", "sphericalcoslat", in_frame_units=True)

    @property
    def velocity(self):
        """
        Retrieve the Cartesian space-motion as a
        `~astropy.coordinates.CartesianDifferential` object.

        This is equivalent to calling ``self.cartesian.differentials['s']``.
        """
        if "s" not in self.data.differentials:
            raise ValueError(
                "Coordinate has no associated velocity (Differential) data information."
            )
        return self.cartesian.differentials["s"]

    @property
    def proper_motion(self):
        """
        The two-dimensional proper motion as a
        `~astropy.units.Quantity` object with angular velocity units. In the
        returned `~astropy.units.Quantity`, ``axis=0`` is the longitude/latitude
        dimension so that ``.proper_motion[0]`` is the longitudinal proper
        motion and ``.proper_motion[1]`` is latitudinal. The longitudinal proper
        motion already includes the cos(latitude) term.
        """
        if "s" not in self.data.differentials:
            raise ValueError(
                "Coordinate has no associated velocity (Differential) data information."
            )

        sph = self.represent_as("spherical", "sphericalcoslat", in_frame_units=True)
        pm_lon = sph.differentials["s"].d_lon_coslat
        pm_lat = sph.differentials["s"].d_lat
        return (
            np.stack((pm_lon.value, pm_lat.to(pm_lon.unit).value), axis=0) * pm_lon.unit
        )

    @property
    def radial_velocity(self):
        """
        The radial or line-of-sight velocity as a
        `~astropy.units.Quantity` object.
        """
        if "s" not in self.data.differentials:
            raise ValueError(
                "Coordinate has no associated velocity (Differential) data information."
            )
        sph = self.represent_as("spherical", in_frame_units=True)
        return sph.differentials["s"].d_distance

    def is_transformable_to(self, new_frame):
        """
        Determines if this coordinate frame can be transformed to another given frame.

        Parameters
        ----------
        new_frame : frame class, frame object, or str
            The proposed frame to transform into.

        Returns
        -------
        transformable : bool or str
            `True` if this can be transformed to ``new_frame``, `False` if
            not, or the string 'same' if ``new_frame`` is the same system as
            this object but no transformation is defined.

        Notes
        -----
        A return value of 'same' means the transformation will work, but it will
        just give back a copy of this object.  The intended usage is::

            if coord.is_transformable_to(some_unknown_frame):
                coord2 = coord.transform_to(some_unknown_frame)

        This will work even if ``some_unknown_frame``  turns out to be the same
        frame class as ``coord``.  This is intended for cases where the frame
        is the same regardless of the frame attributes (e.g. ICRS), but be
        aware that it *might* also indicate that someone forgot to define the
        transformation between two objects of the same frame class but with
        different attributes.
        """
        # TODO! like matplotlib, do string overrides for modified methods
        from .sky_coordinate_parsers import _get_frame_class

        new_frame = (
            _get_frame_class(new_frame) if isinstance(new_frame, str) else new_frame
        )
        return self.frame.is_transformable_to(new_frame)

    @property
    def size(self):
        """The number of coordinate values in this object."""
        return self.data.size

    def to_table(self) -> QTable:
        """
        Convert this coordinate to a |QTable|.

        Any attributes that have the same length as the coordinate will be
        converted to columns of the |QTable|. All other attributes will be
        recorded as metadata.

        Returns
        -------
        `~astropy.table.QTable`
            A |QTable| containing the data of this coordinate.

        Examples
        --------
        >>> from astropy import units as u
        >>> from astropy.coordinates import Coordinate, ICRSFrame, UnitSphericalRepresentation
        >>> coord = Coordinate(
        ...     frame=ICRSFrame(),
        ...     data=UnitSphericalRepresentation(lon=[40, 70]*u.deg, lat=[0, -20]*u.deg),
        ... )
        >>> t = coord.to_table()
        >>> t
        <QTable length=2>
           ra     dec
          deg     deg
        float64 float64
        ------- -------
           40.0     0.0
           70.0   -20.0
        >>> t.meta
        {'representation_type': 'spherical'}
        """
        from astropy.table import QTable

        columns = {}
        metadata = {}
        # Record attributes that have the same length as self as columns in the
        # table, and the other attributes as table metadata.  This matches
        # table.serialize._represent_mixin_as_column().
        for key, value in self.info._represent_as_dict().items():
            if getattr(value, "shape", ())[:1] == (len(self),):
                columns[key] = value
            else:
                metadata[key] = value
        return QTable(columns, meta=metadata)

    def is_equivalent_frame(self, other):
        """
        Checks if this object's frame is the same as that of the ``other`` object.

        To be the same frame, two objects must be the same frame class and have
        the same frame attributes.

        Parameters
        ----------
        other : `~astropy.coordinates.BaseCoordinate`
            The other coordinate to check.

        Returns
        -------
        isequiv : bool
            True if the frames are the same.

        Raises
        ------
        TypeError
            If ``other`` isn't a `~astropy.coordinates.BaseCoordinate` subclass.
        """
        if isinstance(other, BaseCoordinate):
            return self.frame.is_equivalent_frame(other.frame)
        raise TypeError(
            "Tried to do is_equivalent_frame on something that isn't frame-like"
        )

    def __eq__(self, value):
        """Equality operator.

        This implements strict equality and requires that the frames are
        equivalent and that the representation data are exactly equal.
        """
        if not isinstance(value, BaseCoordinate):
            return NotImplemented

        if not self.is_equivalent_frame(value):
            raise TypeError(
                "cannot compare: objects must have equivalent frames: "
                f"{self.frame} vs. {value.frame}"
            )

        return self.data == value.data

    def __ne__(self, value):
        return np.logical_not(self == value)

    def __dir__(self):
        """
        Override the builtin `dir` to include representation
        and differential component names.

        TODO: dynamic representation transforms (i.e. include cylindrical et al.).
        """
        dir_values = (
            set(super().__dir__())
            | set(self.frame.representation_component_names)
            | set(self.frame.get_representation_component_names("s"))
        )
        # Include registered frame-specific properties.
        dir_values.update(BaseCoordinate._get_frame_props(self.frame.name))
        return sorted(dir_values)

    @staticmethod
    def _apply_to_frame(frame, method, *args, **kwargs):
        """Return a new frame with ``method`` applied to shaped attributes.

        Iterates over the frame's non-default attributes and applies the
        shape-changing *method* to any that have a shape.  Returns a new
        frame instance when any attribute was modified, or the original
        frame unchanged.
        """

        def apply_method(value):
            if isinstance(value, ShapedLikeNDArray):
                return value._apply(method, *args, **kwargs)
            elif callable(method):
                return method(value, *args, **kwargs)
            else:
                return getattr(value, method)(*args, **kwargs)

        frame_attrs = {}
        needs_new_frame = False
        for attr in type(frame).frame_attributes:
            if frame.is_frame_attr_default(attr):
                continue
            value = getattr(frame, attr)
            if getattr(value, "shape", ()):
                value = apply_method(value)
                needs_new_frame = True
            elif method == "copy" or method == "flatten":
                value = copy.copy(value)
                needs_new_frame = True
            frame_attrs[attr] = value

        if needs_new_frame:
            return type(frame)(
                representation_type=frame.representation_type,
                differential_type=frame.differential_type,
                **frame_attrs,
            )
        return frame

    def to_string(self, style="decimal", **kwargs):
        """
        A string representation of the coordinates.

        The default styles definitions are::

          'decimal': 'lat': {'decimal': True, 'unit': "deg"}
                     'lon': {'decimal': True, 'unit': "deg"}
          'dms': 'lat': {'unit': "deg"}
                 'lon': {'unit': "deg"}
          'hmsdms': 'lat': {'alwayssign': True, 'pad': True, 'unit': "deg"}
                    'lon': {'pad': True, 'unit': "hour"}

        See :meth:`~astropy.coordinates.Angle.to_string` for details and
        keyword arguments (the two angles forming the coordinates are are
        both :class:`~astropy.coordinates.Angle` instances). Keyword
        arguments have precedence over the style defaults and are passed
        to :meth:`~astropy.coordinates.Angle.to_string`.

        Parameters
        ----------
        style : {'hmsdms', 'dms', 'decimal'}
            The formatting specification to use. These encode the three most
            common ways to represent coordinates. The default is `decimal`.
        **kwargs
            Keyword args passed to :meth:`~astropy.coordinates.Angle.to_string`.
        """
        sph_coord = self.represent_as(SphericalRepresentation)

        styles = {
            "hmsdms": {
                "lonargs": {"unit": u.hour, "pad": True},
                "latargs": {"unit": u.degree, "pad": True, "alwayssign": True},
            },
            "dms": {"lonargs": {"unit": u.degree}, "latargs": {"unit": u.degree}},
            "decimal": {
                "lonargs": {"unit": u.degree, "decimal": True},
                "latargs": {"unit": u.degree, "decimal": True},
            },
        }

        lonargs = {}
        latargs = {}

        if style in styles:
            lonargs.update(styles[style]["lonargs"])
            latargs.update(styles[style]["latargs"])
        else:
            raise ValueError(f"Invalid style.  Valid options are: {','.join(styles)}")

        lonargs.update(kwargs)
        latargs.update(kwargs)

        if np.isscalar(sph_coord.lon.value):
            coord_string = (
                f"{sph_coord.lon.to_string(**lonargs)}"
                f" {sph_coord.lat.to_string(**latargs)}"
            )
        else:
            coord_string = []
            for lonangle, latangle in zip(sph_coord.lon.ravel(), sph_coord.lat.ravel()):
                coord_string += [
                    f"{lonangle.to_string(**lonargs)} {latangle.to_string(**latargs)}"
                ]
            if len(sph_coord.shape) > 1:
                coord_string = np.array(coord_string).reshape(sph_coord.shape)

        return coord_string

    @property
    def masked(self):
        """Whether the underlying data is masked.

        Raises
        ------
        ValueError
            If the coordinate has no associated data.
        """
        return self.data.masked

    def get_mask(self, *attrs):
        """Get the mask associated with these coordinates.

        Parameters
        ----------
        *attrs : str
            Attributes whose masks to combine. By default, get the
            combined mask of all components (including from differentials),
            ignoring possible masks of attributes.

        Returns
        -------
        mask : ~numpy.ndarray of bool
            The combined, read-only mask. If the instance is not masked, it
            is an array of `False` with the correct shape.

        Raises
        ------
        ValueError
            If the coordinate has no associated data.
        """
        if attrs:
            values = operator.attrgetter(*attrs)(self)
            if not isinstance(values, tuple):
                values = (values,)
            masks = [getattr(v, "mask", None) for v in values]
        elif self.data.masked:
            masks = [diff.mask for diff in self.data.differentials.values()]
            masks.append(self.data.mask)
        else:
            masks = []

        # Broadcast makes it readonly too.
        return np.broadcast_to(combine_masks(masks), self.shape)

    @property
    def mask(self):
        """The mask associated with these coordinates.

        Combines the masks of all components of the underlying representation,
        including possible differentials.
        """
        return self.get_mask()

    def _prepare_unit_sphere_coords(self, other, origin_mismatch):
        from . import representation as r
        from .baseframe import frame_transform_graph
        from .errors import (
            NonRotationTransformationError,
            NonRotationTransformationWarning,
        )
        from .transformations import DynamicMatrixTransform, StaticMatrixTransform

        other_frame = getattr(other, "frame", other)
        if not (
            origin_mismatch == "ignore"
            or self.frame.is_equivalent_frame(other_frame)
            or all(
                isinstance(comp, (StaticMatrixTransform, DynamicMatrixTransform))
                for comp in frame_transform_graph.get_transform(
                    type(self.frame), type(other_frame)
                ).transforms
            )
        ):
            if origin_mismatch == "warn":
                warnings.warn(NonRotationTransformationWarning(self.frame, other_frame))
            elif origin_mismatch == "error":
                raise NonRotationTransformationError(self.frame, other_frame)
            else:
                raise ValueError(
                    f"{origin_mismatch=} is invalid. Allowed values are 'ignore', "
                    "'warn' or 'error'."
                )

        # TODO: APE23: simplify when BaseCoordinateFrame deprecated
        self_sph = self.represent_as(r.UnitSphericalRepresentation)
        from .baseframe import BaseCoordinateFrame

        if self.frame.is_equivalent_frame(other_frame):
            # Same frame: no transform needed; use other directly.
            other_in_self = other
        elif isinstance(other_frame, BaseCoordinateFrame) and other_frame.has_data:
            other_in_self = other_frame.transform_to(self.frame)
        else:
            other_in_self = other.transform_to(self.frame, merge_attributes=False)
        other_sph = other_in_self.represent_as(r.UnitSphericalRepresentation)
        return self_sph.lon, self_sph.lat, other_sph.lon, other_sph.lat

    def position_angle(self, other: "BaseCoordinate") -> Angle:
        """Compute the on-sky position angle to another coordinate.

        Parameters
        ----------
        other : `~astropy.coordinates.BaseCoordinate`
            The other coordinate to compute the position angle to.  It is
            treated as the "head" of the vector of the position angle.

        Returns
        -------
        `~astropy.coordinates.Angle`
            The (positive) position angle of the vector pointing from ``self``
            to ``other``, measured East from North.  If either ``self`` or
            ``other`` contain arrays, this will be an array following the
            appropriate `numpy` broadcasting rules.

        Examples
        --------
        >>> from astropy import units as u
        >>> from astropy.coordinates import ICRS, SkyCoord
        >>> c1 = SkyCoord(0*u.deg, 0*u.deg)
        >>> c2 = ICRS(1*u.deg, 0*u.deg)
        >>> c1.position_angle(c2).to(u.deg)
        <Angle 90. deg>
        >>> c2.position_angle(c1).to(u.deg)
        <Angle 270. deg>
        >>> c3 = SkyCoord(1*u.deg, 1*u.deg)
        >>> c1.position_angle(c3).to(u.deg)  # doctest: +FLOAT_CMP
        <Angle 44.995636455344844 deg>
        """
        from .angles import position_angle as _position_angle

        return _position_angle(*self._prepare_unit_sphere_coords(other, "ignore"))

    def separation(
        self,
        other: "BaseCoordinate",
        *,
        origin_mismatch: Literal["ignore", "warn", "error"] = "warn",
    ) -> Angle:
        """
        Compute on-sky separation between this coordinate and another.

        For more on how to use this (and related) functionality, see the
        examples in :ref:`astropy-coordinates-separations-matching`.

        Parameters
        ----------
        other : `~astropy.coordinates.BaseCoordinate`
            The coordinate to get the separation to.
        origin_mismatch : {"warn", "ignore", "error"}, keyword-only
            If the ``other`` coordinates are in a different frame then they
            will have to be transformed, and if the transformation is not a
            pure rotation then ``self.separation(other)`` can be
            different from ``other.separation(self)``. With
            ``origin_mismatch="warn"`` (default) the transformation is
            always performed, but a warning is emitted if it is not a
            pure rotation. If ``origin_mismatch="ignore"`` then the
            required transformation is always performed without warnings.
            If ``origin_mismatch="error"`` then only transformations
            that are pure rotations are allowed.

        Returns
        -------
        sep : `~astropy.coordinates.Angle`
            The on-sky separation between this and the ``other`` coordinate.

        Notes
        -----
        The separation is calculated using the Vincenty formula, which
        is stable at all locations, including poles and antipodes [1]_.

        .. [1] https://en.wikipedia.org/wiki/Great-circle_distance

        """
        from .angles import Angle, angular_separation

        return Angle(
            angular_separation(
                *self._prepare_unit_sphere_coords(other, origin_mismatch)
            ),
            unit=u.degree,
        )

    def separation_3d(self, other):
        """
        Computes three dimensional separation between this coordinate
        and another.

        For more on how to use this (and related) functionality, see the
        examples in :ref:`astropy-coordinates-separations-matching`.

        Parameters
        ----------
        other : `~astropy.coordinates.BaseCoordinate`
            The coordinate system to get the distance to.

        Returns
        -------
        sep : `~astropy.coordinates.Distance`
            The real-space distance between these two coordinates.

        Raises
        ------
        ValueError
            If this or the other coordinate do not have distances.
        """
        from . import representation as r
        from .distances import Distance

        if isinstance(self.data, r.UnitSphericalRepresentation):
            raise ValueError(
                "This object does not have a distance; cannot compute 3d separation."
            )

        # do this first just in case the conversion somehow creates a distance
        _other_frame = getattr(other, "frame", other)
        if hasattr(_other_frame, "transform_to") and getattr(
            _other_frame, "has_data", False
        ):
            other = _other_frame.transform_to(self)
        else:
            other = other.transform_to(self, merge_attributes=False)

        if isinstance(other, r.UnitSphericalRepresentation):
            raise ValueError(
                "The other object does not have a distance; "
                "cannot compute 3d separation."
            )

        # drop the differentials to ensure they don't do anything odd in the
        # subtraction
        dist = (
            self.data.without_differentials().represent_as(r.CartesianRepresentation)
            - other.data.without_differentials().represent_as(r.CartesianRepresentation)
        ).norm()
        return dist if dist.unit == u.one else Distance(dist)

    def _data_repr(self):
        """Returns a string representation of the coordinate data.

        Produces the data portion of the ``__repr__`` string, with
        frame-specific component names substituted for the generic
        representation names (e.g. ``ra`` instead of ``lon``).
        """
        from astropy.coordinates import representation as r

        rep_cls = self.frame.representation_type
        if rep_cls is None:
            return repr(self.data)

        if isinstance(self.data, getattr(rep_cls, "_unit_representation", ())):
            rep_cls = self.data.__class__

        dif_cls = None
        if "s" in self.data.differentials:
            dif_cls = self.frame.get_representation_cls("s")
            dif_data = self.data.differentials["s"]
            if isinstance(
                dif_data,
                (
                    r.UnitSphericalDifferential,
                    r.UnitSphericalCosLatDifferential,
                    r.RadialDifferential,
                ),
            ):
                dif_cls = dif_data.__class__

        data = self.represent_as(rep_cls, dif_cls, in_frame_units=True)

        data_repr = repr(data)
        part1, _, remainder = data_repr.partition("(")
        if remainder:
            comp_str, part2 = remainder.split(")", 1)
            invnames = {
                nmrepr: nmpref
                for nmpref, nmrepr in self.frame.representation_component_names.items()
            }
            comp_names = (invnames.get(name, name) for name in comp_str.split(", "))
            data_repr = f"{part1}({', '.join(comp_names)}){part2}"

        if data_repr.startswith(class_prefix := f"<{type(data).__name__} "):
            data_repr = data_repr.removeprefix(class_prefix).removesuffix(">")
        else:
            data_repr = "Data:\n" + data_repr

        if "s" not in self.data.differentials:
            return data_repr

        data_repr_spl = data_repr.split("\n")
        first, *middle, last = repr(data.differentials["s"]).split("\n")
        if first.startswith("<"):
            first = " " + first.split(" ", 1)[1]
        for frm_nm, rep_nm in self.frame.get_representation_component_names(
            "s"
        ).items():
            first = first.replace(rep_nm, frm_nm)
        data_repr_spl[-1] = "\n".join((first, *middle, last.removesuffix(">")))
        return "\n".join(data_repr_spl)


class Coordinate(BaseCoordinate, MaskableShapedLikeNDArray):
    """
    Lightweight and performant coordinate: a dataless frame instance and a
    `~astropy.coordinates.BaseRepresentation`.

    Unlike `~astropy.coordinates.SkyCoord`, this class does not cache transforms,
    carry extra frame attributes, or perform flexible input parsing.

    Parameters
    ----------
    frame : `~astropy.coordinates.BaseFrame` instance
        A dataless frame instance.
    data : `~astropy.coordinates.BaseRepresentation` subclass instance
        The coordinate data.
    """

    # TODO: APE23: remove when BaseCoordinateFrame is deprecated
    has_data = True

    def __init__(self, frame, data):
        self._frame = frame
        self._data = data

    @property
    def frame(self):
        return self._frame

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self.data.shape

    def _apply(self, method, *args, **kwargs):
        """Return a new `Coordinate` with ``method`` applied to the data
        and shaped frame attributes.
        """
        return Coordinate(
            frame=self._apply_to_frame(self.frame, method, *args, **kwargs),
            data=self.data._apply(method, *args, **kwargs),
        )

    def __repr__(self):
        return (
            f"<Coordinate frame={self.frame.__class__.__name__}, "
            f"representation={self.data.__class__.__name__}>"
        )

    def __replace__(self, **changes):
        """Return a copy with specified fields replaced."""
        return Coordinate(
            frame=changes.get("frame", self.frame),
            data=changes.get("data", self.data),
        )

    def __getattr__(self, attr):
        """
        Allow access to frame attributes, representation components, and
        differential components.
        """
        # Prevent infinite recursion
        if attr.startswith("_"):
            return self.__getattribute__(attr)

        # Frame attribute lookup (e.g., equinox, obstime)
        if attr in type(self.frame).frame_attributes:
            return getattr(self.frame, attr)

        # Representation component lookup (e.g., ra, dec)
        repr_names = self.frame.representation_component_names
        if attr in repr_names:
            rep = self.represent_as(self.frame.representation_type, in_frame_units=True)
            return getattr(rep, repr_names[attr])

        # Differential component lookup (e.g., pm_ra_cosdec, pm_dec)
        diff_names = self.frame.get_representation_component_names("s")
        if attr in diff_names:
            rep = self.represent_as(
                in_frame_units=True, **self.frame.get_representation_cls(None)
            )
            return getattr(rep.differentials["s"], diff_names[attr])

        # Registered frame-specific properties (e.g., AltAz secz, ITRS earth_location).
        frame_props = BaseCoordinate._get_frame_props(self.frame.name)
        if attr in frame_props:
            return frame_props[attr](self)

        return self.__getattribute__(attr)

    def transform_to(self, new_frame, merge_attributes=True):
        """
        Transform to a new frame.

        Parameters
        ----------
        new_frame : `~astropy.coordinates.BaseFrame` instance
            The dataless frame to transform into.
        merge_attributes : bool, optional
            Accepted for API compatibility; not used by ``Coordinate`` (which
            always uses ``new_frame``'s attributes exactly).

        Returns
        -------
        Coordinate: `Coordinate` instance
            A new object with the coordinate data represented in the ``new_frame``
            system.
        """
        new_data = self.frame.transform_data_to(new_frame, self.data)
        return Coordinate(frame=new_frame, data=new_data)

    def __setitem__(self, item, value):
        if value is np.ma.masked or value is np.ma.nomask:
            self._data.__setitem__(item, value)
            return

        if self.__class__ is not value.__class__:
            raise TypeError(
                "can only set from object of same class: "
                f"{self.__class__.__name__} vs. {value.__class__.__name__}"
            )

        if not self._frame.is_equivalent_frame(value._frame):
            raise ValueError("cannot set: frames are not equivalent")

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
            clsnm = type(self._frame).__name__.removesuffix("Frame")
            raise TypeError(
                f"scalar '{clsnm}' frame object does not support item assignment"
            )

        self._data[item] = value._data

    def insert(self, obj, values, axis=0):
        """
        Make a copy with coordinate values inserted before the given indices.

        Parameters
        ----------
        obj : int
            Integer index before which ``values`` is inserted.
        values : array-like
            Value(s) to insert.  If the type of ``values`` is different
            from that of quantity, ``values`` is converted to the matching type.
        axis : int, optional
            Axis along which to insert ``values``.  Default is 0, which is the
            only allowed value and will insert a row.

        Returns
        -------
        coord : `Coordinate`
            Copy of instance with new values inserted.
        """
        return self.info._insert(obj, values, axis)
