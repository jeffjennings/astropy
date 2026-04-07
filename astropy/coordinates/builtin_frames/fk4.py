# Licensed under a 3-clause BSD style license - see LICENSE.rst

import math

import erfa
import numpy as np

from astropy import units as u
from astropy.coordinates.attributes import TimeAttribute
from astropy.coordinates.baseframe import BaseCoordinateFrame, base_doc, base_doc_frame, frame_transform_graph
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.coordinates.representation import (
    CartesianRepresentation,
    UnitSphericalRepresentation,
)
from astropy.coordinates.transformations import (
    DynamicMatrixTransform,
    FunctionTransformWithFiniteDifference,
    RepresentationFunctionTransform,
)

from astropy.time import Time
from astropy.utils.decorators import format_doc

from .baseradec import BaseRADecFrame, doc_components
from .utils import EQUINOX_B1950

__all__ = ["FK4", "FK4NoETerms", "FK4Frame", "FK4NoETermsFrame"]

jd1950 = Time("B1950").jd

doc_footer_fk4 = """
    Other parameters
    ----------------
    equinox : `~astropy.time.Time`
        The equinox of this frame.
    obstime : `~astropy.time.Time`
        The time this frame was observed.  If ``None``, will be the same as
        ``equinox``.
"""


@format_doc(base_doc_frame, footer=doc_footer_fk4)
class FK4Frame(BaseRADecFrame):
    """
    A coordinate or frame in the FK4 system.

    Note that this is a barycentric version of FK4 - that is, the origin for
    this frame is the Solar System Barycenter, *not* the Earth geocenter.

    The frame attributes are listed under **Other Parameters**.

    NOTE:
    This class only holds metadata defining the FK4 reference frame.
    It does not store coordinate data. To store coordinate data in this frame,
    use `~astropy.coordinates.Coordinate`, `~astropy.coordinates.SkyCoord` or the
    legacy `FK4` class.    
    """

    name = "fk4"

    equinox = TimeAttribute(default=EQUINOX_B1950, doc="The equinox time")
    obstime = TimeAttribute(
        default=None,
        secondary_attribute="equinox",
        doc="The reference time (e.g., time of observation)",
    )


@format_doc(base_doc, components=doc_components, footer=doc_footer_fk4)
class FK4(BaseCoordinateFrame, FK4Frame):
    """
    A coordinate or frame in the FK4 system.

    Note that this is a barycentric version of FK4 - that is, the origin for
    this frame is the Solar System Barycenter, *not* the Earth geocenter.

    The frame attributes are listed under **Other Parameters**.
    """
    pass


# the "self" transform


@frame_transform_graph.transform(RepresentationFunctionTransform, FK4Frame, FK4Frame)
def fk4_to_fk4(fk4frame1, fk4frame2):
    # deceptively complicated: need to transform to No E-terms FK4, precess, and
    # then come back, because precession is non-trivial with E-terms
    needs_precess = fk4frame1.equinox != fk4frame2.equinox

    def converter(rep):
        fk4coord1 = FK4(rep, equinox=fk4frame1.equinox, obstime=fk4frame1.obstime)
        fnoe_w_eqx1 = fk4coord1.transform_to(FK4NoETerms(equinox=fk4coord1.equinox))
        if needs_precess:
            fnoe_w_eqx1 = fnoe_w_eqx1.transform_to(FK4NoETerms(equinox=fk4frame2.equinox))
        return fnoe_w_eqx1.transform_to(
            FK4(equinox=fk4frame2.equinox, obstime=fk4frame2.obstime)
        ).data

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, FK4, FK4)
def fk4_to_fk4_legacy(fk4coord1, fk4frame2):
    converter = fk4_to_fk4(fk4coord1, fk4frame2)
    return fk4frame2.realize_frame(converter(fk4coord1.data))


@format_doc(base_doc_frame, footer=doc_footer_fk4)
class FK4NoETermsFrame(BaseRADecFrame):
    """
    A coordinate or frame in the FK4 system, but with the E-terms of aberration
    removed.

    The frame attributes are listed under **Other Parameters**.

    NOTE:
    This class only holds metadata defining the FK4NoETerms reference frame.
    It does not store coordinate data. To store coordinate data in this frame,
    use `~astropy.coordinates.Coordinate`, `~astropy.coordinates.SkyCoord` or the
    legacy `FK4NoETerms` class.      
    """

    name = "fk4noe"

    equinox = TimeAttribute(default=EQUINOX_B1950, doc="The equinox time")
    obstime = TimeAttribute(
        default=None,
        secondary_attribute="equinox",
        doc="The reference time (e.g., time of observation)",
    )

    @staticmethod
    def _precession_matrix(oldequinox, newequinox):
        """
        Compute and return the precession matrix for FK4 using Newcomb's method.
        Used inside some of the transformation functions.

        Parameters
        ----------
        oldequinox : `~astropy.time.Time`
            The equinox to precess from.
        newequinox : `~astropy.time.Time`
            The equinox to precess to.

        Returns
        -------
        newcoord : array
            The precession matrix to transform to the new equinox
        """
        # tropical years
        t1 = (oldequinox.byear - 1850.0) / 1000.0
        dt = (newequinox.byear - 1850.0) / 1000.0 - t1
        dt_over_3600 = dt / 3600

        z1 = zeta1 = (0.060 * t1 + 139.720) * t1 + 23035.545
        zeta2 = -0.27 * t1 + 30.240
        zeta = ((17.995 * dt + zeta2) * dt + zeta1) * dt_over_3600

        z2 = 109.480 + 0.39 * t1
        z = ((18.325 * dt + z2) * dt + z1) * dt_over_3600

        theta1 = (-0.37 * t1 - 85.29) * t1 + 20051.12
        theta2 = -0.37 * t1 - 42.65
        theta = ((-41.8 * dt + theta2) * dt + theta1) * dt_over_3600

        return (
            rotation_matrix(-z, "z")
            @ rotation_matrix(theta, "y")
            @ rotation_matrix(-zeta, "z")
        )


@format_doc(base_doc, components=doc_components, footer=doc_footer_fk4)
class FK4NoETerms(BaseCoordinateFrame, FK4NoETermsFrame):
    """
    A coordinate or frame in the FK4 system, but with the E-terms of aberration
    removed.

    The frame attributes are listed under **Other Parameters**.
    """
    pass


# the "self" transform


@frame_transform_graph.transform(DynamicMatrixTransform, FK4NoETerms, FK4NoETerms)
def fk4noe_to_fk4noe(fk4necoord1, fk4neframe2):
    return fk4necoord1._precession_matrix(fk4necoord1.equinox, fk4neframe2.equinox)

@frame_transform_graph.transform(DynamicMatrixTransform, FK4NoETermsFrame, FK4NoETermsFrame)
def fk4noe_to_fk4noe_frame(fk4neframe1, fk4neframe2):
    return FK4NoETermsFrame._precession_matrix(fk4neframe1.equinox, fk4neframe2.equinox)


# FK4-NO-E to/from FK4 ----------------------------->
# Unlike other frames, this module include *two* frame classes for FK4
# coordinates - one including the E-terms of aberration (FK4), and
# one not including them (FK4NoETerms). The following functions
# implement the transformation between these two.
def fk4_e_terms(equinox):
    """
    Return the e-terms of aberration vector.

    Parameters
    ----------
    equinox : Time object
        The equinox for which to compute the e-terms
    """
    # Constant of aberration at J2000; from Explanatory Supplement to the
    # Astronomical Almanac (Seidelmann, 2005).
    k = 0.0056932  # in degrees (v_earth/c ~ 1e-4 rad ~ 0.0057 deg)

    # Explanatory Supplement to the Astronomical Almanac: P. Kenneth
    #  Seidelmann (ed), University Science Books (1992).
    T = (equinox.jd - jd1950) / 36525.0
    # Eccentricity of the Earth's orbit
    ek = math.radians(k) * ((-0.000000126 * T - 0.00004193) * T + 0.01673011)
    # Mean longitude of perigee of the solar orbit
    g = np.radians((((0.012 * T + 1.65) * T + 6190.67) * T + 1015489.951) / 3600.0)
    minus_ek_cos_g = -ek * np.cos(g)
    # Obliquity of the ecliptic
    o = erfa.obl80(equinox.jd, 0)

    return (ek * np.sin(g), minus_ek_cos_g * np.cos(o), minus_ek_cos_g * np.sin(o))


@frame_transform_graph.transform(RepresentationFunctionTransform, FK4Frame, FK4NoETermsFrame)
def fk4_to_fk4_no_e(fk4frame, fk4noeframe):
    needs_precess = fk4frame.equinox != fk4noeframe.equinox

    def converter(rep):
        # Extract cartesian vector
        cart = rep.represent_as(CartesianRepresentation)

        # Find distance (for re-normalization)
        d_orig = cart.norm()
        cart /= d_orig

        # Apply E-terms of aberration. Note that this depends on the equinox (not
        # the observing time/epoch) of the coordinates. See issue #1496 for a
        # discussion of this.
        eterms_a = CartesianRepresentation(
            u.Quantity(fk4_e_terms(fk4frame.equinox), u.one, copy=None),
            copy=False,
        )
        cart = cart - eterms_a + eterms_a.dot(cart) * cart

        # Find new distance (for re-normalization)
        d_new = cart.norm()

        # Renormalize
        cart *= d_orig / d_new

        # now re-cast into an appropriate Representation, and precess if need be
        if isinstance(rep, UnitSphericalRepresentation):
            cart = cart.represent_as(UnitSphericalRepresentation)

        # if no obstime was given in the new frame, use the old one for consistency
        newobstime = (
            fk4frame.obstime if fk4noeframe.obstime is None else fk4noeframe.obstime
        )

        fk4noe = FK4NoETerms(cart, equinox=fk4frame.equinox, obstime=newobstime)
        if needs_precess:
            # precession
            fk4noe = fk4noe.transform_to(fk4noeframe)
        return fk4noe.data

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, FK4, FK4NoETerms)
def fk4_to_fk4_no_e_legacy(fk4coo, fk4noeframe):
    converter = fk4_to_fk4_no_e(fk4coo, fk4noeframe)
    return fk4noeframe.realize_frame(converter(fk4coo.data))


@frame_transform_graph.transform(RepresentationFunctionTransform, FK4NoETermsFrame, FK4Frame)
def fk4_no_e_to_fk4(fk4noeframe, fk4frame):
    needs_precess = fk4noeframe.equinox != fk4frame.equinox
    working_equinox = fk4frame.equinox if needs_precess else fk4noeframe.equinox

    def converter(rep):
        # first precess, if necessary
        if needs_precess:
            fk4noe_w_fk4equinox = FK4NoETerms(
                equinox=fk4frame.equinox, obstime=fk4noeframe.obstime
            )
            fk4noecoord = FK4NoETerms(
                rep, equinox=fk4noeframe.equinox, obstime=fk4noeframe.obstime
            )
            fk4noecoord = fk4noecoord.transform_to(fk4noe_w_fk4equinox)
            rep = fk4noecoord.data

        # Extract cartesian vector
        cart = rep.represent_as(CartesianRepresentation)

        # Find distance (for re-normalization)
        d_orig = cart.norm()
        cart /= d_orig

        # Apply E-terms of aberration. Note that this depends on the equinox (not
        # the observing time/epoch) of the coordinates. See issue #1496 for a
        # discussion of this.
        eterms_a = CartesianRepresentation(
            u.Quantity(fk4_e_terms(working_equinox), u.one, copy=None),
            copy=False,
        )

        eterms_a_plus_rep_ini = eterms_a + cart
        for _ in range(10):
            cart = eterms_a_plus_rep_ini / (1.0 + eterms_a.dot(cart))

        # Find new distance (for re-normalization)
        d_new = cart.norm()

        # Renormalize
        cart *= d_orig / d_new

        # now re-cast into an appropriate Representation
        if isinstance(rep, UnitSphericalRepresentation):
            cart = cart.represent_as(UnitSphericalRepresentation)

        return cart

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, FK4NoETerms, FK4)
def fk4_no_e_to_fk4_legacy(fk4noecoo, fk4frame):
    converter = fk4_no_e_to_fk4(fk4noecoo, fk4frame)
    return fk4frame.realize_frame(converter(fk4noecoo.data))
