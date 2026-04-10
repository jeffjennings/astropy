# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Contains the transformation functions for getting to/from ecliptic systems.
"""

import erfa
import numpy as np

from astropy import units as u
from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates.coordinate import Coordinate
from astropy.coordinates.errors import UnitsError
from astropy.coordinates.matrix_utilities import matrix_transpose, rotation_matrix
from astropy.coordinates.transformations import (
    AffineTransform,
    DynamicMatrixTransform,
    FunctionTransformWithFiniteDifference,
    RepresentationFunctionTransformWithFiniteDifference,
)

from .ecliptic import (
    BarycentricMeanEcliptic,
    BarycentricTrueEcliptic,
    CustomBarycentricEcliptic,
    GeocentricMeanEcliptic,
    GeocentricMeanEclipticFrame,
    GeocentricTrueEcliptic,
    GeocentricTrueEclipticFrame,
    HeliocentricEclipticIAU76,
    HeliocentricMeanEcliptic,
    HeliocentricTrueEcliptic,
)
from .gcrs import GCRS, GCRSFrame
from .icrs import ICRS
from .utils import EQUINOX_J2000, get_jd12, get_offset_sun_from_barycenter


def _mean_ecliptic_rotation_matrix(equinox):
    # This code just calls ecm06, which uses the precession matrix according to the
    # IAU 2006 model, but leaves out nutation. This brings the results closer to what
    # other libraries give (see https://github.com/astropy/astropy/pull/6508).
    return erfa.ecm06(*get_jd12(equinox, "tt"))


def _true_ecliptic_rotation_matrix(equinox):
    # This code calls the same routines as done in pnm06a from ERFA, which
    # retrieves the precession matrix (including frame bias) according to
    # the IAU 2006 model, and including the nutation.
    # This family of systems is less popular
    # (see https://github.com/astropy/astropy/pull/6508).
    jd1, jd2 = get_jd12(equinox, "tt")
    # Here, we call the three routines from erfa.pnm06a separately,
    # so that we can keep the nutation for calculating the true obliquity
    # (which is a fairly expensive operation); see gh-11000.
    # pnm06a: Fukushima-Williams angles for frame bias and precession.
    # (ERFA names short for F-W's gamma_bar, phi_bar, psi_bar and epsilon_A).
    gamb, phib, psib, epsa = erfa.pfw06(jd1, jd2)
    # pnm06a: Nutation components (in longitude and obliquity).
    dpsi, deps = erfa.nut06a(jd1, jd2)
    # pnm06a: Equinox based nutation x precession x bias matrix.
    rnpb = erfa.fw2m(gamb, phib, psib + dpsi, epsa + deps)
    # calculate the true obliquity of the ecliptic
    obl = erfa.obl06(jd1, jd2) + deps
    return rotation_matrix(obl << u.radian, "x") @ rnpb


def _obliquity_only_rotation_matrix(
    obl=erfa.obl80(EQUINOX_J2000.jd1, EQUINOX_J2000.jd2) * u.radian,
):
    # This code only accounts for the obliquity,
    # which can be passed explicitly.
    # The default value is the IAU 1980 value for J2000,
    # which is computed using obl80 from ERFA:
    #
    # obl = erfa.obl80(EQUINOX_J2000.jd1, EQUINOX_J2000.jd2) * u.radian
    return rotation_matrix(obl, "x")


# MeanEcliptic frames


@frame_transform_graph.transform(
    RepresentationFunctionTransformWithFiniteDifference,
    GCRSFrame,
    GeocentricMeanEclipticFrame,
    finite_difference_frameattr_name="equinox",
)
def gcrs_to_geoecliptic(gcrs_frame, to_frame):
    rmat = _mean_ecliptic_rotation_matrix(to_frame.equinox)
    # Target is geocentric GCRS (zero obsgeoloc/obsgeovel) at to_frame.obstime
    needs_reroute = (
        np.any(gcrs_frame.obstime != to_frame.obstime)
        or np.any(gcrs_frame.obsgeoloc.xyz.value != 0)
        or np.any(gcrs_frame.obsgeovel.xyz.value != 0)
    )

    def converter(rep):
        if needs_reroute:
            # first get us to a 0 pos/vel GCRS at the target obstime
            rep = (
                Coordinate(
                    GCRSFrame(
                        obstime=gcrs_frame.obstime,
                        obsgeoloc=gcrs_frame.obsgeoloc,
                        obsgeovel=gcrs_frame.obsgeovel,
                    ),
                    rep,
                )
                .transform_to(GCRSFrame(obstime=to_frame.obstime))
                .data
            )
        return rep.to_cartesian().transform(rmat)

    return converter


# TODO: APE23: remove when legacy frames are deprecated
@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    GCRS,
    GeocentricMeanEcliptic,
    finite_difference_frameattr_name="equinox",
)
def gcrs_to_geoecliptic_legacy(gcrs_coo, to_frame):
    converter = gcrs_to_geoecliptic(gcrs_coo, to_frame)
    return to_frame.realize_frame(converter(gcrs_coo.data))


@frame_transform_graph.transform(
    RepresentationFunctionTransformWithFiniteDifference,
    GeocentricMeanEclipticFrame,
    GCRSFrame,
)
def geoecliptic_to_gcrs(from_frame, gcrs_frame):
    rmat = _mean_ecliptic_rotation_matrix(from_frame.equinox)
    # Intermediate is geocentric GCRS (zero obsgeoloc/obsgeovel) at from_frame.obstime
    needs_reroute = (
        np.any(from_frame.obstime != gcrs_frame.obstime)
        or np.any(gcrs_frame.obsgeoloc.xyz.value != 0)
        or np.any(gcrs_frame.obsgeovel.xyz.value != 0)
    )

    def converter(rep):
        newrepr = rep.to_cartesian().transform(matrix_transpose(rmat))
        if needs_reroute:
            # now do any needed offsets (no-op if same obstime and 0 pos/vel)
            return (
                Coordinate(GCRSFrame(obstime=from_frame.obstime), newrepr)
                .transform_to(gcrs_frame)
                .data
            )
        return newrepr

    return converter


# TODO: APE23: remove when legacy frames are deprecated
@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference, GeocentricMeanEcliptic, GCRS
)
def geoecliptic_to_gcrs_legacy(from_coo, gcrs_frame):
    converter = geoecliptic_to_gcrs(from_coo, gcrs_frame)
    return gcrs_frame.realize_frame(converter(from_coo.data))


@frame_transform_graph.transform(DynamicMatrixTransform, ICRS, BarycentricMeanEcliptic)
def icrs_to_baryecliptic(from_coo, to_frame):
    return _mean_ecliptic_rotation_matrix(to_frame.equinox)


@frame_transform_graph.transform(DynamicMatrixTransform, BarycentricMeanEcliptic, ICRS)
def baryecliptic_to_icrs(from_coo, to_frame):
    return matrix_transpose(icrs_to_baryecliptic(to_frame, from_coo))


_NEED_ORIGIN_HINT = (
    "The input {0} coordinates do not have length units. This probably means you"
    " created coordinates with lat/lon but no distance.  Heliocentric<->ICRS transforms"
    " cannot function in this case because there is an origin shift."
)


@frame_transform_graph.transform(AffineTransform, ICRS, HeliocentricMeanEcliptic)
def icrs_to_helioecliptic(from_coo, to_frame):
    if not u.m.is_equivalent(from_coo.cartesian.x.unit):
        raise UnitsError(_NEED_ORIGIN_HINT.format(from_coo.__class__.__name__))

    # get the offset of the barycenter from the Sun
    ssb_from_sun = get_offset_sun_from_barycenter(
        to_frame.obstime,
        reverse=True,
        include_velocity=bool(from_coo.data.differentials),
    )

    # now compute the matrix to precess to the right orientation
    rmat = _mean_ecliptic_rotation_matrix(to_frame.equinox)

    return rmat, ssb_from_sun.transform(rmat)


@frame_transform_graph.transform(AffineTransform, HeliocentricMeanEcliptic, ICRS)
def helioecliptic_to_icrs(from_coo, to_frame):
    if not u.m.is_equivalent(from_coo.cartesian.x.unit):
        raise UnitsError(_NEED_ORIGIN_HINT.format(from_coo.__class__.__name__))

    # first un-precess from ecliptic to ICRS orientation
    rmat = _mean_ecliptic_rotation_matrix(from_coo.equinox)

    # now offset back to barycentric, which is the correct center for ICRS
    sun_from_ssb = get_offset_sun_from_barycenter(
        from_coo.obstime, include_velocity=bool(from_coo.data.differentials)
    )

    return matrix_transpose(rmat), sun_from_ssb


# TrueEcliptic frames


@frame_transform_graph.transform(
    RepresentationFunctionTransformWithFiniteDifference,
    GCRSFrame,
    GeocentricTrueEclipticFrame,
    finite_difference_frameattr_name="equinox",
)
def gcrs_to_true_geoecliptic(gcrs_frame, to_frame):
    rmat = _true_ecliptic_rotation_matrix(to_frame.equinox)
    # Target is geocentric GCRS (zero obsgeoloc/obsgeovel) at to_frame.obstime
    needs_reroute = (
        np.any(gcrs_frame.obstime != to_frame.obstime)
        or np.any(gcrs_frame.obsgeoloc.xyz.value != 0)
        or np.any(gcrs_frame.obsgeovel.xyz.value != 0)
    )

    def converter(rep):
        if needs_reroute:
            # first get us to a 0 pos/vel GCRS at the target obstime
            rep = (
                Coordinate(
                    GCRSFrame(
                        obstime=gcrs_frame.obstime,
                        obsgeoloc=gcrs_frame.obsgeoloc,
                        obsgeovel=gcrs_frame.obsgeovel,
                    ),
                    rep,
                )
                .transform_to(GCRSFrame(obstime=to_frame.obstime))
                .data
            )
        return rep.to_cartesian().transform(rmat)

    return converter


# TODO: APE23: remove when legacy frames are deprecated
@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    GCRS,
    GeocentricTrueEcliptic,
    finite_difference_frameattr_name="equinox",
)
def gcrs_to_true_geoecliptic_legacy(gcrs_coo, to_frame):
    converter = gcrs_to_true_geoecliptic(gcrs_coo, to_frame)
    return to_frame.realize_frame(converter(gcrs_coo.data))


@frame_transform_graph.transform(
    RepresentationFunctionTransformWithFiniteDifference,
    GeocentricTrueEclipticFrame,
    GCRSFrame,
)
def true_geoecliptic_to_gcrs(from_frame, gcrs_frame):
    rmat = _true_ecliptic_rotation_matrix(from_frame.equinox)
    # Intermediate is geocentric GCRS (zero obsgeoloc/obsgeovel) at from_frame.obstime
    needs_reroute = (
        np.any(from_frame.obstime != gcrs_frame.obstime)
        or np.any(gcrs_frame.obsgeoloc.xyz.value != 0)
        or np.any(gcrs_frame.obsgeovel.xyz.value != 0)
    )

    def converter(rep):
        newrepr = rep.to_cartesian().transform(matrix_transpose(rmat))
        if needs_reroute:
            # now do any needed offsets (no-op if same obstime and 0 pos/vel)
            return (
                Coordinate(GCRSFrame(obstime=from_frame.obstime), newrepr)
                .transform_to(gcrs_frame)
                .data
            )
        return newrepr

    return converter


# TODO: APE23: remove when legacy frames are deprecated
@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference, GeocentricTrueEcliptic, GCRS
)
def true_geoecliptic_to_gcrs_legacy(from_coo, gcrs_frame):
    converter = true_geoecliptic_to_gcrs(from_coo, gcrs_frame)
    return gcrs_frame.realize_frame(converter(from_coo.data))


@frame_transform_graph.transform(DynamicMatrixTransform, ICRS, BarycentricTrueEcliptic)
def icrs_to_true_baryecliptic(from_coo, to_frame):
    return _true_ecliptic_rotation_matrix(to_frame.equinox)


@frame_transform_graph.transform(DynamicMatrixTransform, BarycentricTrueEcliptic, ICRS)
def true_baryecliptic_to_icrs(from_coo, to_frame):
    return matrix_transpose(icrs_to_true_baryecliptic(to_frame, from_coo))


@frame_transform_graph.transform(AffineTransform, ICRS, HeliocentricTrueEcliptic)
def icrs_to_true_helioecliptic(from_coo, to_frame):
    if not u.m.is_equivalent(from_coo.cartesian.x.unit):
        raise UnitsError(_NEED_ORIGIN_HINT.format(from_coo.__class__.__name__))

    # get the offset of the barycenter from the Sun
    ssb_from_sun = get_offset_sun_from_barycenter(
        to_frame.obstime,
        reverse=True,
        include_velocity=bool(from_coo.data.differentials),
    )

    # now compute the matrix to precess to the right orientation
    rmat = _true_ecliptic_rotation_matrix(to_frame.equinox)

    return rmat, ssb_from_sun.transform(rmat)


@frame_transform_graph.transform(AffineTransform, HeliocentricTrueEcliptic, ICRS)
def true_helioecliptic_to_icrs(from_coo, to_frame):
    if not u.m.is_equivalent(from_coo.cartesian.x.unit):
        raise UnitsError(_NEED_ORIGIN_HINT.format(from_coo.__class__.__name__))

    # first un-precess from ecliptic to ICRS orientation
    rmat = _true_ecliptic_rotation_matrix(from_coo.equinox)

    # now offset back to barycentric, which is the correct center for ICRS
    sun_from_ssb = get_offset_sun_from_barycenter(
        from_coo.obstime, include_velocity=bool(from_coo.data.differentials)
    )

    return matrix_transpose(rmat), sun_from_ssb


# Other ecliptic frames


@frame_transform_graph.transform(AffineTransform, HeliocentricEclipticIAU76, ICRS)
def ecliptic_to_iau76_icrs(from_coo, to_frame):
    # first un-precess from ecliptic to ICRS orientation
    rmat = _obliquity_only_rotation_matrix()

    # now offset back to barycentric, which is the correct center for ICRS
    sun_from_ssb = get_offset_sun_from_barycenter(
        from_coo.obstime, include_velocity=bool(from_coo.data.differentials)
    )

    return matrix_transpose(rmat), sun_from_ssb


@frame_transform_graph.transform(AffineTransform, ICRS, HeliocentricEclipticIAU76)
def icrs_to_iau76_ecliptic(from_coo, to_frame):
    # get the offset of the barycenter from the Sun
    ssb_from_sun = get_offset_sun_from_barycenter(
        to_frame.obstime,
        reverse=True,
        include_velocity=bool(from_coo.data.differentials),
    )

    # now compute the matrix to precess to the right orientation
    rmat = _obliquity_only_rotation_matrix()

    return rmat, ssb_from_sun.transform(rmat)


@frame_transform_graph.transform(
    DynamicMatrixTransform, ICRS, CustomBarycentricEcliptic
)
def icrs_to_custombaryecliptic(from_coo, to_frame):
    return _obliquity_only_rotation_matrix(to_frame.obliquity)


@frame_transform_graph.transform(
    DynamicMatrixTransform, CustomBarycentricEcliptic, ICRS
)
def custombaryecliptic_to_icrs(from_coo, to_frame):
    return icrs_to_custombaryecliptic(to_frame, from_coo).T


# Create loopback transformations
frame_transform_graph._add_merged_transform(
    GeocentricMeanEcliptic, ICRS, GeocentricMeanEcliptic
)
frame_transform_graph._add_merged_transform(
    GeocentricTrueEcliptic, ICRS, GeocentricTrueEcliptic
)
frame_transform_graph._add_merged_transform(
    HeliocentricMeanEcliptic, ICRS, HeliocentricMeanEcliptic
)
frame_transform_graph._add_merged_transform(
    HeliocentricTrueEcliptic, ICRS, HeliocentricTrueEcliptic
)
frame_transform_graph._add_merged_transform(
    HeliocentricEclipticIAU76, ICRS, HeliocentricEclipticIAU76
)
frame_transform_graph._add_merged_transform(
    BarycentricMeanEcliptic, ICRS, BarycentricMeanEcliptic
)
frame_transform_graph._add_merged_transform(
    BarycentricTrueEcliptic, ICRS, BarycentricTrueEcliptic
)
frame_transform_graph._add_merged_transform(
    CustomBarycentricEcliptic, ICRS, CustomBarycentricEcliptic
)
frame_transform_graph._add_merged_transform(
    GeocentricMeanEclipticFrame, GCRSFrame, GeocentricMeanEclipticFrame
)
frame_transform_graph._add_merged_transform(
    GeocentricTrueEclipticFrame, GCRSFrame, GeocentricTrueEclipticFrame
)
