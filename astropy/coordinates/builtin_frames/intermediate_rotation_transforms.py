# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Contains the transformation functions for getting to/from ITRS, TEME, GCRS, and CIRS.
These are distinct from the ICRS and AltAz functions because they are just
rotations without aberration corrections or offsets.
"""

import erfa
import numpy as np

from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_transpose
from astropy.coordinates.transformations import (
    FunctionTransformWithFiniteDifference,
    RepresentationFunctionTransform,
)

from .cirs import CIRS, CIRSFrame
from .equatorial import TEME, TETE, TEMEFrame, TETEFrame
from .gcrs import GCRS, GCRSFrame, PrecessedGeocentric, PrecessedGeocentricFrame
from .icrs import ICRS, ICRSFrame
from .itrs import ITRS, ITRSFrame
from .utils import get_jd12, get_polar_motion

# # first define helper functions


def _needs_reroute_for_attr(a, b):
    """Check if two frame attribute arrays differ, treating shape mismatches as different."""
    try:
        return np.any(a != b)
    except ValueError:
        return True


def teme_to_itrs_mat(time):
    # Sidereal time, rotates from ITRS to mean equinox
    # Use 1982 model for consistency with Vallado et al (2006)
    # https://celestrak.org/publications/aiaa/2006-6753/AIAA-2006-6753.pdf
    gst = erfa.gmst82(*get_jd12(time, "ut1"))

    # Polar Motion
    # Do not include TIO locator s' because it is not used in Vallado 2006
    xp, yp = get_polar_motion(time)
    pmmat = erfa.pom00(xp, yp, 0)

    # rotation matrix
    # c2tcio expects a GCRS->CIRS matrix as it's first argument.
    # Here, we just set that to an I-matrix, because we're already
    # in TEME and the difference between TEME and CIRS is just the
    # rotation by the sidereal time rather than the Earth Rotation Angle
    return erfa.c2tcio(np.eye(3), gst, pmmat)


def gcrs_to_cirs_mat(time):
    # celestial-to-intermediate matrix
    return erfa.c2i06a(*get_jd12(time, "tt"))


def cirs_to_itrs_mat(time):
    # compute the polar motion p-matrix
    xp, yp = get_polar_motion(time)
    sp = erfa.sp00(*get_jd12(time, "tt"))
    pmmat = erfa.pom00(xp, yp, sp)

    # now determine the Earth Rotation Angle for the input obstime
    # era00 accepts UT1, so we convert if need be
    era = erfa.era00(*get_jd12(time, "ut1"))

    # c2tcio expects a GCRS->CIRS matrix, but we just set that to an I-matrix
    # because we're already in CIRS
    return erfa.c2tcio(np.eye(3), era, pmmat)


def tete_to_itrs_mat(time, rbpn=None):
    """Compute the polar motion p-matrix at the given time.

    If the nutation-precession matrix is already known, it should be passed in,
    as this is by far the most expensive calculation.
    """
    xp, yp = get_polar_motion(time)
    sp = erfa.sp00(*get_jd12(time, "tt"))
    pmmat = erfa.pom00(xp, yp, sp)

    # now determine the greenwich apparent sidereal time for the input obstime
    # we use the 2006A model for consistency with RBPN matrix use in GCRS <-> TETE
    ujd1, ujd2 = get_jd12(time, "ut1")
    jd1, jd2 = get_jd12(time, "tt")
    if rbpn is None:
        # erfa.gst06a calls pnm06a to calculate rbpn and then gst06. Use it in
        # favour of getting rbpn with erfa.pnm06a to avoid a possibly large array.
        gast = erfa.gst06a(ujd1, ujd2, jd1, jd2)
    else:
        gast = erfa.gst06(ujd1, ujd2, jd1, jd2, rbpn)

    # c2tcio expects a GCRS->CIRS matrix, but we just set that to an I-matrix
    # because we're already in CIRS equivalent frame
    return erfa.c2tcio(np.eye(3), gast, pmmat)


def gcrs_precession_mat(equinox):
    gamb, phib, psib, epsa = erfa.pfw06(*get_jd12(equinox, "tt"))
    return erfa.fw2m(gamb, phib, psib, epsa)


def get_location_gcrs(location, obstime, ref_to_itrs, gcrs_to_ref):
    """Create a GCRS frame at the location and obstime.

    The reference frame z axis must point to the Celestial Intermediate Pole
    (as is the case for CIRS and TETE).

    This function is here to avoid location.get_gcrs(obstime), which would
    recalculate matrices that are already available below (and return a GCRS
    coordinate, rather than a frame with obsgeoloc and obsgeovel).  Instead,
    it uses the private method that allows passing in the matrices.

    """
    obsgeoloc, obsgeovel = location._get_gcrs_posvel(obstime, ref_to_itrs, gcrs_to_ref)
    return GCRS(obstime=obstime, obsgeoloc=obsgeoloc, obsgeovel=obsgeovel)


# now the actual transforms


@frame_transform_graph.transform(RepresentationFunctionTransform, GCRSFrame, TETEFrame)
def gcrs_to_tete(gcrs_frame, tete_frame):
    # Classical NPB matrix, IAU 2006/2000A
    # (same as in builtin_frames.utils.get_cip).
    rbpn = erfa.pnm06a(*get_jd12(tete_frame.obstime, "tt"))
    # Get GCRS coordinates for the target observer location and time.
    loc_gcrs = get_location_gcrs(
        tete_frame.location,
        tete_frame.obstime,
        tete_to_itrs_mat(tete_frame.obstime, rbpn=rbpn),
        rbpn,
    )
    needs_reroute = (
        _needs_reroute_for_attr(gcrs_frame.obstime, loc_gcrs.obstime)
        or _needs_reroute_for_attr(gcrs_frame.obsgeoloc.xyz.value, loc_gcrs.obsgeoloc.xyz.value)
        or _needs_reroute_for_attr(gcrs_frame.obsgeovel.xyz.value, loc_gcrs.obsgeovel.xyz.value)
    )

    def converter(rep):
        if needs_reroute:
            rep = GCRS(
                rep,
                obstime=gcrs_frame.obstime,
                obsgeoloc=gcrs_frame.obsgeoloc,
                obsgeovel=gcrs_frame.obsgeovel,
            ).transform_to(loc_gcrs).data
        # Now we are relative to the correct observer, do the transform to TETE.
        # These rotations are defined at the geocenter, but can be applied to
        # topocentric positions as well, assuming rigid Earth. See p57 of
        # https://www.usno.navy.mil/USNO/astronomical-applications/publications/Circular_179.pdf
        return rep.to_cartesian().transform(rbpn)

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, GCRS, TETE)
def gcrs_to_tete_legacy(gcrs_coo, tete_frame):
    converter = gcrs_to_tete(gcrs_coo, tete_frame)
    return tete_frame.realize_frame(converter(gcrs_coo.data))


@frame_transform_graph.transform(RepresentationFunctionTransform, TETEFrame, GCRSFrame)
def tete_to_gcrs(tete_frame, gcrs_frame):
    # Compute the pn matrix, and then multiply by its transpose.
    rbpn = erfa.pnm06a(*get_jd12(tete_frame.obstime, "tt"))
    # We will need the GCRS frame for the input location and obstime.
    loc_gcrs = get_location_gcrs(
        tete_frame.location,
        tete_frame.obstime,
        tete_to_itrs_mat(tete_frame.obstime, rbpn=rbpn),
        rbpn,
    )
    needs_reroute = (
        _needs_reroute_for_attr(loc_gcrs.obstime, gcrs_frame.obstime)
        or _needs_reroute_for_attr(loc_gcrs.obsgeoloc.xyz.value, gcrs_frame.obsgeoloc.xyz.value)
        or _needs_reroute_for_attr(loc_gcrs.obsgeovel.xyz.value, gcrs_frame.obsgeovel.xyz.value)
    )

    def converter(rep):
        newrepr = rep.to_cartesian().transform(matrix_transpose(rbpn))
        # We now have a GCRS vector for the input location and obstime.
        if needs_reroute:
            # Move to the target GCRS (no-op if same obstime and location)
            gcrs = loc_gcrs.realize_frame(newrepr)
            return gcrs.transform_to(gcrs_frame).data
        return newrepr

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, TETE, GCRS)
def tete_to_gcrs_legacy(tete_coo, gcrs_frame):
    converter = tete_to_gcrs(tete_coo, gcrs_frame)
    return gcrs_frame.realize_frame(converter(tete_coo.data))


@frame_transform_graph.transform(RepresentationFunctionTransform, TETEFrame, ITRSFrame)
def tete_to_itrs(tete_frame, itrs_frame):
    # now get the pmatrix
    pmat = tete_to_itrs_mat(itrs_frame.obstime)
    needs_reroute = np.any(tete_frame.obstime != itrs_frame.obstime) or np.any(
        tete_frame.location != itrs_frame.location
    )

    def converter(rep):
        if needs_reroute:
            # first get us to TETE at the target obstime, and location (no-op if same)
            rep = TETE(
                rep, obstime=tete_frame.obstime, location=tete_frame.location
            ).transform_to(TETE(obstime=itrs_frame.obstime, location=itrs_frame.location)).data
        return rep.to_cartesian().transform(pmat)

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, TETE, ITRS)
def tete_to_itrs_legacy(tete_coo, itrs_frame):
    converter = tete_to_itrs(tete_coo, itrs_frame)
    return itrs_frame.realize_frame(converter(tete_coo.data))


@frame_transform_graph.transform(RepresentationFunctionTransform, ITRSFrame, TETEFrame)
def itrs_to_tete(itrs_frame, tete_frame):
    # compute the pmatrix, and then multiply by its transpose
    pmat = tete_to_itrs_mat(itrs_frame.obstime)
    needs_reroute = np.any(itrs_frame.obstime != tete_frame.obstime) or np.any(
        itrs_frame.location != tete_frame.location
    )

    def converter(rep):
        newrepr = rep.to_cartesian().transform(matrix_transpose(pmat))
        if needs_reroute:
            tete = TETE(newrepr, obstime=itrs_frame.obstime, location=itrs_frame.location)
            # now do any needed offsets (no-op if same obstime and location)
            return tete.transform_to(tete_frame).data
        return newrepr

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ITRS, TETE)
def itrs_to_tete_legacy(itrs_coo, tete_frame):
    converter = itrs_to_tete(itrs_coo, tete_frame)
    return tete_frame.realize_frame(converter(itrs_coo.data))


@frame_transform_graph.transform(RepresentationFunctionTransform, GCRSFrame, CIRSFrame)
def gcrs_to_cirs(gcrs_frame, cirs_frame):
    # first get the pmatrix
    pmat = gcrs_to_cirs_mat(cirs_frame.obstime)
    # Get GCRS coordinates for the target observer location and time.
    loc_gcrs = get_location_gcrs(
        cirs_frame.location,
        cirs_frame.obstime,
        cirs_to_itrs_mat(cirs_frame.obstime),
        pmat,
    )
    needs_reroute = (
        _needs_reroute_for_attr(gcrs_frame.obstime, loc_gcrs.obstime)
        or _needs_reroute_for_attr(gcrs_frame.obsgeoloc.xyz.value, loc_gcrs.obsgeoloc.xyz.value)
        or _needs_reroute_for_attr(gcrs_frame.obsgeovel.xyz.value, loc_gcrs.obsgeovel.xyz.value)
    )

    def converter(rep):
        if needs_reroute:
            rep = GCRS(
                rep,
                obstime=gcrs_frame.obstime,
                obsgeoloc=gcrs_frame.obsgeoloc,
                obsgeovel=gcrs_frame.obsgeovel,
            ).transform_to(loc_gcrs).data
        # Now we are relative to the correct observer, do the transform to CIRS.
        return rep.to_cartesian().transform(pmat)

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, GCRS, CIRS)
def gcrs_to_cirs_legacy(gcrs_coo, cirs_frame):
    converter = gcrs_to_cirs(gcrs_coo, cirs_frame)
    return cirs_frame.realize_frame(converter(gcrs_coo.data))


@frame_transform_graph.transform(RepresentationFunctionTransform, CIRSFrame, GCRSFrame)
def cirs_to_gcrs(cirs_frame, gcrs_frame):
    # Compute the pmatrix, and then multiply by its transpose,
    pmat = gcrs_to_cirs_mat(cirs_frame.obstime)
    # We will need the GCRS frame for the input location and obstime.
    loc_gcrs = get_location_gcrs(
        cirs_frame.location, cirs_frame.obstime, cirs_to_itrs_mat(cirs_frame.obstime), pmat
    )
    needs_reroute = (
        _needs_reroute_for_attr(loc_gcrs.obstime, gcrs_frame.obstime)
        or _needs_reroute_for_attr(loc_gcrs.obsgeoloc.xyz.value, gcrs_frame.obsgeoloc.xyz.value)
        or _needs_reroute_for_attr(loc_gcrs.obsgeovel.xyz.value, gcrs_frame.obsgeovel.xyz.value)
    )

    def converter(rep):
        newrepr = rep.to_cartesian().transform(matrix_transpose(pmat))
        # We now have a GCRS vector for the input location and obstime.
        if needs_reroute:
            # Move to the target GCRS (no-op if same obstime and location)
            gcrs = loc_gcrs.realize_frame(newrepr)
            return gcrs.transform_to(gcrs_frame).data
        return newrepr

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, CIRS, GCRS)
def cirs_to_gcrs_legacy(cirs_coo, gcrs_frame):
    converter = cirs_to_gcrs(cirs_coo, gcrs_frame)
    return gcrs_frame.realize_frame(converter(cirs_coo.data))


@frame_transform_graph.transform(RepresentationFunctionTransform, CIRSFrame, ITRSFrame)
def cirs_to_itrs(cirs_frame, itrs_frame):
    # now get the pmatrix
    pmat = cirs_to_itrs_mat(itrs_frame.obstime)
    needs_reroute = np.any(cirs_frame.obstime != itrs_frame.obstime) or np.any(
        cirs_frame.location != itrs_frame.location
    )

    def converter(rep):
        if needs_reroute:
            # first get us to CIRS at the target obstime, and location (no-op if same)
            rep = CIRS(
                rep, obstime=cirs_frame.obstime, location=cirs_frame.location
            ).transform_to(CIRS(obstime=itrs_frame.obstime, location=itrs_frame.location)).data
        return rep.to_cartesian().transform(pmat)

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, CIRS, ITRS)
def cirs_to_itrs_legacy(cirs_coo, itrs_frame):
    converter = cirs_to_itrs(cirs_coo, itrs_frame)
    return itrs_frame.realize_frame(converter(cirs_coo.data))


@frame_transform_graph.transform(RepresentationFunctionTransform, ITRSFrame, CIRSFrame)
def itrs_to_cirs(itrs_frame, cirs_frame):
    # compute the pmatrix, and then multiply by its transpose
    pmat = cirs_to_itrs_mat(itrs_frame.obstime)
    needs_reroute = np.any(itrs_frame.obstime != cirs_frame.obstime) or np.any(
        itrs_frame.location != cirs_frame.location
    )

    def converter(rep):
        newrepr = rep.to_cartesian().transform(matrix_transpose(pmat))
        if needs_reroute:
            cirs = CIRS(newrepr, obstime=itrs_frame.obstime, location=itrs_frame.location)
            # now do any needed offsets (no-op if same obstime and location)
            return cirs.transform_to(cirs_frame).data
        return newrepr

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ITRS, CIRS)
def itrs_to_cirs_legacy(itrs_coo, cirs_frame):
    converter = itrs_to_cirs(itrs_coo, cirs_frame)
    return cirs_frame.realize_frame(converter(itrs_coo.data))


# TODO: implement GCRS<->CIRS if there's call for it.  The thing that's awkward
# is that they both have obstimes, so an extra set of transformations are necessary.
# so unless there's a specific need for that, better to just have it go through the above
# two steps anyway


@frame_transform_graph.transform(RepresentationFunctionTransform, GCRSFrame, PrecessedGeocentricFrame)
def gcrs_to_precessedgeo(gcrs_frame, precessedgeo_frame):
    # now precess to the requested equinox
    pmat = gcrs_precession_mat(precessedgeo_frame.equinox)
    needs_reroute = (
        _needs_reroute_for_attr(gcrs_frame.obstime, precessedgeo_frame.obstime)
        or _needs_reroute_for_attr(gcrs_frame.obsgeoloc.xyz.value, precessedgeo_frame.obsgeoloc.xyz.value)
        or _needs_reroute_for_attr(gcrs_frame.obsgeovel.xyz.value, precessedgeo_frame.obsgeovel.xyz.value)
    )

    def converter(rep):
        if needs_reroute:
            # first get us to GCRS with the right attributes
            rep = GCRS(
                rep,
                obstime=gcrs_frame.obstime,
                obsgeoloc=gcrs_frame.obsgeoloc,
                obsgeovel=gcrs_frame.obsgeovel,
            ).transform_to(
                GCRS(
                    obstime=precessedgeo_frame.obstime,
                    obsgeoloc=precessedgeo_frame.obsgeoloc,
                    obsgeovel=precessedgeo_frame.obsgeovel,
                )
            ).data
        return rep.to_cartesian().transform(pmat)

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, GCRS, PrecessedGeocentric)
def gcrs_to_precessedgeo_legacy(gcrs_coo, precessedgeo_frame):
    converter = gcrs_to_precessedgeo(gcrs_coo, precessedgeo_frame)
    return precessedgeo_frame.realize_frame(converter(gcrs_coo.data))


@frame_transform_graph.transform(RepresentationFunctionTransform, PrecessedGeocentricFrame, GCRSFrame)
def precessedgeo_to_gcrs(precessedgeo_frame, gcrs_frame):
    # first un-precess
    pmat = gcrs_precession_mat(precessedgeo_frame.equinox)
    needs_reroute = (
        _needs_reroute_for_attr(precessedgeo_frame.obstime, gcrs_frame.obstime)
        or _needs_reroute_for_attr(precessedgeo_frame.obsgeoloc.xyz.value, gcrs_frame.obsgeoloc.xyz.value)
        or _needs_reroute_for_attr(precessedgeo_frame.obsgeovel.xyz.value, gcrs_frame.obsgeovel.xyz.value)
    )

    def converter(rep):
        crepr = rep.to_cartesian().transform(matrix_transpose(pmat))
        if needs_reroute:
            gcrs_coo = GCRS(
                crepr,
                obstime=precessedgeo_frame.obstime,
                obsgeoloc=precessedgeo_frame.obsgeoloc,
                obsgeovel=precessedgeo_frame.obsgeovel,
            )
            # then move to the GCRS that's actually desired
            return gcrs_coo.transform_to(gcrs_frame).data
        return crepr

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, PrecessedGeocentric, GCRS)
def precessedgeo_to_gcrs_legacy(precessedgeo_coo, gcrs_frame):
    converter = precessedgeo_to_gcrs(precessedgeo_coo, gcrs_frame)
    return gcrs_frame.realize_frame(converter(precessedgeo_coo.data))


@frame_transform_graph.transform(RepresentationFunctionTransform, TEMEFrame, ITRSFrame)
def teme_to_itrs(teme_frame, itrs_frame):
    # use the pmatrix to transform to ITRS in the source obstime
    pmat = teme_to_itrs_mat(teme_frame.obstime)
    needs_reroute = np.any(teme_frame.obstime != itrs_frame.obstime)

    def converter(rep):
        crepr = rep.to_cartesian().transform(pmat)
        if needs_reroute:
            itrs = ITRS(crepr, obstime=teme_frame.obstime)
            # transform the ITRS coordinate to the target obstime
            return itrs.transform_to(itrs_frame).data
        return crepr

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, TEME, ITRS)
def teme_to_itrs_legacy(teme_coo, itrs_frame):
    converter = teme_to_itrs(teme_coo, itrs_frame)
    return itrs_frame.realize_frame(converter(teme_coo.data))


@frame_transform_graph.transform(RepresentationFunctionTransform, ITRSFrame, TEMEFrame)
def itrs_to_teme(itrs_frame, teme_frame):
    pmat = teme_to_itrs_mat(teme_frame.obstime)
    needs_reroute = np.any(itrs_frame.obstime != teme_frame.obstime)

    def converter(rep):
        if needs_reroute:
            # transform the ITRS coordinate to the target obstime
            rep = ITRS(rep, obstime=itrs_frame.obstime).transform_to(
                ITRS(obstime=teme_frame.obstime)
            ).data
        # compute the pmatrix, and then multiply by its transpose
        return rep.to_cartesian().transform(matrix_transpose(pmat))

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ITRS, TEME)
def itrs_to_teme_legacy(itrs_coo, teme_frame):
    converter = itrs_to_teme(itrs_coo, teme_frame)
    return teme_frame.realize_frame(converter(itrs_coo.data))


# Create loopback transformations
frame_transform_graph._add_merged_transform(ITRS, CIRS, ITRS)
frame_transform_graph._add_merged_transform(
    PrecessedGeocentric, GCRS, PrecessedGeocentric
)
frame_transform_graph._add_merged_transform(TEME, ITRS, TEME)
frame_transform_graph._add_merged_transform(TETE, ICRS, TETE)
frame_transform_graph._add_merged_transform(ITRSFrame, CIRSFrame, ITRSFrame)
frame_transform_graph._add_merged_transform(
    PrecessedGeocentricFrame, GCRSFrame, PrecessedGeocentricFrame
)
frame_transform_graph._add_merged_transform(TEMEFrame, ITRSFrame, TEMEFrame)
frame_transform_graph._add_merged_transform(TETEFrame, ICRSFrame, TETEFrame)
