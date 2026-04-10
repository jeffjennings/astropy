# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Contains the transformation functions for getting to "observed" systems from ICRS.
"""

import erfa

from astropy import units as u
from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates.builtin_frames.utils import atciqz, aticq
from astropy.coordinates.erfa_astrom import erfa_astrom
from astropy.coordinates.representation import (
    CartesianRepresentation,
    SphericalRepresentation,
    UnitSphericalRepresentation,
)
from astropy.coordinates.transformations import (
    FunctionTransformWithFiniteDifference,
    RepresentationFunctionTransformWithFiniteDifference,
)

from .altaz import AltAz, AltAzFrame
from .hadec import HADec, HADecFrame
from .icrs import ICRS, ICRSFrame
from .utils import PIOVER2


@frame_transform_graph.transform(
    RepresentationFunctionTransformWithFiniteDifference, ICRSFrame, AltAzFrame
)
@frame_transform_graph.transform(
    RepresentationFunctionTransformWithFiniteDifference, ICRSFrame, HADecFrame
)
def icrs_to_observed(icrs_frame, observed_frame):
    # first set up the astrometry context for ICRS<->observed
    astrom = erfa_astrom.get().apco(observed_frame)
    is_altaz = isinstance(observed_frame, AltAzFrame)

    def converter(rep):
        # if the data are UnitSphericalRepresentation, we can skip the distance calculations
        is_unitspherical = isinstance(rep, UnitSphericalRepresentation) or (
            rep.represent_as(CartesianRepresentation).x.unit == u.one
        )
        if is_unitspherical:
            srepr = rep.represent_as(SphericalRepresentation)
        else:
            # correct for parallax to find BCRS direction from observer (as in erfa.pmpx)
            observer_icrs = CartesianRepresentation(
                astrom["eb"], unit=u.au, xyz_axis=-1, copy=None
            )
            srepr = (
                rep.represent_as(CartesianRepresentation) - observer_icrs
            ).represent_as(SphericalRepresentation)

        # convert to topocentric CIRS
        cirs_ra, cirs_dec = atciqz(srepr, astrom)

        # now perform observed conversion
        if is_altaz:
            lon, zen, _, _, _ = erfa.atioq(cirs_ra, cirs_dec, astrom)
            lat = PIOVER2 - zen
        else:
            _, _, lon, lat, _ = erfa.atioq(cirs_ra, cirs_dec, astrom)

        if is_unitspherical:
            return UnitSphericalRepresentation(
                lon << u.radian, lat << u.radian, copy=False
            )
        else:
            return SphericalRepresentation(
                lon << u.radian, lat << u.radian, srepr.distance, copy=False
            )

    return converter


# TODO: APE23: remove when legacy frames are deprecated
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ICRS, AltAz)
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ICRS, HADec)
def icrs_to_observed_legacy(icrs_coo, observed_frame):
    converter = icrs_to_observed(icrs_coo, observed_frame)
    return observed_frame.realize_frame(converter(icrs_coo.data))


@frame_transform_graph.transform(
    RepresentationFunctionTransformWithFiniteDifference, AltAzFrame, ICRSFrame
)
@frame_transform_graph.transform(
    RepresentationFunctionTransformWithFiniteDifference, HADecFrame, ICRSFrame
)
def observed_to_icrs(observed_frame, icrs_frame):
    # first set up the astrometry context for ICRS<->observed at the observed frame's time
    astrom = erfa_astrom.get().apco(observed_frame)
    is_altaz = isinstance(observed_frame, AltAzFrame)
    # 'A' indicates zen/az inputs, 'H' hour angle/dec
    coord_type = "A" if is_altaz else "H"

    def converter(rep):
        # if the data are UnitSphericalRepresentation, we can skip the distance calculations
        is_unitspherical = isinstance(rep, UnitSphericalRepresentation) or (
            rep.represent_as(CartesianRepresentation).x.unit == u.one
        )
        usrepr = rep.represent_as(UnitSphericalRepresentation)
        lon = usrepr.lon.to_value(u.radian)
        lat = usrepr.lat.to_value(u.radian)
        if is_altaz:
            lat = PIOVER2 - lat

        # Topocentric CIRS
        cirs_ra, cirs_dec = erfa.atoiq(coord_type, lon, lat, astrom) << u.radian

        if is_unitspherical:
            srepr_3d = SphericalRepresentation(cirs_ra, cirs_dec, 1, copy=None)
        else:
            distance = rep.represent_as(SphericalRepresentation).distance
            srepr_3d = SphericalRepresentation(
                lon=cirs_ra, lat=cirs_dec, distance=distance, copy=None
            )

        # BCRS (Astrometric) direction to source
        bcrs_ra, bcrs_dec = aticq(srepr_3d, astrom) << u.radian

        # Correct for parallax to get ICRS representation
        if is_unitspherical:
            return UnitSphericalRepresentation(bcrs_ra, bcrs_dec, copy=None)
        else:
            icrs_srepr = SphericalRepresentation(
                lon=bcrs_ra, lat=bcrs_dec, distance=distance, copy=None
            )
            observer_icrs = CartesianRepresentation(
                astrom["eb"], unit=u.au, xyz_axis=-1, copy=None
            )
            return (icrs_srepr.to_cartesian() + observer_icrs).represent_as(
                SphericalRepresentation
            )

    return converter


# TODO: APE23: remove when legacy frames are deprecated
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, AltAz, ICRS)
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, HADec, ICRS)
def observed_to_icrs_legacy(observed_coo, icrs_frame):
    converter = observed_to_icrs(observed_coo, icrs_frame)
    return icrs_frame.realize_frame(converter(observed_coo.data))


# Create loopback transformations
frame_transform_graph._add_merged_transform(AltAz, ICRS, AltAz)
frame_transform_graph._add_merged_transform(HADec, ICRS, HADec)
frame_transform_graph._add_merged_transform(AltAzFrame, ICRSFrame, AltAzFrame)
frame_transform_graph._add_merged_transform(HADecFrame, ICRSFrame, HADecFrame)
# for now we just implement this through ICRS to make sure we get everything
# covered
# Before, this was using CIRS as intermediate frame, however this is much
# slower than the direct observed<->ICRS transform added in 4.3
# due to how the frame attribute broadcasting works, see
# https://github.com/astropy/astropy/pull/10994#issuecomment-722617041
