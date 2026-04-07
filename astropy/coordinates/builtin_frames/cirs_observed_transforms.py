# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Contains the transformation functions for getting to "observed" systems from CIRS.
"""

import erfa
import numpy as np

from astropy import units as u
from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates.erfa_astrom import erfa_astrom
from astropy.coordinates.representation import (
    CartesianRepresentation,
    SphericalRepresentation,
    UnitSphericalRepresentation,
)
from astropy.coordinates.transformations import (
    FunctionTransformWithFiniteDifference,
    RepresentationFunctionTransform,
)

from .altaz import AltAz, AltAzFrame
from .cirs import CIRS, CIRSFrame
from .hadec import HADec, HADecFrame
from .utils import PIOVER2


@frame_transform_graph.transform(RepresentationFunctionTransform, CIRSFrame, AltAzFrame)
@frame_transform_graph.transform(RepresentationFunctionTransform, CIRSFrame, HADecFrame)
def cirs_to_observed(cirs_frame, observed_frame):
    needs_reroute = np.any(cirs_frame.location != observed_frame.location) or np.any(
        cirs_frame.obstime != observed_frame.obstime
    )
    # set up the astrometry context for CIRS<->observed (uses observed_frame attrs only)
    astrom = erfa_astrom.get().apio(observed_frame)
    is_altaz = isinstance(observed_frame, AltAzFrame)

    def converter(rep):
        if needs_reroute:
            rep = CIRS(
                rep, obstime=cirs_frame.obstime, location=cirs_frame.location
            ).transform_to(
                CIRS(obstime=observed_frame.obstime, location=observed_frame.location)
            ).data
        # if the data are UnitSphericalRepresentation, we can skip distance calculations
        is_unitspherical = isinstance(rep, UnitSphericalRepresentation) or (
            rep.represent_as(CartesianRepresentation).x.unit == u.one
        )
        # We used to do "astrometric" corrections here, but these are no longer necessary
        # CIRS has proper topocentric behaviour
        usrepr = rep.represent_as(UnitSphericalRepresentation)
        cirs_ra = usrepr.lon.to_value(u.radian)
        cirs_dec = usrepr.lat.to_value(u.radian)
        if is_altaz:
            lon, zen, _, _, _ = erfa.atioq(cirs_ra, cirs_dec, astrom)
            lat = PIOVER2 - zen
        else:
            _, _, lon, lat, _ = erfa.atioq(cirs_ra, cirs_dec, astrom)
        if is_unitspherical:
            return UnitSphericalRepresentation(lon << u.radian, lat << u.radian, copy=False)
        else:
            # since we've transformed to CIRS at the observatory location, just use CIRS distance
            return SphericalRepresentation(
                lon << u.radian,
                lat << u.radian,
                rep.represent_as(SphericalRepresentation).distance,
                copy=False,
            )

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, CIRS, AltAz)
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, CIRS, HADec)
def cirs_to_observed_legacy(cirs_coo, observed_frame):
    converter = cirs_to_observed(cirs_coo, observed_frame)
    return observed_frame.realize_frame(converter(cirs_coo.data))


@frame_transform_graph.transform(RepresentationFunctionTransform, AltAzFrame, CIRSFrame)
@frame_transform_graph.transform(RepresentationFunctionTransform, HADecFrame, CIRSFrame)
def observed_to_cirs(observed_frame, cirs_frame):
    is_altaz = isinstance(observed_frame, AltAzFrame)
    # the 'A' indicates zen/az inputs, 'H' for HA/Dec
    coord_type = "A" if is_altaz else "H"
    # set up the astrometry context for observed<->CIRS at the observed_frame time
    astrom = erfa_astrom.get().apio(observed_frame)
    needs_reroute = np.any(observed_frame.obstime != cirs_frame.obstime) or np.any(
        observed_frame.location != cirs_frame.location
    )

    def converter(rep):
        is_unitspherical = isinstance(rep, UnitSphericalRepresentation) or (
            rep.represent_as(CartesianRepresentation).x.unit == u.one
        )
        usrepr = rep.represent_as(UnitSphericalRepresentation)
        lon = usrepr.lon.to_value(u.radian)
        lat = usrepr.lat.to_value(u.radian)
        if is_altaz:
            lat = PIOVER2 - lat
        cirs_ra, cirs_dec = erfa.atoiq(coord_type, lon, lat, astrom) << u.radian
        if is_unitspherical:
            cirs_rep = UnitSphericalRepresentation(lon=cirs_ra, lat=cirs_dec, copy=None)
        else:
            distance = rep.represent_as(SphericalRepresentation).distance
            cirs_rep = SphericalRepresentation(
                lon=cirs_ra, lat=cirs_dec, distance=distance, copy=None
            )
        # this final transform may be a no-op if the obstimes and locations are the same
        if needs_reroute:
            cirs_at_obs = CIRS(
                cirs_rep, obstime=observed_frame.obstime, location=observed_frame.location
            )
            return cirs_at_obs.transform_to(
                CIRS(obstime=cirs_frame.obstime, location=cirs_frame.location)
            ).data
        return cirs_rep

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, AltAz, CIRS)
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, HADec, CIRS)
def observed_to_cirs_legacy(observed_coo, cirs_frame):
    converter = observed_to_cirs(observed_coo, cirs_frame)
    return cirs_frame.realize_frame(converter(observed_coo.data))
