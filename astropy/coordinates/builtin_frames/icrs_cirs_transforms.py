# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Contains the transformation functions for getting from ICRS/HCRS to CIRS and
anything in between (currently that means GCRS).
"""

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
    AffineTransform,
    FunctionTransformWithFiniteDifference,
    RepresentationFunctionTransform,
)

from .cirs import CIRS, CIRSFrame
from .gcrs import GCRS, GCRSFrame
from .hcrs import HCRS, HCRSFrame
from .icrs import ICRS, ICRSFrame
from .utils import atciqz, aticq, get_offset_sun_from_barycenter


# First the ICRS/CIRS related transforms
@frame_transform_graph.transform(RepresentationFunctionTransform, ICRSFrame, CIRSFrame)
def icrs_to_cirs(icrs_frame, cirs_frame):
    # first set up the astrometry context for ICRS<->CIRS
    astrom = erfa_astrom.get().apco(cirs_frame)

    def converter(rep):
        if isinstance(rep, UnitSphericalRepresentation) or (
            rep.represent_as(CartesianRepresentation).x.unit == u.one
        ):
            # if no distance, just do the infinite-distance/no parallax calculation
            srepr = rep.represent_as(SphericalRepresentation)
            cirs_ra, cirs_dec = atciqz(srepr.without_differentials(), astrom)
            return UnitSphericalRepresentation(
                lat=u.Quantity(cirs_dec, u.radian, copy=None),
                lon=u.Quantity(cirs_ra, u.radian, copy=None),
                copy=False,
            )
        else:
            # When there is a distance, we first offset for parallax to get the
            # astrometric coordinate direction and *then* run the ERFA transform for
            # no parallax/PM. This ensures reversibility and is more sensible for
            # inside solar system objects
            astrom_eb = CartesianRepresentation(
                astrom["eb"], unit=u.au, xyz_axis=-1, copy=None
            )
            newcart = rep.represent_as(CartesianRepresentation) - astrom_eb
            srepr = newcart.represent_as(SphericalRepresentation)
            cirs_ra, cirs_dec = atciqz(srepr.without_differentials(), astrom)
            return SphericalRepresentation(
                lat=u.Quantity(cirs_dec, u.radian, copy=None),
                lon=u.Quantity(cirs_ra, u.radian, copy=None),
                distance=srepr.distance,
                copy=False,
            )

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ICRS, CIRS)
def icrs_to_cirs_legacy(icrs_coo, cirs_frame):
    converter = icrs_to_cirs(icrs_coo, cirs_frame)
    return cirs_frame.realize_frame(converter(icrs_coo.data))


@frame_transform_graph.transform(RepresentationFunctionTransform, CIRSFrame, ICRSFrame)
def cirs_to_icrs(cirs_frame, icrs_frame):
    # set up the astrometry context for ICRS<->CIRS and then convert to
    # astrometric coordinate direction
    astrom = erfa_astrom.get().apco(cirs_frame)

    def converter(rep):
        srepr = rep.represent_as(SphericalRepresentation)
        i_ra, i_dec = aticq(srepr.without_differentials(), astrom)
        if isinstance(rep, UnitSphericalRepresentation) or (
            rep.represent_as(CartesianRepresentation).x.unit == u.one
        ):
            # if no distance, just use the coordinate direction to yield the
            # infinite-distance/no parallax answer
            return UnitSphericalRepresentation(
                lat=u.Quantity(i_dec, u.radian, copy=None),
                lon=u.Quantity(i_ra, u.radian, copy=None),
                copy=False,
            )
        else:
            # When there is a distance, apply the parallax/offset to the SSB as the
            # last step - ensures round-tripping with the icrs_to_cirs transform

            # the distance in intermedrep is *not* a real distance as it does not
            # include the offset back to the SSB
            intermedrep = SphericalRepresentation(
                lat=u.Quantity(i_dec, u.radian, copy=None),
                lon=u.Quantity(i_ra, u.radian, copy=None),
                distance=srepr.distance,
                copy=False,
            )
            astrom_eb = CartesianRepresentation(
                astrom["eb"], unit=u.au, xyz_axis=-1, copy=None
            )
            return intermedrep + astrom_eb

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, CIRS, ICRS)
def cirs_to_icrs_legacy(cirs_coo, icrs_frame):
    converter = cirs_to_icrs(cirs_coo, icrs_frame)
    return icrs_frame.realize_frame(converter(cirs_coo.data))


# Now the GCRS-related transforms to/from ICRS


@frame_transform_graph.transform(RepresentationFunctionTransform, ICRSFrame, GCRSFrame)
def icrs_to_gcrs(icrs_frame, gcrs_frame):
    # first set up the astrometry context for ICRS<->GCRS.
    astrom = erfa_astrom.get().apcs(gcrs_frame)

    def converter(rep):
        if isinstance(rep, UnitSphericalRepresentation) or (
            rep.represent_as(CartesianRepresentation).x.unit == u.one
        ):
            # if no distance, just do the infinite-distance/no parallax calculation
            srepr = rep.represent_as(SphericalRepresentation)
            gcrs_ra, gcrs_dec = atciqz(srepr.without_differentials(), astrom)
            return UnitSphericalRepresentation(
                lat=u.Quantity(gcrs_dec, u.radian, copy=None),
                lon=u.Quantity(gcrs_ra, u.radian, copy=None),
                copy=False,
            )
        else:
            # When there is a distance, we first offset for parallax to get the
            # BCRS coordinate direction and *then* run the ERFA transform for no
            # parallax/PM. This ensures reversibility and is more sensible for
            # inside solar system objects
            astrom_eb = CartesianRepresentation(
                astrom["eb"], unit=u.au, xyz_axis=-1, copy=None
            )
            newcart = rep.represent_as(CartesianRepresentation) - astrom_eb
            srepr = newcart.represent_as(SphericalRepresentation)
            gcrs_ra, gcrs_dec = atciqz(srepr.without_differentials(), astrom)
            return SphericalRepresentation(
                lat=u.Quantity(gcrs_dec, u.radian, copy=None),
                lon=u.Quantity(gcrs_ra, u.radian, copy=None),
                distance=srepr.distance,
                copy=False,
            )

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ICRS, GCRS)
def icrs_to_gcrs_legacy(icrs_coo, gcrs_frame):
    converter = icrs_to_gcrs(icrs_coo, gcrs_frame)
    return gcrs_frame.realize_frame(converter(icrs_coo.data))


@frame_transform_graph.transform(RepresentationFunctionTransform, GCRSFrame, ICRSFrame)
def gcrs_to_icrs(gcrs_frame, icrs_frame):
    # set up the astrometry context for ICRS<->GCRS and then convert to BCRS
    # coordinate direction
    astrom = erfa_astrom.get().apcs(gcrs_frame)

    def converter(rep):
        srepr = rep.represent_as(SphericalRepresentation)
        i_ra, i_dec = aticq(srepr.without_differentials(), astrom)
        if isinstance(rep, UnitSphericalRepresentation) or (
            rep.represent_as(CartesianRepresentation).x.unit == u.one
        ):
            # if no distance, just use the coordinate direction to yield the
            # infinite-distance/no parallax answer
            return UnitSphericalRepresentation(
                lat=u.Quantity(i_dec, u.radian, copy=None),
                lon=u.Quantity(i_ra, u.radian, copy=None),
                copy=False,
            )
        else:
            # When there is a distance, apply the parallax/offset to the SSB as the
            # last step - ensures round-tripping with the icrs_to_gcrs transform

            # the distance in intermedrep is *not* a real distance as it does not
            # include the offset back to the SSB
            intermedrep = SphericalRepresentation(
                lat=u.Quantity(i_dec, u.radian, copy=None),
                lon=u.Quantity(i_ra, u.radian, copy=None),
                distance=srepr.distance,
                copy=False,
            )
            astrom_eb = CartesianRepresentation(
                astrom["eb"], unit=u.au, xyz_axis=-1, copy=None
            )
            return intermedrep + astrom_eb

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, GCRS, ICRS)
def gcrs_to_icrs_legacy(gcrs_coo, icrs_frame):
    converter = gcrs_to_icrs(gcrs_coo, icrs_frame)
    return icrs_frame.realize_frame(converter(gcrs_coo.data))


@frame_transform_graph.transform(RepresentationFunctionTransform, GCRSFrame, HCRSFrame)
def gcrs_to_hcrs(gcrs_frame, hcrs_frame):
    needs_reroute = np.any(gcrs_frame.obstime != hcrs_frame.obstime)
    if needs_reroute:
        # if the GCRS and HCRS obstimes are not the same, we first have to move
        # to a GCRS where they are.  Use class defaults for obsgeoloc/obsgeovel
        # (i.e., geocentric) since HCRS is a geocentric frame.
        reroute_attrs = dict(GCRSFrame.get_frame_attr_defaults())
        reroute_attrs["obstime"] = hcrs_frame.obstime
        astrom = erfa_astrom.get().apcs(GCRSFrame(**reroute_attrs))
    else:
        astrom = erfa_astrom.get().apcs(gcrs_frame)

    def converter(rep):
        if needs_reroute:
            rep = GCRS(
                rep,
                obstime=gcrs_frame.obstime,
                obsgeoloc=gcrs_frame.obsgeoloc,
                obsgeovel=gcrs_frame.obsgeovel,
            ).transform_to(GCRS(**reroute_attrs)).data
        # set up the astrometry context for ICRS<->GCRS and then convert to ICRS
        # coordinate direction
        srepr = rep.represent_as(SphericalRepresentation)
        i_ra, i_dec = aticq(srepr.without_differentials(), astrom)
        # convert to Quantity objects
        i_ra = u.Quantity(i_ra, u.radian, copy=None)
        i_dec = u.Quantity(i_dec, u.radian, copy=None)
        if isinstance(rep, UnitSphericalRepresentation) or (
            rep.represent_as(CartesianRepresentation).x.unit == u.one
        ):
            # if no distance, just use the coordinate direction to yield the
            # infinite-distance/no parallax answer
            return UnitSphericalRepresentation(lat=i_dec, lon=i_ra, copy=False)
        else:
            # When there is a distance, apply the parallax/offset to the
            # Heliocentre as the last step to ensure round-tripping with the
            # hcrs_to_gcrs transform

            # Note that the distance in intermedrep is *not* a real distance as it
            # does not include the offset back to the Heliocentre
            intermedrep = SphericalRepresentation(
                lat=i_dec, lon=i_ra, distance=srepr.distance, copy=False
            )
            # astrom['eh'] and astrom['em'] contain Sun to observer unit vector,
            # and distance, respectively. Shapes are (X) and (X,3), where (X) is
            # the shape resulting from broadcasting the shape of the times object
            # against the shape of the pv array.
            # broadcast em to eh and scale eh
            eh = astrom["eh"] * astrom["em"][..., np.newaxis]
            eh = CartesianRepresentation(eh, unit=u.au, xyz_axis=-1, copy=None)
            return intermedrep.to_cartesian() + eh

    return converter


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, GCRS, HCRS)
def gcrs_to_hcrs_legacy(gcrs_coo, hcrs_frame):
    converter = gcrs_to_hcrs(gcrs_coo, hcrs_frame)
    return hcrs_frame.realize_frame(converter(gcrs_coo.data))


_NEED_ORIGIN_HINT = (
    "The input {0} coordinates do not have length units. This probably means you"
    " created coordinates with lat/lon but no distance.  Heliocentric<->ICRS transforms"
    " cannot function in this case because there is an origin shift."
)


@frame_transform_graph.transform(AffineTransform, HCRS, ICRS)
def hcrs_to_icrs(hcrs_coo, icrs_frame):
    # this is just an origin translation so without a distance it cannot go ahead
    if isinstance(hcrs_coo.data, UnitSphericalRepresentation):
        raise u.UnitsError(_NEED_ORIGIN_HINT.format(hcrs_coo.__class__.__name__))

    return None, get_offset_sun_from_barycenter(
        hcrs_coo.obstime, include_velocity=bool(hcrs_coo.data.differentials)
    )


@frame_transform_graph.transform(AffineTransform, ICRS, HCRS)
def icrs_to_hcrs(icrs_coo, hcrs_frame):
    # this is just an origin translation so without a distance it cannot go ahead
    if isinstance(icrs_coo.data, UnitSphericalRepresentation):
        raise u.UnitsError(_NEED_ORIGIN_HINT.format(icrs_coo.__class__.__name__))

    return None, get_offset_sun_from_barycenter(
        hcrs_frame.obstime,
        reverse=True,
        include_velocity=bool(icrs_coo.data.differentials),
    )


# Create loopback transformations
frame_transform_graph._add_merged_transform(CIRS, ICRS, CIRS)
# The CIRS<-> CIRS transform going through ICRS has a
# subtle implication that a point in CIRS is uniquely determined
# by the corresponding astrometric ICRS coordinate *at its
# current time*.  This has some subtle implications in terms of GR, but
# is sort of glossed over in the current scheme because we are dropping
# distances anyway.
frame_transform_graph._add_merged_transform(GCRS, ICRS, GCRS)
frame_transform_graph._add_merged_transform(HCRS, ICRS, HCRS)
frame_transform_graph._add_merged_transform(CIRSFrame, ICRSFrame, CIRSFrame)
frame_transform_graph._add_merged_transform(GCRSFrame, ICRSFrame, GCRSFrame)
frame_transform_graph._add_merged_transform(HCRSFrame, ICRSFrame, HCRSFrame)
