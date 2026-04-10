import erfa
import numpy as np

from astropy import units as u
from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates.coordinate import Coordinate
from astropy.coordinates.matrix_utilities import matrix_transpose, rotation_matrix
from astropy.coordinates.representation import CartesianRepresentation
from astropy.coordinates.transformations import (
    FunctionTransformWithFiniteDifference,
    RepresentationFunctionTransformWithFiniteDifference,
)

from .altaz import AltAz, AltAzFrame
from .hadec import HADec, HADecFrame
from .itrs import ITRS, ITRSFrame

# Minimum cos(alt) and sin(alt) for refraction purposes
CELMIN = 1e-6
SELMIN = 0.05
# Latitude of the north pole.
NORTH_POLE = 90.0 * u.deg


def itrs_to_altaz_mat(lon, lat):
    # form ITRS to AltAz matrix
    # AltAz frame is left handed
    minus_x = np.eye(3)
    minus_x[0][0] = -1.0
    return minus_x @ rotation_matrix(NORTH_POLE - lat, "y") @ rotation_matrix(lon, "z")


def itrs_to_hadec_mat(lon):
    # form ITRS to HADec matrix
    # HADec frame is left handed
    minus_y = np.eye(3)
    minus_y[1][1] = -1.0
    return minus_y @ rotation_matrix(lon, "z")


def altaz_to_hadec_mat(lat):
    # form AltAz to HADec matrix
    z180 = np.eye(3)
    z180[0][0] = -1.0
    z180[1][1] = -1.0
    return z180 @ rotation_matrix(NORTH_POLE - lat, "y")


def add_refraction(aa_crepr, observed_frame):
    # add refraction to AltAz cartesian representation
    refa, refb = erfa.refco(
        observed_frame.pressure.to_value(u.hPa),
        observed_frame.temperature.to_value(u.deg_C),
        observed_frame.relative_humidity.value,
        observed_frame.obswl.to_value(u.micron),
    )
    # reference: erfa.atioq()
    norm, uv = erfa.pn(aa_crepr.get_xyz(xyz_axis=-1).to_value())
    # Cosine and sine of altitude, with precautions.
    sel = np.maximum(uv[..., 2], SELMIN)
    cel = np.maximum(np.sqrt(uv[..., 0] ** 2 + uv[..., 1] ** 2), CELMIN)
    # A*tan(z)+B*tan^3(z) model, with Newton-Raphson correction.
    tan_z = cel / sel
    w = refb * tan_z**2
    delta_el = (refa + w) * tan_z / (1.0 + (refa + 3.0 * w) / (sel**2))
    # Apply the change, giving observed vector
    cosdel = 1.0 - 0.5 * delta_el**2
    f = cosdel - delta_el * sel / cel
    uv[..., 0] *= f
    uv[..., 1] *= f
    uv[..., 2] = cosdel * uv[..., 2] + delta_el * cel
    # Need to renormalize to get agreement with CIRS->Observed on distance
    norm2, uv = erfa.pn(uv)
    uv = erfa.sxp(norm, uv)
    return CartesianRepresentation(uv, xyz_axis=-1, unit=aa_crepr.x.unit, copy=False)


def remove_refraction(aa_crepr, observed_frame):
    # remove refraction from AltAz cartesian representation
    refa, refb = erfa.refco(
        observed_frame.pressure.to_value(u.hPa),
        observed_frame.temperature.to_value(u.deg_C),
        observed_frame.relative_humidity.value,
        observed_frame.obswl.to_value(u.micron),
    )
    # reference: erfa.atoiq()
    norm, uv = erfa.pn(aa_crepr.get_xyz(xyz_axis=-1).to_value())
    # Cosine and sine of altitude, with precautions.
    sel = np.maximum(uv[..., 2], SELMIN)
    cel = np.sqrt(uv[..., 0] ** 2 + uv[..., 1] ** 2)
    # A*tan(z)+B*tan^3(z) model
    tan_z = cel / sel
    delta_el = (refa + refb * tan_z**2) * tan_z
    # Apply the change, giving observed vector.
    az, el = erfa.c2s(uv)
    el -= delta_el
    uv = erfa.s2c(az, el)
    uv = erfa.sxp(norm, uv)
    return CartesianRepresentation(uv, xyz_axis=-1, unit=aa_crepr.x.unit, copy=False)


@frame_transform_graph.transform(
    RepresentationFunctionTransformWithFiniteDifference, ITRSFrame, AltAzFrame
)
@frame_transform_graph.transform(
    RepresentationFunctionTransformWithFiniteDifference, ITRSFrame, HADecFrame
)
def itrs_to_observed(itrs_frame, observed_frame):
    needs_reroute = np.any(itrs_frame.location != observed_frame.location) or np.any(
        itrs_frame.obstime != observed_frame.obstime
    )
    lon, lat, height = observed_frame.location.to_geodetic("WGS84")
    use_altaz = isinstance(observed_frame, AltAzFrame) or (
        observed_frame.pressure > 0.0
    )
    apply_refraction = observed_frame.pressure > 0.0
    is_hadec = isinstance(observed_frame, HADecFrame)
    mat = itrs_to_altaz_mat(lon, lat) if use_altaz else itrs_to_hadec_mat(lon)

    def converter(rep):
        cart = rep.represent_as(CartesianRepresentation)
        if needs_reroute:
            # This transform will go through the CIRS and alter stellar aberration.
            cart = (
                Coordinate(
                    ITRSFrame(obstime=itrs_frame.obstime, location=itrs_frame.location),
                    cart,
                )
                .transform_to(
                    ITRSFrame(
                        obstime=observed_frame.obstime, location=observed_frame.location
                    )
                )
                .data
            )
        result = cart.transform(mat)
        if apply_refraction:
            result = add_refraction(result, observed_frame)
            if is_hadec:
                result = result.transform(altaz_to_hadec_mat(lat))
        return result

    return converter


# TODO: APE23: remove when legacy frames are deprecated
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ITRS, AltAz)
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ITRS, HADec)
def itrs_to_observed_legacy(itrs_coo, observed_frame):
    converter = itrs_to_observed(itrs_coo, observed_frame)
    return observed_frame.realize_frame(converter(itrs_coo.cartesian))


@frame_transform_graph.transform(
    RepresentationFunctionTransformWithFiniteDifference, AltAzFrame, ITRSFrame
)
@frame_transform_graph.transform(
    RepresentationFunctionTransformWithFiniteDifference, HADecFrame, ITRSFrame
)
def observed_to_itrs(observed_frame, itrs_frame):
    lon, lat, height = observed_frame.location.to_geodetic("WGS84")
    is_altaz = isinstance(observed_frame, AltAzFrame)
    apply_refraction = observed_frame.pressure > 0.0
    is_hadec = isinstance(observed_frame, HADecFrame)
    needs_reroute = np.any(observed_frame.obstime != itrs_frame.obstime) or np.any(
        observed_frame.location != itrs_frame.location
    )

    def converter(rep):
        cart = rep.represent_as(CartesianRepresentation)
        if is_altaz or apply_refraction:
            if apply_refraction:
                if is_hadec:
                    cart = cart.transform(matrix_transpose(altaz_to_hadec_mat(lat)))
                cart = remove_refraction(cart, observed_frame)
            cart = cart.transform(matrix_transpose(itrs_to_altaz_mat(lon, lat)))
        else:
            cart = cart.transform(matrix_transpose(itrs_to_hadec_mat(lon)))
        # This final transform may be a no-op if the obstimes and locations are the same.
        # Otherwise, this transform will go through the CIRS and alter stellar aberration.
        if needs_reroute:
            return (
                Coordinate(
                    ITRSFrame(
                        obstime=observed_frame.obstime, location=observed_frame.location
                    ),
                    cart,
                )
                .transform_to(
                    ITRSFrame(obstime=itrs_frame.obstime, location=itrs_frame.location)
                )
                .data
            )
        return cart

    return converter


# TODO: APE23: remove when legacy frames are deprecated
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, AltAz, ITRS)
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, HADec, ITRS)
def observed_to_itrs_legacy(observed_coo, itrs_frame):
    converter = observed_to_itrs(observed_coo, itrs_frame)
    return itrs_frame.realize_frame(converter(observed_coo.cartesian))
