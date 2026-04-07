# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.coordinates.attributes import TimeAttribute
from astropy.coordinates.baseframe import BaseCoordinateFrame, base_doc, base_doc_frame
from astropy.utils.decorators import format_doc

from .baseradec import BaseRADecFrame, doc_components
from .utils import DEFAULT_OBSTIME

__all__ = ["HCRS", "HCRSFrame"]


doc_footer = """
    Other parameters
    ----------------
    obstime : `~astropy.time.Time`
        The time at which the observation is taken.  Used for determining the
        position of the Sun.
"""


@format_doc(base_doc_frame, footer=doc_footer)
class HCRSFrame(BaseRADecFrame):
    """
    A coordinate or frame in a Heliocentric system, with axes aligned to ICRS.

    The ICRS has an origin at the Barycenter and axes which are fixed with
    respect to space.

    This coordinate system is distinct from ICRS mainly in that it is relative
    to the Sun's center-of-mass rather than the solar system Barycenter.
    In principle, therefore, this frame should include the effects of
    aberration (unlike ICRS), but this is not done, since they are very small,
    of the order of 8 milli-arcseconds.

    For more background on the ICRS and related coordinate transformations, see
    the references provided in the :ref:`astropy:astropy-coordinates-seealso`
    section of the documentation.

    The frame attributes are listed under **Other Parameters**.

    NOTE:
    This class only holds metadata defining the HCRS reference frame.
    It does not store coordinate data. To store coordinate data in this frame,
    use `~astropy.coordinates.Coordinate`, `~astropy.coordinates.SkyCoord` or the
    legacy `HCRS` class.       
    """

    name = "hcrs"

    obstime = TimeAttribute(
        default=DEFAULT_OBSTIME, doc="The reference time (e.g., time of observation)"
    )


@format_doc(base_doc, components=doc_components, footer=doc_footer)
class HCRS(BaseCoordinateFrame, HCRSFrame):
    """
    A coordinate or frame in a Heliocentric system, with axes aligned to ICRS.

    The ICRS has an origin at the Barycenter and axes which are fixed with
    respect to space.

    This coordinate system is distinct from ICRS mainly in that it is relative
    to the Sun's center-of-mass rather than the solar system Barycenter.
    In principle, therefore, this frame should include the effects of
    aberration (unlike ICRS), but this is not done, since they are very small,
    of the order of 8 milli-arcseconds.

    For more background on the ICRS and related coordinate transformations, see
    the references provided in the :ref:`astropy:astropy-coordinates-seealso`
    section of the documentation.

    The frame attributes are listed under **Other Parameters**.
    """
    pass


# Transformations are defined in icrs_circ_transforms.py
