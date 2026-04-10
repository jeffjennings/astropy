# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This package contains the coordinate frames implemented by astropy.

Users shouldn't use this module directly, but rather import from the
`astropy.coordinates` module.  While it is likely to exist for the long-term,
the existence of this package and details of its organization should be
considered an implementation detail, and is not guaranteed to hold for future
versions of astropy.

Notes
-----
The builtin frame classes are all imported automatically into this package's
namespace, so there's no need to access the sub-modules directly.

To implement a new frame in Astropy, a developer should add the frame as a new
module in this package.  Any "self" transformations (i.e., those that transform
from one frame to another frame of the same class) should be included in that
module.  Transformation functions connecting the new frame to other frames
should be in a separate module, which should be imported in this package's
``__init__.py`` to ensure the transformations are hooked up when this package is
imported.  Placing the transformation functions in separate modules avoids
circular dependencies, because they need references to the frame classes.
"""

from astropy.coordinates.baseframe import frame_transform_graph

from .altaz import AltAz, AltAzFrame
from .baseradec import BaseRADecFrame
from .cirs import CIRS, CIRSFrame
from .ecliptic import (
    BarycentricMeanEcliptic,
    BarycentricMeanEclipticFrame,
    BarycentricTrueEcliptic,
    BarycentricTrueEclipticFrame,
    BaseEclipticFrame,
    CustomBarycentricEcliptic,
    CustomBarycentricEclipticFrame,
    GeocentricMeanEcliptic,
    GeocentricMeanEclipticFrame,
    GeocentricTrueEcliptic,
    GeocentricTrueEclipticFrame,
    HeliocentricEclipticIAU76,
    HeliocentricEclipticIAU76Frame,
    HeliocentricMeanEcliptic,
    HeliocentricMeanEclipticFrame,
    HeliocentricTrueEcliptic,
    HeliocentricTrueEclipticFrame,
)
from .equatorial import TEME, TETE, TEMEFrame, TETEFrame
from .fk4 import FK4, FK4NoETerms, FK4Frame, FK4NoETermsFrame
from .fk5 import FK5, FK5Frame
from .galactic import Galactic, GalacticFrame
from .galactocentric import Galactocentric, GalactocentricFrame, galactocentric_frame_defaults
from .gcrs import GCRS, PrecessedGeocentric, GCRSFrame, PrecessedGeocentricFrame
from .hadec import HADec, HADecFrame
from .hcrs import HCRS, HCRSFrame
from .icrs import ICRS, ICRSFrame
from .itrs import ITRS, ITRSFrame
from .skyoffset import SkyOffsetFrame
from .supergalactic import Supergalactic, SupergalacticFrame

# isort: split
# need to import transformations so that they get registered in the graph
from . import (
    cirs_observed_transforms,
    fk4_fk5_transforms,
    galactic_transforms,
    icrs_cirs_transforms,
    icrs_fk5_transforms,
    icrs_observed_transforms,
    intermediate_rotation_transforms,
    itrs_observed_transforms,
    supergalactic_transforms,
)

# isort: split
from . import ecliptic_transforms

# isort: split
# Import this after importing other frames, since this requires various
# transformations to set up the LSR frames
from .lsr import LSR, LSRD, LSRK, GalacticLSR, LSRFrame, LSRDFrame, LSRKFrame, GalacticLSRFrame

# we define an __all__ because otherwise the transformation modules
# get included.  Note that the order here determines the order in the
# documentation of the built-in frames (see make_transform_graphs_docs).

# ignoring RUF022 here because case-sensitive  alphanumeric sorting doesn't work here
# (as of ruff 0.8). RUF100 also ignored so older versions of ruff don't remove
# the first noqa comment
# ref https://github.com/astropy/astropy/pull/17437#discussion_r1856780149
__all__ = [  # noqa: RUF022, RUF100
    "ICRS",
    "ICRSFrame",
    "FK5",
    "FK5Frame",
    "FK4",
    "FK4Frame",
    "FK4NoETerms",
    "FK4NoETermsFrame",
    "Galactic",
    "GalacticFrame",
    "Galactocentric",
    "GalactocentricFrame",
    "Supergalactic",
    "SupergalacticFrame",
    "AltAz",
    "AltAzFrame",
    "HADec",
    "HADecFrame",
    "GCRS",
    "GCRSFrame",
    "CIRS",
    "CIRSFrame",
    "ITRS",
    "ITRSFrame",
    "HCRS",
    "HCRSFrame",
    "TEME",
    "TEMEFrame",
    "TETE",
    "TETEFrame",
    "PrecessedGeocentric",
    "PrecessedGeocentricFrame",
    "GeocentricMeanEcliptic",
    "GeocentricMeanEclipticFrame",
    "BarycentricMeanEcliptic",
    "BarycentricMeanEclipticFrame",
    "HeliocentricMeanEcliptic",
    "HeliocentricMeanEclipticFrame",
    "GeocentricTrueEcliptic",
    "GeocentricTrueEclipticFrame",
    "BarycentricTrueEcliptic",
    "BarycentricTrueEclipticFrame",
    "HeliocentricTrueEcliptic",
    "HeliocentricTrueEclipticFrame",
    "HeliocentricEclipticIAU76",
    "HeliocentricEclipticIAU76Frame",
    "CustomBarycentricEcliptic",
    "CustomBarycentricEclipticFrame",
    "LSR",
    "LSRFrame",
    "LSRK",
    "LSRKFrame",
    "LSRD",
    "LSRDFrame",
    "GalacticLSR",
    "GalacticLSRFrame",
    "SkyOffsetFrame",
    "BaseEclipticFrame",
    "BaseRADecFrame",
    "galactocentric_frame_defaults",
    "make_transform_graph_docs",
]


def _get_doc_header(cls):
    """Get the first line of a docstring.

    Skips possible empty first lines, and then combine following text until
    the first period or a fully empty line.
    """
    out = []

    # NOTE: cls.__doc__ is None for -OO flag
    if not cls.__doc__:
        return ""

    for line in cls.__doc__.splitlines():
        if line:
            parts = line.split(".")
            out.append(parts[0].strip())
            if len(parts) > 1:
                break
        elif out:
            break
    return " ".join(out) + "."


def make_transform_graph_docs(transform_graph):
    """
    Generates a string that can be used in other docstrings to include a
    transformation graph, showing the available transforms and
    coordinate systems.

    Parameters
    ----------
    transform_graph : `~astropy.coordinates.TransformGraph`

    Returns
    -------
    docstring : str
        A string that can be added to the end of a docstring to show the
        transform graph.
    """
    from textwrap import dedent

    coosys = {
        (cls := transform_graph.lookup_name(item)).__name__: cls
        for item in transform_graph.get_names()
    }

    # currently, all of the priorities are set to 1, so we don't need to show
    #   then in the transform graph.
    graphstr = transform_graph.to_dot_graph(
        addnodes=list(coosys.values()), priorities=False
    )

    docstr = """
    The diagram below shows all of the built in coordinate systems,
    their aliases (useful for converting other coordinates to them using
    attribute-style access) and the pre-defined transformations between
    them.  The user is free to override any of these transformations by
    defining new transformations between these systems, but the
    pre-defined transformations should be sufficient for typical usage.

    The color of an edge in the graph (i.e., the transformations between two
    frames) is set by the type of transformation; the legend box defines the
    mapping from transform class name to color.

    .. Wrap the graph in a div with a custom class to allow theming.
    .. container:: frametransformgraph

        .. graphviz::

    """

    docstr = dedent(docstr) + "        " + graphstr.replace("\n", "\n        ")

    # colors are in dictionary mapping transform class to color
    from astropy.coordinates.transformations.graph import trans_to_color

    html_list_items = []
    for cls, color in trans_to_color.items():
        block = f"""
            <li style='list-style: none;'>
                <p style="font-size: 12px;line-height: 24px;font-weight: normal;color: #848484;padding: 0;margin: 0;">
                    <b>{cls.__name__}:</b>
                    <span style="font-size: 24px; color: {color};"><b>➝</b></span>
                </p>
            </li>
        """
        html_list_items.append(block)

    nl = "\n"
    graph_legend = f"""
    .. raw:: html

        <ul class="cooframelegend">
            {nl.join(html_list_items)}
        </ul>
    """
    docstr = docstr + dedent(graph_legend)

    # Add table with built-in frame classes.
    template = """
       * - `~astropy.coordinates.{}`
         - {}
    """
    table = """
    Built-in Frame Classes
    ^^^^^^^^^^^^^^^^^^^^^^
    .. list-table::
       :widths: 20 80
    """ + "".join(
        template.format(name, _get_doc_header(coosys[name]))
        for name in __all__
        if name in coosys
    )

    return docstr + dedent(table)


_transform_graph_docs = make_transform_graph_docs(frame_transform_graph)

# Here, we override the module docstring so that sphinx renders the transform
# graph without the developer documentation in the main docstring above.
__doc__ = _transform_graph_docs
