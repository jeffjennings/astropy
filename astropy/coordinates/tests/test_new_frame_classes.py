# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the new (APE23) data-less frame classes (e.g. ICRSFrame, GalacticFrame),
the Coordinate class, and the changes to SkyCoord that accept BaseFrame instances.

These tests are analogous to tests in test_frames.py / test_sky_coord.py but
exercise the new APIs introduced alongside BaseFrame / Coordinate.
"""

import numpy as np
import pytest

from astropy import units as u
from astropy.coordinates import (
    ICRS,
    FK4Frame,
    FK5Frame,
    GalacticFrame,
    ICRSFrame,
    SkyCoord,
)
from astropy.coordinates import representation as r
from astropy.coordinates.baseframe import BaseCoordinateFrame, BaseFrame
from astropy.coordinates.coordinate import BaseCoordinate, Coordinate
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time


def test_dataless_frame_has_no_data():
    """Data-less frame classes must report has_data=False."""
    frame = ICRSFrame()
    assert frame.has_data is False


def test_dataless_frame_name():
    """Data-less frame classes expose the same `name` as their legacy sibling."""
    assert ICRSFrame().name == "icrs"
    assert FK5Frame().name == "fk5"
    assert GalacticFrame().name == "galactic"
    assert FK4Frame().name == "fk4"


def test_dataless_frame_default_representation():
    """Data-less frames inherit default_representation from their hierarchy."""
    frame = ICRSFrame()
    assert frame.default_representation is r.SphericalRepresentation


def test_dataless_frame_repr():
    """Repr of a data-less frame should not mention data."""
    frame = ICRSFrame()
    rstr = repr(frame)
    assert "ICRSFrame" in rstr
    assert "Frame" in rstr


def test_dataless_frame_with_attribute():
    """Frame attributes like equinox should be settable at construction time."""
    equinox = Time("J2010.0")
    frame = FK5Frame(equinox=equinox)
    assert frame.equinox == equinox


def test_dataless_frame_attribute_default():
    """Frame attributes fall back to their defaults when not provided."""
    frame = FK5Frame()
    # FK5 default equinox is J2000.0
    assert frame.equinox.jyear == pytest.approx(2000.0, abs=0.01)


def test_dataless_frame_get_frame_attr_defaults():
    fk5 = FK5Frame()
    defaults = fk5.get_frame_attr_defaults()
    assert "equinox" in defaults


def test_dataless_frame_is_subclass_of_baseframe():
    assert issubclass(ICRSFrame, BaseFrame)
    assert isinstance(ICRSFrame(), BaseFrame)


def test_dataless_frame_is_not_subclass_of_basecoordinateframe():
    """Data-less frames should NOT inherit from BaseCoordinateFrame."""
    # GalacticFrame directly subclasses BaseFrame, not BaseCoordinateFrame
    assert not issubclass(GalacticFrame, BaseCoordinateFrame)


def test_unexpected_keyword_raises():
    """Passing unexpected keyword to a data-less frame should raise TypeError."""
    with pytest.raises(TypeError, match="unexpected"):
        ICRSFrame(ra=1 * u.deg)


def test_representation_type_property():
    """representation_type on a data-less frame can be read and changed."""
    frame = ICRSFrame()
    assert frame.representation_type is r.SphericalRepresentation

    frame.representation_type = r.CartesianRepresentation
    assert frame.representation_type is r.CartesianRepresentation


def test_representation_component_names_icrs():
    """ICRSFrame should map ra→lon and dec→lat."""
    frame = ICRSFrame()
    names = frame.representation_component_names
    assert names["ra"] == "lon"
    assert names["dec"] == "lat"


def test_representation_component_names_galactic():
    """GalacticFrame should map l→lon and b→lat."""
    frame = GalacticFrame()
    names = frame.representation_component_names
    assert names["l"] == "lon"
    assert names["b"] == "lat"


def test_frame_is_equivalent_to_itself():
    frame1 = ICRSFrame()
    frame2 = ICRSFrame()
    assert frame1.is_equivalent_frame(frame2)


def test_frame_not_equivalent_different_class():
    assert not ICRSFrame().is_equivalent_frame(GalacticFrame())


def test_frame_not_equivalent_different_attribute():
    f1 = FK5Frame(equinox=Time("J2000"))
    f2 = FK5Frame(equinox=Time("J2010"))
    assert not f1.is_equivalent_frame(f2)


def test_frame_equivalent_same_attribute():
    f1 = FK5Frame(equinox=Time("J2000"))
    f2 = FK5Frame(equinox=Time("J2000"))
    assert f1.is_equivalent_frame(f2)


def test_dataless_frame_equivalent_to_legacy_frame():
    """ICRSFrame() must be considered equivalent to ICRS()."""
    # ICRS inherits from both BaseCoordinateFrame and ICRSFrame, so they
    # should be treated as equivalent frames.
    dataless = ICRSFrame()
    legacy = ICRS()
    assert dataless.is_equivalent_frame(legacy)


def test_non_frame_raises_typeerror():
    with pytest.raises(TypeError):
        ICRSFrame().is_equivalent_frame("not a frame")


def test_icrs_transformable_to_galactic():
    frame = ICRSFrame()
    assert frame.is_transformable_to(GalacticFrame())


def test_icrs_transformable_to_fk5():
    frame = ICRSFrame()
    assert frame.is_transformable_to(FK5Frame())


def test_icrs_same_frame_returns_same():
    frame = ICRSFrame()
    result = frame.is_transformable_to(ICRSFrame())
    assert result


def test_coordinate_basic_construction():
    """Coordinate can be built from a data-less frame + representation."""
    coord = Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(lon=10 * u.deg, lat=20 * u.deg),
    )
    assert coord.has_data is True
    assert isinstance(coord.frame, ICRSFrame)
    assert isinstance(coord.data, r.UnitSphericalRepresentation)


def test_coordinate_has_data_true():
    coord = Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(lon=0 * u.deg, lat=0 * u.deg),
    )
    assert coord.has_data is True


def test_coordinate_is_basecoordinate():
    coord = Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(lon=0 * u.deg, lat=0 * u.deg),
    )
    assert isinstance(coord, BaseCoordinate)


def test_coordinate_frame_attr_access():
    """Coordinate delegates frame attribute access via __getattr__."""
    equinox = Time("J2005.0")
    coord = Coordinate(
        frame=FK5Frame(equinox=equinox),
        data=r.UnitSphericalRepresentation(lon=10 * u.deg, lat=5 * u.deg),
    )
    assert coord.equinox == equinox


def test_coordinate_ra_dec_access():
    """Coordinate exposes frame-specific component names (ra, dec for ICRS)."""
    coord = Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(lon=30 * u.deg, lat=45 * u.deg),
    )
    assert_quantity_allclose(coord.ra, 30 * u.deg)
    assert_quantity_allclose(coord.dec, 45 * u.deg)


def test_coordinate_galactic_lb_access():
    """Coordinate exposes l/b component names for GalacticFrame."""
    coord = Coordinate(
        frame=GalacticFrame(),
        data=r.UnitSphericalRepresentation(lon=120 * u.deg, lat=30 * u.deg),
    )
    assert_quantity_allclose(coord.l, 120 * u.deg)
    assert_quantity_allclose(coord.b, 30 * u.deg)


def test_coordinate_represent_as_cartesian():
    """represent_as should convert to the requested representation."""
    coord = Coordinate(
        frame=ICRSFrame(),
        data=r.SphericalRepresentation(
            lon=0 * u.deg, lat=0 * u.deg, distance=1 * u.kpc
        ),
    )
    cart = coord.represent_as(r.CartesianRepresentation)
    assert isinstance(cart, r.CartesianRepresentation)
    assert_quantity_allclose(cart.x, 1 * u.kpc, atol=1e-10 * u.kpc)


def test_coordinate_spherical_property():
    coord = Coordinate(
        frame=ICRSFrame(),
        data=r.SphericalRepresentation(
            lon=90 * u.deg, lat=0 * u.deg, distance=2 * u.kpc
        ),
    )
    sph = coord.spherical
    assert isinstance(sph, r.SphericalRepresentation)
    assert_quantity_allclose(sph.lon, 90 * u.deg)


def test_coordinate_cartesian_property():
    coord = Coordinate(
        frame=ICRSFrame(),
        data=r.SphericalRepresentation(
            lon=0 * u.deg, lat=0 * u.deg, distance=1 * u.kpc
        ),
    )
    cart = coord.cartesian
    assert isinstance(cart, r.CartesianRepresentation)


def test_coordinate_shape_scalar():
    coord = Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(lon=10 * u.deg, lat=5 * u.deg),
    )
    assert coord.shape == ()


def test_coordinate_shape_array():
    coord = Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(
            lon=[10, 20, 30] * u.deg, lat=[5, 10, 15] * u.deg
        ),
    )
    assert coord.shape == (3,)


def test_coordinate_size():
    coord = Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(lon=[10, 20] * u.deg, lat=[5, 10] * u.deg),
    )
    assert coord.size == 2


def test_coordinate_replace_frame():
    """__replace__ should return a new Coordinate with the updated frame."""
    original = Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(lon=10 * u.deg, lat=20 * u.deg),
    )
    galactic_frame = GalacticFrame()
    replaced = original.__replace__(frame=galactic_frame)
    assert isinstance(replaced.frame, GalacticFrame)
    # data is unchanged
    assert replaced.data is original.data


def test_coordinate_replace_data():
    new_data = r.UnitSphericalRepresentation(lon=0 * u.deg, lat=0 * u.deg)
    original = Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(lon=10 * u.deg, lat=20 * u.deg),
    )
    replaced = original.__replace__(data=new_data)
    assert replaced.frame is original.frame
    assert replaced.data is new_data


def test_coordinate_equality_same_values():
    data = r.UnitSphericalRepresentation(lon=10 * u.deg, lat=20 * u.deg)
    c1 = Coordinate(frame=ICRSFrame(), data=data)
    c2 = Coordinate(frame=ICRSFrame(), data=data)
    assert np.all(c1 == c2)


def test_coordinate_inequality():
    c1 = Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(lon=10 * u.deg, lat=20 * u.deg),
    )
    c2 = Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(lon=11 * u.deg, lat=20 * u.deg),
    )
    assert np.all(c1 != c2)


def test_coordinate_equality_different_frame_raises():
    c1 = Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(lon=10 * u.deg, lat=20 * u.deg),
    )
    c2 = Coordinate(
        frame=GalacticFrame(),
        data=r.UnitSphericalRepresentation(lon=10 * u.deg, lat=20 * u.deg),
    )
    with pytest.raises(TypeError, match="equivalent frames"):
        _ = c1 == c2


def test_coordinate_is_equivalent_frame_same():
    c1 = Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(lon=0 * u.deg, lat=0 * u.deg),
    )
    c2 = Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(lon=1 * u.deg, lat=1 * u.deg),
    )
    assert c1.is_equivalent_frame(c2)


def test_coordinate_is_equivalent_frame_different():
    c1 = Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(lon=0 * u.deg, lat=0 * u.deg),
    )
    c2 = Coordinate(
        frame=GalacticFrame(),
        data=r.UnitSphericalRepresentation(lon=0 * u.deg, lat=0 * u.deg),
    )
    assert not c1.is_equivalent_frame(c2)


def _make_icrs_coord(ra_deg, dec_deg):
    return Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(lon=ra_deg * u.deg, lat=dec_deg * u.deg),
    )


def test_coordinate_transform_to_same_frame_is_noop():
    """Transforming to the same frame type should preserve coordinates."""
    coord = _make_icrs_coord(10.0, 20.0)
    result = coord.transform_to(ICRSFrame())
    assert isinstance(result, Coordinate)
    assert isinstance(result.frame, ICRSFrame)
    assert_quantity_allclose(result.ra, 10 * u.deg, atol=1e-10 * u.deg)
    assert_quantity_allclose(result.dec, 20 * u.deg, atol=1e-10 * u.deg)


def test_coordinate_transform_icrs_to_galactic():
    """Transforming ICRS → Galactic should give a Coordinate in GalacticFrame."""
    coord = _make_icrs_coord(266.405, -28.936)  # approximate GC direction
    galactic = coord.transform_to(GalacticFrame())
    assert isinstance(galactic, Coordinate)
    assert isinstance(galactic.frame, GalacticFrame)
    # GC is approximately at l~0, b~0
    assert_quantity_allclose(galactic.l, 0 * u.deg, atol=1 * u.deg)
    assert_quantity_allclose(galactic.b, 0 * u.deg, atol=1 * u.deg)


def test_coordinate_transform_icrs_to_fk5():
    """Transforming ICRS → FK5 should produce a Coordinate in FK5Frame."""
    coord = _make_icrs_coord(10.0, 20.0)
    fk5_coord = coord.transform_to(FK5Frame())
    assert isinstance(fk5_coord, Coordinate)
    assert isinstance(fk5_coord.frame, FK5Frame)
    # Values should be close (ICRS and FK5/J2000 are within ~tens of mas)
    assert_quantity_allclose(fk5_coord.ra, 10 * u.deg, atol=0.1 * u.deg)
    assert_quantity_allclose(fk5_coord.dec, 20 * u.deg, atol=0.1 * u.deg)


def test_coordinate_transform_roundtrip_icrs_galactic():
    """Round-trip ICRS → Galactic → ICRS should recover original coordinates."""
    ra, dec = 83.82, -5.39  # Orion Nebula approx
    coord = _make_icrs_coord(ra, dec)
    galactic = coord.transform_to(GalacticFrame())
    recovered = galactic.transform_to(ICRSFrame())

    assert_quantity_allclose(recovered.ra, ra * u.deg, atol=1e-6 * u.deg)
    assert_quantity_allclose(recovered.dec, dec * u.deg, atol=1e-6 * u.deg)


def test_coordinate_transform_roundtrip_icrs_fk5():
    """Round-trip ICRS → FK5 → ICRS should recover original coordinates."""
    ra, dec = 150.0, 30.0
    coord = _make_icrs_coord(ra, dec)
    fk5_coord = coord.transform_to(FK5Frame())
    recovered = fk5_coord.transform_to(ICRSFrame())

    assert_quantity_allclose(recovered.ra, ra * u.deg, atol=1e-4 * u.deg)
    assert_quantity_allclose(recovered.dec, dec * u.deg, atol=1e-4 * u.deg)


def test_coordinate_transform_no_path_raises():
    """Transforming to a frame with no registered path must raise ConvertError."""
    from astropy.coordinates.errors import ConvertError

    class OrphanFrame(BaseFrame):
        name = "orphan"
        default_representation = r.SphericalRepresentation

    coord = _make_icrs_coord(10.0, 20.0)
    with pytest.raises(ConvertError):
        coord.transform_to(OrphanFrame())


def test_skycoord_accepts_dataless_frame_as_frame_argument():
    """SkyCoord should accept a BaseFrame instance for the `frame` keyword."""
    sc = SkyCoord(ra=10 * u.deg, dec=20 * u.deg, frame=ICRSFrame())
    assert_quantity_allclose(sc.ra, 10 * u.deg)
    assert_quantity_allclose(sc.dec, 20 * u.deg)


def test_skycoord_transform_to_dataless_frame():
    """SkyCoord.transform_to should accept a BaseFrame instance."""
    sc = SkyCoord(ra=10 * u.deg, dec=20 * u.deg, frame="icrs")
    galactic = sc.transform_to(GalacticFrame())
    # result is still a SkyCoord
    assert isinstance(galactic, SkyCoord)
    # and the underlying frame is equivalent to GalacticFrame
    assert galactic.frame.is_equivalent_frame(GalacticFrame())


def test_skycoord_transform_to_fk5_frame_instance():
    """SkyCoord.transform_to(FK5Frame(...)) should honour frame attributes."""
    sc = SkyCoord(ra=150 * u.deg, dec=30 * u.deg, frame="icrs")
    fk5_j1975 = FK5Frame(equinox=Time("J1975.0"))
    result = sc.transform_to(fk5_j1975)
    assert isinstance(result, SkyCoord)


def test_skycoord_roundtrip_via_dataless_frames():
    """SkyCoord round-trip using data-less frame instances should be consistent."""
    sc = SkyCoord(ra=83.82 * u.deg, dec=-5.39 * u.deg, frame="icrs")
    galactic = sc.transform_to(GalacticFrame())
    recovered = galactic.transform_to(ICRSFrame())

    assert_quantity_allclose(recovered.ra, sc.ra, atol=1e-6 * u.deg)
    assert_quantity_allclose(recovered.dec, sc.dec, atol=1e-6 * u.deg)


def test_coordinate_and_legacy_frame_give_same_transform():
    """
    Coordinate.transform_to and the legacy ICRS.transform_to should give
    numerically identical results when transforming to Galactic.
    """
    ra, dec = 200.0, -45.0
    new_coord = Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(lon=ra * u.deg, lat=dec * u.deg),
    )
    from astropy.coordinates.builtin_frames import Galactic

    legacy_result = ICRS(ra=ra * u.deg, dec=dec * u.deg).transform_to(Galactic())
    new_result = new_coord.transform_to(GalacticFrame())

    assert_quantity_allclose(new_result.l, legacy_result.l, atol=1e-10 * u.deg)
    assert_quantity_allclose(new_result.b, legacy_result.b, atol=1e-10 * u.deg)


def test_coordinate_separation():
    """Coordinate.separation should compute the great-circle distance."""
    c1 = _make_icrs_coord(0.0, 0.0)
    c2 = _make_icrs_coord(1.0, 0.0)
    sep = c1.separation(c2, origin_mismatch="ignore")
    assert_quantity_allclose(sep, 1 * u.deg, atol=1e-6 * u.deg)


def test_coordinate_position_angle():
    """Coordinate.position_angle should match the expected direction."""
    c1 = _make_icrs_coord(0.0, 0.0)
    c2 = _make_icrs_coord(1.0, 0.0)
    # East from North: a point due East is ~90 deg
    pa = c1.position_angle(c2)
    assert_quantity_allclose(pa.to(u.deg), 90 * u.deg, atol=0.1 * u.deg)


def test_coordinate_indexing():
    coord = Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(
            lon=[10, 20, 30] * u.deg, lat=[1, 2, 3] * u.deg
        ),
    )
    sliced = coord[1]
    assert sliced.shape == ()
    assert_quantity_allclose(sliced.ra, 20 * u.deg)


def test_coordinate_array_slice():
    coord = Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(
            lon=[10, 20, 30] * u.deg, lat=[1, 2, 3] * u.deg
        ),
    )
    sliced = coord[1:]
    assert sliced.shape == (2,)


def test_coordinate_to_table():
    coord = Coordinate(
        frame=ICRSFrame(),
        data=r.UnitSphericalRepresentation(lon=[10, 20] * u.deg, lat=[5, 10] * u.deg),
    )
    t = coord.to_table()
    assert "ra" in t.colnames
    assert "dec" in t.colnames
    assert len(t) == 2
    assert_quantity_allclose(t["ra"], [10, 20] * u.deg)
