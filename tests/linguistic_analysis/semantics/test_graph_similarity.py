import numpy as np
import math
import pytest

from linguistic_analysis.semantics.similarity import Triangle

@pytest.fixture()
def triangle_iso():
    yield Triangle(1, 1, 1, "a", "b", "c")

@pytest.fixture()
def triangle_sq():
    # ba: 1
    # ac: 1
    # bc: sqrt(2)
    yield Triangle(1, 1, math.sqrt(2), "b", "a", "c")

def test_triangle_angles(triangle_iso: Triangle, triangle_sq: Triangle):
    assert round(triangle_iso.cos_a, 5) == round(math.cos(math.pi/3), 5)
    assert round(triangle_iso.cos_b, 5) == round(math.cos(math.pi/3), 5)
    assert round(triangle_iso.cos_c, 5) == round(math.cos(math.pi/3), 5)
    assert round(triangle_sq.cos_a, 5) == round(math.cos(math.pi/4), 5)
    assert round(triangle_sq.cos_b, 5) == round(math.cos(math.pi/2), 5)
    assert round(triangle_sq.cos_c, 5) == round(math.cos(math.pi/4), 5)

def test_triangle_name(triangle_iso: Triangle, triangle_sq: Triangle):
    assert triangle_iso.name == "a_b_c"
    assert triangle_sq.name == "a_b_c"

def test_sorted_vnames(triangle_iso: Triangle, triangle_sq: Triangle):
    assert triangle_iso.sorted_vnames == ["a", "b", "c"]
    assert triangle_sq.sorted_vnames == ["a", "b", "c"]

def test_get_vertex_distance(triangle_iso: Triangle, triangle_sq: Triangle):
    assert triangle_iso.get_vertex_distance("a", "b") == 1
    assert triangle_iso.get_vertex_distance("b", "c") == 1
    assert triangle_iso.get_vertex_distance("a", "c") == 1
    assert triangle_sq.get_vertex_distance("a", "b") == 1
    assert triangle_sq.get_vertex_distance("b", "c") == math.sqrt(2)
    assert triangle_sq.get_vertex_distance("a", "c") == 1

def test_get_angle_distance(triangle_iso: Triangle, triangle_sq: Triangle):
    assert round(triangle_iso.get_angle_distance(triangle_sq), 5) == \
           round(triangle_sq.get_angle_distance(triangle_iso), 5)
    assert round(triangle_iso.get_angle_distance(triangle_sq), 5) == \
           round(np.linalg.norm(np.array([math.cos(math.pi/3), math.cos(math.pi/3), math.cos(math.pi/3)]) -
                          np.array([math.cos(math.pi/4), math.cos(math.pi/2), math.cos(math.pi/4)])), 5)
