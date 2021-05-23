import numpy as np
import math
import pytest

from linguistic_analysis.semantics.similarity import Triangle, NormalizationType

@pytest.fixture()
def triangle_iso():
    yield Triangle(1, 1, 1, "a", "b", "c")

@pytest.fixture()
def triangle_sq():
    # ba: 1
    # ac: 1
    # bc: sqrt(2)
    yield Triangle(1/2, 1/2, math.sqrt(2)/2, "b", "a", "c")

@pytest.fixture()
def triangle_3():
    # ab: 0.4143933653831482
    # ac: 0.49820971488952637
    # bc: 1
    #Triangle[name: acquisition_appointement_crédit, acquisition-appointement: 0.4143933653831482, appointement-crédit: 0.49820971488952637 ,acquisition-crédit: 1.0; cos_acquisition: 1.1142902111975075, cos_appointement: -1.4048237617557404, cos_crédit:1.0803593612937017]
    yield Triangle(0.4143933653831482, 0.49820971488952637, 1, "a", "b", "c")

@pytest.fixture()
def triangle_distance_normalized_1():
    yield Triangle(1, 1, 1, "a", "b", "c", normalization=NormalizationType.NORM_DISTANCE)

@pytest.fixture()
def triangle_distance_normalized_2():
    yield Triangle(1, 1, 0.5, "a", "b", "c", normalization=NormalizationType.NORM_DISTANCE)

@pytest.fixture()
def triangle_distance_normalized_3():
    yield Triangle(1, 0.6, 0.6, "a", "b", "c", normalization=NormalizationType.NORM_DISTANCE)


def test_triangle_angles(triangle_iso: Triangle, triangle_sq: Triangle):
    assert round(triangle_iso.cos_a, 5) == round(math.cos(math.pi/3), 5)
    assert round(triangle_iso.alpha, 5) == round(math.pi / 3, 5)
    assert round(triangle_iso.cos_b, 5) == round(math.cos(math.pi/3), 5)
    assert round(triangle_iso.beta, 5) == round(math.pi/3, 5)
    assert round(triangle_iso.cos_c, 5) == round(math.cos(math.pi/3), 5)
    assert round(triangle_iso.gamma, 5) == round(math.pi/3, 5)
    assert round(triangle_sq.cos_a, 5) == round(math.cos(math.pi/4), 5)
    assert round(triangle_sq.alpha, 5) == round(math.pi/4, 5)
    assert round(triangle_sq.cos_b, 5) == round(math.cos(math.pi/2), 5)
    assert round(triangle_sq.beta, 5) == round(math.pi/2, 5)
    assert round(triangle_sq.cos_c, 5) == round(math.cos(math.pi/4), 5)
    assert round(triangle_sq.gamma, 5) == round(math.pi/4, 5)


def test_triangle_angles_distance_normalized(triangle_distance_normalized_1: Triangle,
                                             triangle_distance_normalized_2: Triangle,
                                             triangle_distance_normalized_3: Triangle):
    assert round(triangle_distance_normalized_1.cos_a, 5) == 1
    assert round(triangle_distance_normalized_1.alpha, 5) == 0
    assert round(triangle_distance_normalized_1.cos_b, 5) == 1
    assert round(triangle_distance_normalized_1.beta, 5) == 0
    assert round(triangle_distance_normalized_1.cos_c, 5) == 1
    assert round(triangle_distance_normalized_1.gamma, 5) == 0

    assert round(triangle_distance_normalized_2.cos_a, 5) == 0
    assert round(triangle_distance_normalized_2.alpha, 5) == round(math.pi/2, 5)
    assert round(triangle_distance_normalized_2.cos_b, 5) == 1
    assert round(triangle_distance_normalized_2.beta, 5) == 0
    assert round(triangle_distance_normalized_2.cos_c, 5) == 0
    assert round(triangle_distance_normalized_2.gamma, 5) == round(math.pi/2, 5)

    assert round(triangle_distance_normalized_3.cos_a, 5) == 1
    assert round(triangle_distance_normalized_3.alpha, 5) == 0
    assert round(triangle_distance_normalized_3.cos_b, 5) == 1
    assert round(triangle_distance_normalized_3.beta, 5) == 0
    assert round(triangle_distance_normalized_3.cos_c, 5) == -1
    assert round(triangle_distance_normalized_3.gamma, 5) == round(math.pi, 5)


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
    assert triangle_sq.get_vertex_distance("a", "b") == 1/2
    assert triangle_sq.get_vertex_distance("b", "c") == math.sqrt(2)/2
    assert triangle_sq.get_vertex_distance("a", "c") == 1/2


def test_get_vertex_distance_distance_normalized(triangle_distance_normalized_1: Triangle,
                                                 triangle_distance_normalized_2: Triangle,
                                                 triangle_distance_normalized_3: Triangle):
    assert triangle_distance_normalized_1.get_vertex_distance("a", "b") == 1
    assert triangle_distance_normalized_1.get_vertex_distance("b", "c") == 1
    assert triangle_distance_normalized_1.get_vertex_distance("a", "c") == 1

    assert triangle_distance_normalized_2.get_vertex_distance("a", "b") == 1
    assert triangle_distance_normalized_2.get_vertex_distance("b", "c") == 1
    assert triangle_distance_normalized_2.get_vertex_distance("a", "c") == 0.5

    assert triangle_distance_normalized_3.get_vertex_distance("a", "b") == 1
    assert triangle_distance_normalized_3.get_vertex_distance("b", "c") == 0.6
    assert triangle_distance_normalized_3.get_vertex_distance("a", "c") == 0.6


def test_get_angle_distance(triangle_iso: Triangle, triangle_sq: Triangle):
    assert round(triangle_iso.get_angle_distance(triangle_sq), 5) == \
           round(triangle_sq.get_angle_distance(triangle_iso), 5)
    assert round(triangle_iso.get_angle_distance(triangle_sq, rad=True), 5) == \
           round(triangle_sq.get_angle_distance(triangle_iso, rad=True), 5)
    assert round(triangle_iso.get_angle_distance(triangle_sq), 5) == \
           round(np.linalg.norm(np.array([math.cos(math.pi/3), math.cos(math.pi/3), math.cos(math.pi/3)]) -
                          np.array([math.cos(math.pi/4), math.cos(math.pi/2), math.cos(math.pi/4)])), 5)
    assert round(triangle_iso.get_angle_distance(triangle_sq, rad=True), 5) == \
           round(np.linalg.norm(np.array([math.pi/3, math.pi/3, math.pi/3]) -
                          np.array([math.pi/4, math.pi/2, math.pi/4])), 5)

def test_triangle_3(triangle_3):
    print("*********** " + str(triangle_3))
