import math
import pytest

from linguistic_analysis.semantics.similarity import SemGraph

@pytest.fixture()
def semgraph_1() -> SemGraph:
    nodes = ["a", "b", "c", "d"]
    res = SemGraph(nodes)
    res.add_edge_with_names("a", "b", 1 / 2)
    res.add_edge_with_names("a", "c", 1 / 2)
    res.add_edge_with_names("b", "c", 1 - (math.sqrt(2) / 2))
    yield res

@pytest.fixture()
def semgraph_2() -> SemGraph:
    nodes = ["a", "b", "c", "d"]
    res = SemGraph(nodes)
    res.add_edge_with_names("b", "c", 1 - (math.sqrt(2) / 2))
    res.add_edge_with_names("b", "d", 1 / 2)
    res.add_edge_with_names("c", "d", 1 / 2)
    yield res

def test_get_triangulation(semgraph_1: SemGraph, semgraph_2: SemGraph):
    t_1 = semgraph_1.get_triangulation()
    #    print(str(t_1))
    assert len(t_1.triangles) == 4
    assert t_1.triangles[0].name == "a_b_c"
    assert t_1.triangles[1].name == "a_b_d"
    assert t_1.triangles[2].name == "a_c_d"
    assert t_1.triangles[3].name == "b_c_d"
    t_2 = semgraph_2.get_triangulation()
    print(str(t_2))
    assert len(t_2.triangles) == 4
    assert t_2.triangles[0].name == "a_b_c"
    assert t_2.triangles[1].name == "a_b_d"
    assert t_2.triangles[2].name == "a_c_d"
    assert t_2.triangles[3].name == "b_c_d"
    d = t_1.get_angle_distance(t_2)
    print(d)




