"""A bridge part of a few metres at a junction is not an abutment of the chain it hangs on (2026-11-13, Sofia:
Botevgradsko shosse and Madrid, Gotse Delchev and Tsar Boris III, Penchov Slaveykov): its street was graded 8 m up to
clear a road under the chain's other parts."""
import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import grading  # noqa: E402
import roads  # noqa: E402


def line(*pts):
    return np.array(pts, np.float64)


def flat(p):
    return np.zeros(len(np.asarray(p).reshape(-1, 2)))


class StubLiftTest(unittest.TestCase):
    """Two spans and a 5 m stub at their joint, a road crossing under the second span 20 m from the joint."""

    def setUp(self):
        self.a = line((-40.0, 0.0), (0.0, 0.0))
        self.b = line((0.0, 0.0), (40.0, 0.0))
        self.stub = line((0.0, 0.0), (0.0, -5.0))
        self.road = [line((20.0, -30.0), (20.0, 30.0))]

    def lifts(self, line_, others):
        bridges = [self.a, self.b, self.stub]
        return grading.bridge_lifts(line_, bridges, self.road, flat, chain=grading.bridge_chain(line_, bridges))

    def test_a_stub_at_a_junction_is_not_an_abutment(self):
        self.assertEqual(self.lifts(self.stub, None), [])

    def test_it_was_lifted_for_the_roads_under_the_chain(self):
        old, grading.STUB_ON = grading.STUB_ON, False
        try:
            got = self.lifts(self.stub, None)
        finally:
            grading.STUB_ON = old
        self.assertEqual(len(got), 1)
        self.assertGreater(got[0][1], 5.0, "the street was raised to the deck's clearance within 5 m of its junction")

    def test_the_spans_keep_their_abutments(self):
        got = self.lifts(self.b, None)
        self.assertEqual(len(got), 1, "the far end of the span over the road is raised as before")
        self.assertAlmostEqual(float(got[0][0][0]), 40.0)

    def test_a_short_part_at_the_end_of_a_simple_chain_is_an_abutment(self):
        # (the deck carried a few metres onto the street joined at its end, roads._skew_decks: GitHub #306; one part joined)
        tail = line((40.0, 0.0), (40.0, -5.0))          # (a 5 m part joined to b's end only)
        got = grading.bridge_lifts(tail, [self.b, tail], self.road, flat, chain=grading.bridge_chain(tail, [self.b, tail]))
        self.assertEqual(len(got), 1)
        self.assertGreater(got[0][1], 5.0)


if __name__ == "__main__":
    unittest.main()
