"""Small artificial tests for the Step 2 interaction rules."""

import math
import unittest

from traffic_workload import (
    analyze_frame,
    distance,
    is_critical_interaction,
    relative_velocity,
    same_frame,
    same_or_adjacent_lane,
    simplified_ttc,
    within_interaction_radius,
)


def vehicle(vehicle_id, frame_id, lane_id, x, y, velocity):
    return {
        "Vehicle_ID": vehicle_id,
        "Frame_ID": frame_id,
        "Global_Time": 1000,
        "Lane_ID": lane_id,
        "Local_X": x,
        "Local_Y": y,
        "v_Vel": velocity,
        "v_Acc": 0.0,
    }


class TrafficWorkloadTests(unittest.TestCase):
    def test_distance(self):
        self.assertEqual(distance(0.0, 0.0, 3.0, 4.0), 5.0)

    def test_relative_velocity(self):
        self.assertEqual(relative_velocity(12.0, 7.0), 5.0)

    def test_simplified_ttc_and_zero_division_avoidance(self):
        self.assertEqual(simplified_ttc(12.0, -4.0), 3.0)
        self.assertTrue(math.isinf(simplified_ttc(12.0, 0.0)))
        self.assertTrue(math.isinf(simplified_ttc(12.0, 4.0)))

    def test_same_frame_filter(self):
        self.assertTrue(same_frame(10, 10))
        self.assertFalse(same_frame(10, 11))
        _, interactions = analyze_frame(
            [vehicle(1, 10, 1, 0.0, 0.0, 10.0), vehicle(2, 11, 1, 1.0, 0.0, 14.0)]
        )
        self.assertEqual(interactions, [])

    def test_same_or_adjacent_lane_filter(self):
        self.assertTrue(same_or_adjacent_lane(2, 3))
        self.assertFalse(same_or_adjacent_lane(1, 3))
        _, adjacent_lane_interactions = analyze_frame(
            [vehicle(1, 10, 2, 0.0, 0.0, 10.0), vehicle(2, 10, 3, 3.0, 4.0, 15.0)]
        )
        _, distant_lane_interactions = analyze_frame(
            [vehicle(1, 10, 1, 0.0, 0.0, 10.0), vehicle(2, 10, 3, 3.0, 4.0, 15.0)]
        )
        self.assertEqual(len(adjacent_lane_interactions), 1)
        self.assertEqual(distant_lane_interactions, [])

    def test_interaction_radius_filter(self):
        self.assertTrue(within_interaction_radius(50.0))
        self.assertFalse(within_interaction_radius(50.01))
        _, interactions = analyze_frame(
            [vehicle(1, 10, 1, 0.0, 0.0, 10.0), vehicle(2, 10, 1, 51.0, 0.0, 14.0)]
        )
        self.assertEqual(interactions, [])

    def test_pairs_are_not_duplicated(self):
        repeated_vehicle = vehicle(1, 10, 1, 0.0, 0.0, 10.0)
        candidate_pairs, interactions = analyze_frame(
            [
                repeated_vehicle,
                dict(repeated_vehicle),
                vehicle(2, 10, 1, 8.0, 0.0, 14.0),
            ]
        )
        pair_keys = {
            (item["Frame_ID"], item["Vehicle_ID_1"], item["Vehicle_ID_2"])
            for item in interactions
        }
        self.assertEqual(candidate_pairs, 1)
        self.assertEqual(len(interactions), 1)
        self.assertEqual(len(pair_keys), len(interactions))

    def test_critical_interaction_detection(self):
        self.assertTrue(is_critical_interaction(2.0))
        self.assertFalse(is_critical_interaction(3.1))
        self.assertFalse(is_critical_interaction(math.inf))
        _, interactions = analyze_frame(
            [vehicle(1, 10, 1, 0.0, 0.0, 10.0), vehicle(2, 10, 1, 8.0, 0.0, 14.0)]
        )
        self.assertTrue(interactions[0]["Critical_Interaction"])
        self.assertEqual(interactions[0]["TTC"], 2.0)


if __name__ == "__main__":
    unittest.main()