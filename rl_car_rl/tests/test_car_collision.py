import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.car import Car, check_rect_collision


class TestCarCollision(unittest.TestCase):
    def test_no_collision_separated(self):
        car1 = Car(init_x=100, init_y=100, init_heading=0)
        car2 = Car(init_x=200, init_y=100, init_heading=0)
        self.assertFalse(car1.collides_with(car2))

    def test_collision_overlapping(self):
        car1 = Car(init_x=100, init_y=100, init_heading=0)
        car2 = Car(init_x=110, init_y=100, init_heading=0)
        self.assertTrue(car1.collides_with(car2))

    def test_sat_separated_rects(self):
        rect1 = [(0, 0), (10, 0), (10, 10), (0, 10)]
        rect2 = [(20, 0), (30, 0), (30, 10), (20, 10)]
        self.assertFalse(check_rect_collision(rect1, rect2))

    def test_sat_overlapping_rects(self):
        rect1 = [(0, 0), (10, 0), (10, 10), (0, 10)]
        rect2 = [(5, 5), (15, 5), (15, 15), (5, 15)]
        self.assertTrue(check_rect_collision(rect1, rect2))

    def test_collision_rotated(self):
        car1 = Car(init_x=100, init_y=100, init_heading=0.5)
        car2 = Car(init_x=100, init_y=100, init_heading=0.5)
        self.assertTrue(car1.collides_with(car2))

    def test_collision_perpendicular(self):
        # Two cars perpendicular and overlapping
        car1 = Car(init_x=100, init_y=100, init_heading=0)
        car2 = Car(init_x=110, init_y=110, init_heading=1.57)
        self.assertTrue(car1.collides_with(car2))


if __name__ == "__main__":
    unittest.main()
