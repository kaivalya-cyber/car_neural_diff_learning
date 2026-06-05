import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.car import Car
from env.physics import PhysicsEngine


class TestPhysics(unittest.TestCase):
    def setUp(self):
        self.car = Car(init_x=400, init_y=300, init_heading=0)
        self.physics = PhysicsEngine()

    def test_acceleration(self):
        self.car.velocity = 0.0
        self.physics.update(self.car, 0.0, 1.0, dt=0.1)
        self.assertGreater(self.car.velocity, 0.0)

    def test_steering_turns_car(self):
        self.car.velocity = 10.0
        old_heading = self.car.heading
        self.physics.update(self.car, 1.0, 0.5, dt=0.1)
        self.assertNotEqual(self.car.heading, old_heading)

    def test_max_speed_clamped(self):
        self.car.velocity = self.physics.max_speed + 10
        self.physics.update(self.car, 0.0, 0.0, dt=0.1)
        self.assertLessEqual(self.car.velocity, self.physics.max_speed)

    def test_drag_slows_car(self):
        self.car.velocity = 30.0
        self.physics.update(self.car, 0.0, 0.0, dt=0.1)
        self.assertLess(self.car.velocity, 30.0)

    def test_position_updates(self):
        old_x, old_y = self.car.x, self.car.y
        self.car.velocity = 10.0
        self.physics.update(self.car, 0.0, 1.0, dt=0.1)
        # With heading=0, only x changes (cos(0)=1, sin(0)=0)
        self.assertNotEqual(self.car.x, old_x)
        self.assertGreater(self.car.x, old_x)

    def test_no_movement_at_zero_speed_no_throttle(self):
        self.car.velocity = 0.0
        old_x, old_y = self.car.x, self.car.y
        self.physics.update(self.car, 0.0, 0.0, dt=0.1)
        # Position may still change slightly due to heading changes
        # but should be minimal
        self.assertAlmostEqual(self.car.x, old_x, delta=0.01)
        self.assertAlmostEqual(self.car.y, old_y, delta=0.01)


if __name__ == "__main__":
    unittest.main()
