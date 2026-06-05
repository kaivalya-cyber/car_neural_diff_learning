import pygame
import math
from env.environment import CarEnv


class Renderer:
    def __init__(self, env: CarEnv, fps: int = 60):
        self.env = env
        self.width = getattr(env.track, "track_width_px", env.track.track_width)
        self.height = getattr(env.track, "track_height_px", env.track.track_width)

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RL Car Simulation - Neural Tesla")
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.font = pygame.font.SysFont(None, 22)
        self.font_small = pygame.font.SysFont(None, 16)
        self.episode_reward = 0.0

    def render(self, reward: float = 0.0, done: bool = False) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False

        self.screen.fill((30, 30, 40))

        # Render track boundaries
        outer, inner = self.env.track.get_boundaries()
        pygame.draw.polygon(self.screen, (180, 180, 190), outer)
        pygame.draw.polygon(self.screen, (30, 30, 40), inner)

        # Render centerline (subtle)
        center_pts = self.env.track.center_points
        if len(center_pts) > 2:
            pygame.draw.lines(self.screen, (80, 80, 90), True, center_pts, 1)

        # Render obstacles
        for ox, oy, radius in self.env.track.get_obstacles():
            pygame.draw.circle(self.screen, (200, 80, 80), (int(ox), int(oy)), int(radius))
            pygame.draw.circle(self.screen, (255, 100, 100), (int(ox), int(oy)), int(radius), 2)

        # Render car
        car_corners = self.env.car.get_corners()
        pygame.draw.polygon(self.screen, (0, 180, 0), car_corners)
        # Car direction indicator
        car_x, car_y = self.env.car.x, self.env.car.y
        heading = self.env.car.heading
        front_x = car_x + math.cos(heading) * 15
        front_y = car_y + math.sin(heading) * 15
        pygame.draw.line(self.screen, (255, 255, 255), (car_x, car_y), (front_x, front_y), 2)

        # Render sensor rays with color coding
        readings = self.env.sensors.get_readings(self.env.car, self.env.track)
        sensor_max = self.env.sensors.max_distance

        for i, angle in enumerate(self.env.sensors.angles):
            ray_heading = heading + angle
            dist = readings[i] * sensor_max
            end_x = car_x + math.cos(ray_heading) * dist
            end_y = car_y + math.sin(ray_heading) * dist

            # Color code: red=near (<0.3), yellow=medium (<0.6), green=far
            if readings[i] < 0.3:
                color = (255, 60, 60)
            elif readings[i] < 0.6:
                color = (255, 200, 40)
            else:
                color = (60, 220, 60)

            pygame.draw.line(
                self.screen, color, (car_x, car_y), (end_x, end_y), 1
            )
            # Smaller endpoint dots
            if readings[i] < 0.8:
                pygame.draw.circle(
                    self.screen, color, (int(end_x), int(end_y)), 2
                )

        # Display HUD info
        y_offset = 10
        lines = [
            f"Speed: {self.env.car.velocity:.1f}",
            f"Steer: {self.env.car.steering_angle:.2f}",
            f"Step: {self.env.current_step}/{self.env.max_steps}",
        ]
        for line in lines:
            txt = self.font_small.render(line, True, (220, 220, 220))
            self.screen.blit(txt, (10, y_offset))
            y_offset += 18

        # Right side HUD
        y_offset = 10
        self.episode_reward += reward
        if done:
            self.episode_reward = 0.0
        right_lines = [
            f"Laps: {self.env.lap_count}",
            f"Reward: {self.episode_reward:.1f}",
            f"Obstacles: {len(self.env.track.obstacles)}",
        ]
        for line in right_lines:
            txt = self.font_small.render(line, True, (220, 220, 220))
            tw = txt.get_width()
            self.screen.blit(txt, (self.width - tw - 10, y_offset))
            y_offset += 18

        # Track width indicator
        tw_txt = self.font_small.render(
            f"Track: {self.env.track.track_width:.0f}px",
            True, (180, 180, 180),
        )
        self.screen.blit(tw_txt, (10, self.height - 20))

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def get_frame(self) -> "np.ndarray | None":
        """Capture the current screen as an RGB numpy array for video recording."""
        try:
            import numpy as np
            data = pygame.surfarray.array3d(self.screen)
            return np.transpose(data, (1, 0, 2))  # (width, height, 3) -> (height, width, 3)
        except (ImportError, pygame.error) as e:
            print(f"Frame capture failed: {e}")
            return None

    def close(self) -> None:
        pygame.quit()
