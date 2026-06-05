import pygame
import math


class MultiCarRenderer:
    """Renderer for multi-car racing environments."""

    def __init__(self, multi_env, fps: int = 60):
        self.env = multi_env
        self.width = getattr(
            multi_env.track, "track_width_px", multi_env.track.track_width
        )
        self.height = getattr(
            multi_env.track, "track_height_px", multi_env.track.track_width
        )

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RL Multi-Car Racing - Neural Tesla")
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.font = pygame.font.SysFont(None, 20)
        self.font_small = pygame.font.SysFont(None, 14)

    def render(self, done: bool = False) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False

        self.screen.fill((30, 30, 40))

        # Track
        outer, inner = self.env.track.get_boundaries()
        pygame.draw.polygon(self.screen, (180, 180, 190), outer)
        pygame.draw.polygon(self.screen, (30, 30, 40), inner)

        # Centerline
        center_pts = self.env.track.center_points
        if len(center_pts) > 2:
            pygame.draw.lines(self.screen, (80, 80, 90), True, center_pts, 1)

        # Obstacles
        for ox, oy, radius in self.env.track.get_obstacles():
            pygame.draw.circle(
                self.screen, (200, 80, 80), (int(ox), int(oy)), int(radius)
            )
            pygame.draw.circle(
                self.screen, (255, 100, 100), (int(ox), int(oy)), int(radius), 2
            )

        # Render each car
        for i, car in enumerate(self.env.cars):
            # Car body
            car_corners = car.get_corners()
            pygame.draw.polygon(self.screen, car.color, car_corners)

            # Direction indicator
            car_x, car_y = car.x, car.y
            heading = car.heading
            front_x = car_x + math.cos(heading) * 15
            front_y = car_y + math.sin(heading) * 15
            pygame.draw.line(
                self.screen, (255, 255, 255), (car_x, car_y), (front_x, front_y), 2
            )

            # Sensor rays (only for first car to avoid clutter)
            if i == 0:
                readings = self.env.sensors_list[i].get_readings(car, self.env.track)
                sensor_max = self.env.sensors_list[i].max_distance
                for j, angle in enumerate(self.env.sensors_list[i].angles):
                    ray_h = heading + angle
                    dist = readings[j] * sensor_max
                    ex = car_x + math.cos(ray_h) * dist
                    ey = car_y + math.sin(ray_h) * dist
                    if readings[j] < 0.3:
                        color = (255, 60, 60)
                    elif readings[j] < 0.6:
                        color = (255, 200, 40)
                    else:
                        color = (60, 220, 60)
                    pygame.draw.line(
                        self.screen, color, (car_x, car_y), (ex, ey), 1
                    )

        # HUD - left side
        y_offset = 10
        for i, car in enumerate(self.env.cars):
            txt = self.font_small.render(
                f"{car.name}: {car.velocity:.1f} px/s  Laps: {self.env.lap_counts[i]}",
                True, car.color,
            )
            self.screen.blit(txt, (10, y_offset))
            y_offset += 16

        # HUD - step counter
        step_txt = self.font_small.render(
            f"Step: {self.env.current_step}/{self.env.max_steps}",
            True, (200, 200, 200),
        )
        self.screen.blit(step_txt, (10, self.height - 18))

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def close(self) -> None:
        pygame.quit()
