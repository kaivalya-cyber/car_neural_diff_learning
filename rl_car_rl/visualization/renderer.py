import pygame
import math
from env.environment import CarEnv

class Renderer:
    def __init__(self, env: CarEnv, fps=60):
        self.env = env
        self.width = env.track.track_width
        self.height = env.track.track_height
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RL Car Simulation")
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.font = pygame.font.SysFont(None, 24)

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
                
        self.screen.fill((50, 50, 50))
        
        # Render track boundaries
        outer, inner = self.env.track.get_boundaries()
        pygame.draw.polygon(self.screen, (200, 200, 200), outer)
        pygame.draw.polygon(self.screen, (50, 50, 50), inner)
        
        # Render car position and orientation
        car_corners = self.env.car.get_corners()
        pygame.draw.polygon(self.screen, (255, 0, 0), car_corners)
        
        # Render sensor rays
        car_x = self.env.car.x
        car_y = self.env.car.y
        heading = self.env.car.heading
        
        readings = self.env.sensors.get_readings(self.env.car, self.env.track)
        for i, angle in enumerate(self.env.sensors.angles):
            ray_heading = heading + angle
            dist = readings[i] * self.env.sensors.max_distance
            end_x = car_x + math.cos(ray_heading) * dist
            end_y = car_y + math.sin(ray_heading) * dist
            
            pygame.draw.line(self.screen, (0, 255, 0), (car_x, car_y), (end_x, end_y), 1)
            pygame.draw.circle(self.screen, (0, 0, 255), (int(end_x), int(end_y)), 3)
            
        # Display debugging info
        info_text = f"Spd: {self.env.car.velocity:.1f} | Step: {self.env.current_step} | Steer: {self.env.car.steering_angle:.2f}"
        text_surface = self.font.render(info_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def close(self):
        pygame.quit()
