import pygame
import cv2
import random
import sys
import os
import numpy as np
from fractions import Fraction
import mediapipe as mp

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

class FractionEggGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Fraction Egg Catching Game - Head Control Pan")
        self.clock = pygame.time.Clock()
        
        # Initialize camera and MediaPipe
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        
        # Game state
        self.current_level = 1
        self.max_level = 5
        self.health = 3
        self.score = 0
        self.game_over = False
        self.game_started = False
        
        # Pan properties (controlled by face)
        self.pan_x = SCREEN_WIDTH // 2
        self.pan_y = SCREEN_HEIGHT - 150
        self.pan_width = 100
        self.pan_height = 80
        self.face_detected = False
        
        # Egg properties
        self.eggs = []
        self.broken_eggs = []  # For ground animation
        self.egg_spawn_timer = 0
        self.egg_spawn_delay = 90
        self.egg_speed = 4
        
        # Load assets
        self.load_assets()
        
        # Level definitions - REAL MATH CHALLENGES!
        self.level_data = {
            1: {
                'target': Fraction(1, 3),
                'correct_choices': [
                    Fraction(2, 6), Fraction(4, 12), Fraction(5, 15), Fraction(7, 21),
                    Fraction(8, 24), Fraction(9, 27), Fraction(11, 33), Fraction(13, 39),
                    Fraction(14, 42), Fraction(16, 48), Fraction(17, 51), Fraction(19, 57)
                ],
                'wrong_choices': [
                    Fraction(1, 4), Fraction(2, 5), Fraction(3, 8), Fraction(4, 9),
                    Fraction(5, 12), Fraction(7, 18), Fraction(8, 19), Fraction(9, 22),
                    Fraction(11, 25), Fraction(13, 31), Fraction(15, 37), Fraction(17, 41)
                ]
            },
            2: {
                'target': Fraction(2, 5),
                'correct_choices': [
                    Fraction(4, 10), Fraction(6, 15), Fraction(8, 20), Fraction(10, 25),
                    Fraction(12, 30), Fraction(14, 35), Fraction(16, 40), Fraction(18, 45),
                    Fraction(20, 50), Fraction(22, 55), Fraction(24, 60), Fraction(26, 65)
                ],
                'wrong_choices': [
                    Fraction(3, 7), Fraction(5, 11), Fraction(7, 16), Fraction(9, 20),
                    Fraction(11, 24), Fraction(13, 28), Fraction(15, 32), Fraction(17, 36),
                    Fraction(19, 41), Fraction(21, 45), Fraction(23, 49), Fraction(25, 53)
                ]
            },
            3: {
                'target': Fraction(3, 4),
                'correct_choices': [
                    Fraction(6, 8), Fraction(9, 12), Fraction(12, 16), Fraction(15, 20),
                    Fraction(18, 24), Fraction(21, 28), Fraction(24, 32), Fraction(27, 36),
                    Fraction(30, 40), Fraction(33, 44), Fraction(36, 48), Fraction(39, 52)
                ],
                'wrong_choices': [
                    Fraction(5, 7), Fraction(7, 10), Fraction(8, 11), Fraction(10, 13),
                    Fraction(11, 15), Fraction(13, 17), Fraction(14, 19), Fraction(16, 21),
                    Fraction(17, 23), Fraction(19, 25), Fraction(20, 27), Fraction(22, 29)
                ]
            },
            4: {
                'target': Fraction(2, 3),
                'correct_choices': [
                    Fraction(4, 6), Fraction(6, 9), Fraction(8, 12), Fraction(10, 15),
                    Fraction(12, 18), Fraction(14, 21), Fraction(16, 24), Fraction(18, 27),
                    Fraction(20, 30), Fraction(22, 33), Fraction(24, 36), Fraction(26, 39)
                ],
                'wrong_choices': [
                    Fraction(3, 5), Fraction(5, 8), Fraction(7, 11), Fraction(9, 14),
                    Fraction(11, 17), Fraction(13, 20), Fraction(15, 23), Fraction(17, 26),
                    Fraction(19, 29), Fraction(21, 32), Fraction(23, 35), Fraction(25, 38)
                ]
            },
            5: {
                'target': Fraction(4, 5),
                'correct_choices': [
                    Fraction(8, 10), Fraction(12, 15), Fraction(16, 20), Fraction(20, 25),
                    Fraction(24, 30), Fraction(28, 35), Fraction(32, 40), Fraction(36, 45),
                    Fraction(40, 50), Fraction(44, 55), Fraction(48, 60), Fraction(52, 65)
                ],
                'wrong_choices': [
                    Fraction(7, 9), Fraction(9, 11), Fraction(11, 14), Fraction(13, 16),
                    Fraction(15, 19), Fraction(17, 21), Fraction(19, 24), Fraction(21, 26),
                    Fraction(23, 29), Fraction(25, 31), Fraction(27, 34), Fraction(29, 37)
                ]
            }
        }
        
        # Font
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
    def load_assets(self):
        """Load all game assets"""
        # Background
        try:
            self.background = pygame.image.load('backgroundgame.png').convert()
            self.background = pygame.transform.scale(self.background, (SCREEN_WIDTH, SCREEN_HEIGHT))
        except:
            print("Background image not found, creating gradient background")
            self.background = self.create_gradient_background()
        
        # Pan image (main player sprite)
        try:
            self.pan_image = pygame.image.load('pan.png').convert_alpha()
            self.pan_image = pygame.transform.scale(self.pan_image, (self.pan_width, self.pan_height))
            print("Pan image loaded successfully!")
        except:
            print("Pan image not found, creating fallback pan")
            self.pan_image = self.create_pan_sprite()
        
        # Egg sprites (4 stages: normal → cracking → more cracked → broken)
        self.egg_images = []
        for i in range(1, 5):
            try:
                egg_img = pygame.image.load(f'egg{i}.png').convert_alpha()
                egg_img = pygame.transform.scale(egg_img, (50, 60))
                self.egg_images.append(egg_img)
                print(f"Egg{i} loaded!")
            except:
                print(f"Egg{i} image not found, creating drawn egg")
                self.egg_images.append(self.create_egg_sprite(50, 60, i))
        
        # Heart sprites
        try:
            self.heart_full = pygame.image.load('heart_full.png').convert_alpha()
            self.heart_empty = pygame.image.load('heart_empty.png').convert_alpha()
            self.heart_full = pygame.transform.scale(self.heart_full, (35, 35))
            self.heart_empty = pygame.transform.scale(self.heart_empty, (35, 35))
        except:
            print("Heart images not found, creating drawn hearts")
            self.heart_full = self.create_heart_sprite(35, 35, True)
            self.heart_empty = self.create_heart_sprite(35, 35, False)
        
        # Sound effects
        try:
            pygame.mixer.init()
            self.background_music = pygame.mixer.Sound('background_music.wav')
            
            try:
                self.correct_sound = pygame.mixer.Sound('correct_sound.wav')
            except:
                self.correct_sound = self.create_correct_sound()
                
            try:
                self.wrong_sound = pygame.mixer.Sound('wrong_sound.wav')
            except:
                self.wrong_sound = self.create_wrong_sound()
            
            self.background_music.set_volume(0.2)
            self.music_playing = False
        except:
            print("Sound system not available")
            self.background_music = None
            self.correct_sound = None
            self.wrong_sound = None
    
    def create_pan_sprite(self):
        """Create a pan sprite as fallback"""
        surface = pygame.Surface((self.pan_width, self.pan_height), pygame.SRCALPHA)
        
        # Draw frying pan
        # Pan body (circle)
        pygame.draw.circle(surface, (50, 50, 50), (40, 50), 35, 4)
        pygame.draw.circle(surface, (80, 80, 80), (40, 50), 30)
        
        # Pan handle
        pygame.draw.rect(surface, (50, 50, 50), (75, 45, 20, 10))
        
        return surface
    
    def create_egg_sprite(self, width, height, stage):
        """Create egg sprite for different stages"""
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        if stage == 1:  # Normal egg
            pygame.draw.ellipse(surface, (255, 248, 220), (0, 0, width, height))
            pygame.draw.ellipse(surface, (139, 69, 19), (0, 0, width, height), 2)
        elif stage == 2:  # Small crack
            pygame.draw.ellipse(surface, (255, 248, 220), (0, 0, width, height))
            pygame.draw.ellipse(surface, (139, 69, 19), (0, 0, width, height), 2)
            pygame.draw.line(surface, (100, 100, 100), (width//3, height//3), (width//2, 2*height//3), 2)
        elif stage == 3:  # More cracks
            pygame.draw.ellipse(surface, (255, 248, 220), (0, 0, width, height))
            pygame.draw.ellipse(surface, (139, 69, 19), (0, 0, width, height), 2)
            pygame.draw.line(surface, (100, 100, 100), (width//3, height//3), (width//2, 2*height//3), 2)
            pygame.draw.line(surface, (100, 100, 100), (2*width//3, height//4), (width//3, 3*height//4), 2)
        else:  # stage == 4: Broken
            # Draw broken shell pieces
            pygame.draw.arc(surface, (255, 248, 220), (0, 0, width//2, height//2), 0, 3.14, 3)
            pygame.draw.arc(surface, (255, 248, 220), (width//2, height//2, width//2, height//2), 3.14, 6.28, 3)
            # Draw yolk
            pygame.draw.circle(surface, (255, 255, 0), (width//2, height//2), width//4)
            pygame.draw.circle(surface, (255, 200, 0), (width//2, height//2), width//5)
        
        return surface
    
    def create_heart_sprite(self, width, height, filled):
        """Create heart sprite"""
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        color = (255, 50, 50) if filled else (100, 100, 100)
        
        # Heart shape
        pygame.draw.circle(surface, color, (width//4, height//3), width//5)
        pygame.draw.circle(surface, color, (3*width//4, height//3), width//5)
        points = [(width//6, height//2), (width//2, 4*height//5), (5*width//6, height//2)]
        pygame.draw.polygon(surface, color, points)
        return surface
    
    def create_gradient_background(self):
        """Create gradient background"""
        background = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        for y in range(SCREEN_HEIGHT):
            color_value = int(100 + (155 * y / SCREEN_HEIGHT))
            color = (color_value//2, color_value//2, color_value)
            pygame.draw.line(background, color, (0, y), (SCREEN_WIDTH, y))
        return background
    
    def create_correct_sound(self):
        """Create correct sound effect"""
        try:
            sample_rate = 22050
            duration = 0.3
            frequency = 800
            frames = int(duration * sample_rate)
            arr = np.zeros(frames)
            for i in range(frames):
                arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate) * np.exp(-i / (sample_rate * 0.3))
            arr = (arr * 32767).astype(np.int16)
            return pygame.sndarray.make_sound(arr)
        except:
            return None
    
    def create_wrong_sound(self):
        """Create wrong sound effect"""
        try:
            sample_rate = 22050
            duration = 0.5
            frequency = 200
            frames = int(duration * sample_rate)
            arr = np.zeros(frames)
            for i in range(frames):
                arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate) * 0.5
            arr = (arr * 32767).astype(np.int16)
            return pygame.sndarray.make_sound(arr)
        except:
            return None
    
    def detect_face_and_control_pan(self):
        """Use face detection to control pan position"""
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            self.face_detected = True
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Get face center
            h, w, _ = frame.shape
            face_center_x = (bbox.xmin + bbox.width / 2) * w
            
            # Map face X position to pan X position (full screen range)
            self.pan_x = int((face_center_x / w) * SCREEN_WIDTH) - self.pan_width // 2
            
            # Keep pan within screen bounds
            self.pan_x = max(0, min(self.pan_x, SCREEN_WIDTH - self.pan_width))
        else:
            self.face_detected = False
    
    def start_background_music(self):
        """Start background music"""
        if self.background_music and not self.music_playing:
            self.background_music.play(-1)
            self.music_playing = True
    
    def create_egg(self):
        """Create falling egg with fraction"""
        level_info = self.level_data[self.current_level]
        
        # 70% correct, 30% wrong
        if random.random() < 0.7:
            fraction = random.choice(level_info['correct_choices'])
            is_correct = True
        else:
            fraction = random.choice(level_info['wrong_choices'])
            is_correct = False
        
        return {
            'x': random.randint(50, SCREEN_WIDTH - 100),
            'y': -60,
            'fraction': fraction,
            'is_correct': is_correct,
            'sprite_stage': 1,  # Always start with egg1.png
            'width': 50,
            'height': 60,
            'caught': False,
            'breaking': False,
            'break_timer': 0
        }
    
    def update_eggs(self):
        """Update eggs and handle ground breaking animation"""
        eggs_to_remove = []
        
        for i, egg in enumerate(self.eggs):
            if not egg['caught'] and not egg['breaking']:
                # Move egg down
                egg['y'] += self.egg_speed
                
                # Check collision with pan
                if self.check_collision_with_pan(egg):
                    egg['caught'] = True
                    if egg['is_correct']:
                        self.score += 10
                        if self.correct_sound:
                            self.correct_sound.play()
                        print(f"🎉 Correct! {egg['fraction']} = {self.level_data[self.current_level]['target']}")
                    else:
                        self.health -= 1
                        if self.wrong_sound:
                            self.wrong_sound.play()
                        print(f"❌ Wrong! {egg['fraction']} ≠ {self.level_data[self.current_level]['target']}")
                    eggs_to_remove.append(i)
                
                # Check if egg hits ground - START BREAKING ANIMATION
                elif egg['y'] >= SCREEN_HEIGHT - 80:
                    egg['breaking'] = True
                    egg['break_timer'] = 0
                    egg['y'] = SCREEN_HEIGHT - 80  # Keep at ground level
                    if egg['is_correct']:
                        self.health -= 1
                        print(f"💔 Missed correct answer: {egg['fraction']}")
            
            # Handle breaking animation
            elif egg['breaking']:
                egg['break_timer'] += 1
                
                # Change sprite stages during breaking animation
                if egg['break_timer'] == 15:  # egg2.png
                    egg['sprite_stage'] = 2
                elif egg['break_timer'] == 30:  # egg3.png
                    egg['sprite_stage'] = 3
                elif egg['break_timer'] == 45:  # egg4.png
                    egg['sprite_stage'] = 4
                elif egg['break_timer'] >= 90:  # Remove after full animation
                    eggs_to_remove.append(i)
        
        # Remove finished eggs
        for i in reversed(eggs_to_remove):
            del self.eggs[i]
    
    def check_collision_with_pan(self, egg):
        """Check if egg collides with pan"""
        pan_rect = pygame.Rect(self.pan_x, self.pan_y, self.pan_width, self.pan_height)
        egg_rect = pygame.Rect(egg['x'], egg['y'], egg['width'], egg['height'])
        return pan_rect.colliderect(egg_rect)
    
    def check_level_progression(self):
        """Check level progression"""
        target_score = self.current_level * 50
        if self.score >= target_score and self.current_level < self.max_level:
            self.current_level += 1
            self.egg_speed += 0.5
            self.egg_spawn_delay = max(40, self.egg_spawn_delay - 8)
            print(f"🎊 Level {self.current_level}! Target: {self.level_data[self.current_level]['target']}")
    
    def draw_ui(self):
        """Draw game UI"""
        # Target fraction
        target = self.level_data[self.current_level]['target']
        target_text = self.font_large.render(f"Level {self.current_level}: Catch = {target}", True, WHITE)
        target_rect = target_text.get_rect(center=(SCREEN_WIDTH//2, 50))
        
        pygame.draw.rect(self.screen, BLACK, target_rect.inflate(30, 20))
        pygame.draw.rect(self.screen, YELLOW, target_rect.inflate(30, 20), 3)
        self.screen.blit(target_text, target_rect)
        
        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(topleft=(20, 20))
        pygame.draw.rect(self.screen, BLACK, score_rect.inflate(10, 5))
        pygame.draw.rect(self.screen, WHITE, score_rect.inflate(10, 5), 2)
        self.screen.blit(score_text, score_rect)
        
        # Hearts
        for i in range(3):
            heart_x = SCREEN_WIDTH - 170 + (i * 45)
            heart_y = 20
            if i < self.health:
                self.screen.blit(self.heart_full, (heart_x, heart_y))
            else:
                self.screen.blit(self.heart_empty, (heart_x, heart_y))
        
        # Face detection status
        face_status = "Head Detected ✓" if self.face_detected else "Move your head to control pan!"
        face_color = GREEN if self.face_detected else RED
        face_text = self.font_small.render(face_status, True, face_color)
        face_rect = face_text.get_rect(topleft=(20, 70))
        pygame.draw.rect(self.screen, BLACK, face_rect.inflate(10, 5))
        self.screen.blit(face_text, face_rect)
        
        # Instructions
        if not self.game_started:
            instructions = [
                "🍳 HEAD-CONTROLLED PAN GAME!",
                "Move your HEAD left/right to control the pan",
                "Catch eggs with fractions EQUAL to the target!",
                "Watch eggs break with animation when they hit ground!",
                "Press SPACE to start"
            ]
            
            for i, instruction in enumerate(instructions):
                color = YELLOW if i == 0 else WHITE
                text = self.font_medium.render(instruction, True, color)
                text_rect = text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 80 + i*35))
                pygame.draw.rect(self.screen, BLACK, text_rect.inflate(20, 8))
                if i == 0:
                    pygame.draw.rect(self.screen, YELLOW, text_rect.inflate(20, 8), 2)
                self.screen.blit(text, text_rect)
    
    def draw_eggs(self):
        """Draw all eggs with fractions"""
        for egg in self.eggs:
            if not egg['caught']:
                # Draw egg sprite based on current stage
                sprite_index = egg['sprite_stage'] - 1  # Convert to 0-based index
                self.screen.blit(self.egg_images[sprite_index], (egg['x'], egg['y']))
                
                # Draw fraction text (only if not fully broken)
                if egg['sprite_stage'] < 4:
                    fraction_text = f"{egg['fraction'].numerator}/{egg['fraction'].denominator}"
                    text_surface = self.font_medium.render(fraction_text, True, BLACK)
                    text_rect = text_surface.get_rect(center=(egg['x'] + egg['width']//2, egg['y'] + egg['height']//2))
                    
                    pygame.draw.rect(self.screen, WHITE, text_rect.inflate(8, 4))
                    pygame.draw.rect(self.screen, BLACK, text_rect.inflate(8, 4), 2)
                    self.screen.blit(text_surface, text_rect)
    
    def draw_game_over(self):
        """Draw game over screen"""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        if self.health <= 0:
            title = "GAME OVER!"
            subtitle = "Keep practicing fractions!"
            color = RED
        else:
            title = "AMAZING!"
            subtitle = "You're a fraction master!"
            color = GREEN
        
        title_text = self.font_large.render(title, True, color)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 60))
        self.screen.blit(title_text, title_rect)
        
        subtitle_text = self.font_medium.render(subtitle, True, WHITE)
        subtitle_rect = subtitle_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 20))
        self.screen.blit(subtitle_text, subtitle_rect)
        
        score_text = self.font_medium.render(f"Final Score: {self.score}", True, YELLOW)
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 20))
        self.screen.blit(score_text, score_rect)
        
        restart_text = self.font_medium.render("Press R to restart or Q to quit", True, WHITE)
        restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 60))
        self.screen.blit(restart_text, restart_rect)
    
    def reset_game(self):
        """Reset game"""
        self.current_level = 1
        self.health = 3
        self.score = 0
        self.game_over = False
        self.game_started = False
        self.eggs = []
        self.broken_eggs = []
        self.egg_spawn_timer = 0
        self.egg_speed = 4
        self.egg_spawn_delay = 90
    
    def run(self):
        """Main game loop"""
        print("🍳 HEAD-CONTROLLED PAN FRACTION GAME!")
        print("📸 Move your head left/right to control the pan!")
        print("🥚 Watch eggs break with cool animations!")
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not self.game_started and not self.game_over:
                        self.game_started = True
                        self.start_background_music()
                        print(f"🎮 Game Started! Level {self.current_level}")
                        print(f"🎯 Target: {self.level_data[self.current_level]['target']}")
                    elif event.key == pygame.K_r and self.game_over:
                        self.reset_game()
                    elif event.key == pygame.K_q:
                        running = False
            
            # Head control
            self.detect_face_and_control_pan()
            
            # Game logic
            if self.game_started and not self.game_over:
                # Spawn eggs
                self.egg_spawn_timer += 1
                if self.egg_spawn_timer >= self.egg_spawn_delay:
                    self.eggs.append(self.create_egg())
                    self.egg_spawn_timer = 0
                
                # Update eggs
                self.update_eggs()
                
                # Check level progression
                self.check_level_progression()
                
                # Check game over
                if self.health <= 0:
                    self.game_over = True
                elif self.current_level > self.max_level:
                    self.game_over = True
            
            # Draw everything
            self.screen.blit(self.background, (0, 0))
            
            if self.game_started:
                self.draw_eggs()
                # Draw pan
                self.screen.blit(self.pan_image, (self.pan_x, self.pan_y))
            
            self.draw_ui()
            
            if self.game_over:
                self.draw_game_over()
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    print("🍳 Head-Controlled Pan Fraction Game")
    print("Files needed in same directory:")
    print("- backgroundgame.png")
    print("- pan.png  ← Your cooking pan sprite")
    print("- egg1.png (normal), egg2.png (small crack)")
    print("- egg3.png (more cracks), egg4.png (broken)")
    print("- heart_full.png, heart_empty.png")
    print("- background_music.wav, correct_sound.wav, wrong_sound.wav")
    print()
    
    try:
        game = FractionEggGame()
        game.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Install: pip install pygame opencv-python mediapipe")