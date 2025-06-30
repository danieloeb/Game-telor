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

# NEW: Egg size constants - CHANGE THESE TO ADJUST EGG SIZE!
EGG_WIDTH = 75   # Change this number to make eggs wider/narrower
EGG_HEIGHT = 100  # Change this number to make eggs taller/shorter

# Egg size suggestions:
# Small eggs:   EGG_WIDTH = 40,  EGG_HEIGHT = 50
# Normal eggs:  EGG_WIDTH = 50,  EGG_HEIGHT = 60  (current)
# Large eggs:   EGG_WIDTH = 70,  EGG_HEIGHT = 80
# Giant eggs:   EGG_WIDTH = 90,  EGG_HEIGHT = 110

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
        
        # NEW: Face background system
        self.face_background_surface = None
        self.face_background_opacity = 15  # REALLY REALLY LOW opacity (0-255)
        
        # NEW: Runtime egg size adjustment
        self.current_egg_width = EGG_WIDTH
        self.current_egg_height = EGG_HEIGHT
        
        # Game state
        self.current_level = 1
        self.max_level = 5
        self.health = 5
        self.score = 0
        self.game_over = False
        self.game_started = False
        
        # IMPROVED: Separate countdown for game start and level change
        self.game_countdown_active = False
        self.game_countdown_timer = 0
        self.game_countdown_stage = 0
        
        # Quick level change countdown (0.5 seconds total)
        self.level_countdown_active = False
        self.level_countdown_timer = 0
        self.level_countdown_stage = 0
        
        # Pan properties (controlled by face)
        self.pan_x = SCREEN_WIDTH // 2
        self.pan_y = SCREEN_HEIGHT - 150
        self.pan_width = 400
        self.pan_height = 160
        self.face_detected = False
        self.camera_surface = None
        
        # NEW: All eggs system - every fraction falls once per level
        self.eggs = []
        self.level_fraction_queue = []  # Queue of all fractions to spawn for current level
        self.egg_spawn_timer = 0
        self.egg_spawn_delay = 80  # Spawn delay between eggs
        self.egg_speed = 3
        
        # Load assets
        self.load_assets()
        
        # Level definitions with ORIGINAL display numbers preserved
        self.level_data = {
            1: {
                'target': Fraction(1, 3),
                'correct_choices': [
                    # Store as (numerator, denominator) tuples to preserve original display
                    (7, 21), (13, 39), (17, 51), (19, 57),
                    (23, 69), (29, 87), (31, 93), (37, 99),
                    (11, 33), (2, 6), (4, 12), (5, 15)
                ],
                'wrong_choices': [
                    (8, 23), (11, 32), (13, 38), (17, 49),
                    (19, 55), (23, 67), (29, 85), (31, 91),
                    (37, 97), (3, 8), (5, 14), (7, 20)
                ]
            },
            2: {
                'target': Fraction(2, 5),
                'correct_choices': [
                    (14, 35), (18, 45), (22, 55), (26, 65),
                    (34, 85), (38, 95), (4, 10), (6, 15),
                    (8, 20), (10, 25), (12, 30), (16, 40)
                ],
                'wrong_choices': [
                    (13, 34), (17, 44), (21, 54), (25, 64),
                    (29, 74), (33, 84), (37, 94), (3, 8),
                    (7, 18), (9, 23), (11, 28), (13, 33)
                ]
            },
            3: {
                'target': Fraction(3, 4),
                'correct_choices': [
                    (21, 28), (27, 36), (33, 44), (39, 52),
                    (45, 60), (51, 68), (57, 76), (6, 8),
                    (9, 12), (12, 16), (15, 20), (18, 24)
                ],
                'wrong_choices': [
                    (20, 27), (26, 35), (32, 43), (38, 51),
                    (44, 59), (50, 67), (5, 7), (7, 10),
                    (8, 11), (13, 18), (17, 23), (19, 26)
                ]
            },
            4: {
                'target': Fraction(2, 3),
                'correct_choices': [
                    (14, 21), (18, 27), (22, 33), (26, 39),
                    (34, 51), (38, 57), (42, 63), (4, 6),
                    (6, 9), (8, 12), (10, 15), (12, 18)
                ],
                'wrong_choices': [
                    (13, 20), (17, 26), (21, 32), (25, 38),
                    (29, 44), (33, 50), (37, 56), (3, 5),
                    (5, 8), (7, 11), (9, 14), (11, 17)
                ]
            },
            5: {
                'target': Fraction(4, 5),
                'correct_choices': [
                    (28, 35), (36, 45), (44, 55), (52, 65),
                    (68, 85), (76, 95), (8, 10), (12, 15),
                    (16, 20), (20, 25), (24, 30), (32, 40)
                ],
                'wrong_choices': [
                    (27, 34), (35, 44), (43, 54), (51, 64),
                    (59, 74), (67, 84), (7, 9), (9, 11),
                    (13, 16), (17, 21), (21, 26), (25, 31)
                ]
            }
        }
        
        # Initialize first level
        self.setup_level_fractions()
        
        # Font
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
    def setup_level_fractions(self):
        """Setup all fractions for current level in random order"""
        level_info = self.level_data[self.current_level]
        
        # Create list of ALL fractions for this level (correct + wrong)
        all_fractions = []
        
        # Add correct fractions with ORIGINAL numerator/denominator for display
        for num, den in level_info['correct_choices']:
            all_fractions.append({
                'fraction': Fraction(num, den),  # For equivalence checking
                'display_num': num,  # ORIGINAL numbers for display
                'display_den': den,
                'is_correct': True
            })
        
        # Add wrong fractions with ORIGINAL numerator/denominator for display
        for num, den in level_info['wrong_choices']:
            all_fractions.append({
                'fraction': Fraction(num, den),  # For equivalence checking
                'display_num': num,  # ORIGINAL numbers for display
                'display_den': den,
                'is_correct': False
            })
        
        # Shuffle the order randomly
        random.shuffle(all_fractions)
        
        # Store as queue for spawning
        self.level_fraction_queue = all_fractions.copy()
        
        print(f"🎯 Level {self.current_level} setup: {len(all_fractions)} fractions to spawn")
        print(f"Target: {level_info['target']}")
        print(f"Correct: {len(level_info['correct_choices'])}, Wrong: {len(level_info['wrong_choices'])}")
    
    def reload_egg_sprites(self):
        """Reload egg sprites with new size"""
        self.egg_images = []
        for i in range(1, 5):
            try:
                egg_img = pygame.image.load(f'egg{i}.png').convert_alpha()
                egg_img = pygame.transform.scale(egg_img, (self.current_egg_width, self.current_egg_height))
                self.egg_images.append(egg_img)
            except:
                self.egg_images.append(self.create_egg_sprite(self.current_egg_width, self.current_egg_height, i))
    
    def load_assets(self):
        """Load all game assets"""
        # Background
        try:
            self.background = pygame.image.load('backgroundgame.png').convert()
            self.background = pygame.transform.scale(self.background, (SCREEN_WIDTH, SCREEN_HEIGHT))
        except:
            self.background = self.create_gradient_background()
        
        # Pan image
        try:
            self.pan_image = pygame.image.load('pan.png').convert_alpha()
            self.pan_image = pygame.transform.scale(self.pan_image, (self.pan_width, self.pan_height))
        except:
            self.pan_image = self.create_pan_sprite()
        
        # Egg sprites
        self.egg_images = []
        for i in range(1, 5):
            try:
                egg_img = pygame.image.load(f'egg{i}.png').convert_alpha()
                egg_img = pygame.transform.scale(egg_img, (EGG_WIDTH, EGG_HEIGHT))  # Use constants
                self.egg_images.append(egg_img)
            except:
                self.egg_images.append(self.create_egg_sprite(EGG_WIDTH, EGG_HEIGHT, i))  # Use constants
        
        # Heart sprites
        try:
            self.heart_full = pygame.image.load('heart_full.png').convert_alpha()
            self.heart_empty = pygame.image.load('heart_empty.png').convert_alpha()
            self.heart_full = pygame.transform.scale(self.heart_full, (35, 35))
            self.heart_empty = pygame.transform.scale(self.heart_empty, (35, 35))
        except:
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
            
            try:
                self.collision_sound = pygame.mixer.Sound('collision_sound.wav')
            except:
                self.collision_sound = self.create_collision_sound()
            
            try:
                self.success_sound = pygame.mixer.Sound('success_sound.wav')  
            except:
                self.success_sound = self.create_success_sound()
            
            self.background_music.set_volume(0.2)
            self.music_playing = False
        except:
            self.background_music = None
            self.correct_sound = None
            self.wrong_sound = None
            self.collision_sound = None
            self.success_sound = None
    
    def create_pan_sprite(self):
        """Create a pan sprite as fallback"""
        surface = pygame.Surface((self.pan_width, self.pan_height), pygame.SRCALPHA)
        pygame.draw.circle(surface, (50, 50, 50), (40, 50), 35, 4)
        pygame.draw.circle(surface, (80, 80, 80), (40, 50), 30)
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
            pygame.draw.arc(surface, (255, 248, 220), (0, 0, width//2, height//2), 0, 3.14, 3)
            pygame.draw.arc(surface, (255, 248, 220), (width//2, height//2, width//2, height//2), 3.14, 6.28, 3)
            pygame.draw.circle(surface, (255, 255, 0), (width//2, height//2), width//4)
            pygame.draw.circle(surface, (255, 200, 0), (width//2, height//2), width//5)
        
        return surface
    
    def create_heart_sprite(self, width, height, filled):
        """Create heart sprite"""
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        color = (255, 50, 50) if filled else (100, 100, 100)
        
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
    
    def create_collision_sound(self):
        """Create collision sound when egg hits pan"""
        try:
            sample_rate = 22050
            duration = 0.2
            frequency = 400
            frames = int(duration * sample_rate)
            arr = np.zeros(frames)
            for i in range(frames):
                arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate) * np.exp(-i / (sample_rate * 0.1)) * 0.3
            arr = (arr * 32767).astype(np.int16)
            return pygame.sndarray.make_sound(arr)
        except:
            return None
    
    def create_success_sound(self):
        """Create success sound for correct fraction catch"""
        try:
            sample_rate = 22050
            duration = 0.8
            frames = int(duration * sample_rate)
            arr = np.zeros(frames)
            
            frequencies = [523, 659, 784]  # C, E, G notes
            for freq in frequencies:
                for i in range(frames//3):
                    time_offset = (frequencies.index(freq) * frames//3)
                    if i + time_offset < frames:
                        arr[i + time_offset] += np.sin(2 * np.pi * freq * i / sample_rate) * np.exp(-i / (sample_rate * 0.4)) * 0.3
            
            arr = (arr * 32767).astype(np.int16)
            return pygame.sndarray.make_sound(arr)
        except:
            return None
    
    def detect_face_and_control_pan(self):
        """Use face detection to control pan position AND create giant face background"""
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
            
            # Get face center for pan control
            h, w, _ = frame.shape
            face_center_x = (bbox.xmin + bbox.width / 2) * w
            
            # Map face X position to pan X position (full screen range)
            self.pan_x = int((face_center_x / w) * SCREEN_WIDTH) - self.pan_width // 2
            self.pan_x = max(0, min(self.pan_x, SCREEN_WIDTH - self.pan_width))
            
            # NEW: Extract JUST the face for giant background
            self.create_face_background(frame, bbox)
        else:
            self.face_detected = False
        
        # Convert camera frame to pygame surface for small preview (keep this too)
        frame_resized = cv2.resize(frame, (200, 150))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        self.camera_surface = frame_surface
    
    def create_face_background(self, frame, bbox):
        """Extract face region and create giant transparent background"""
        h, w, _ = frame.shape
        
        # Calculate face region with some padding
        padding = 0.3  # 30% padding around face
        x_min = max(0, int((bbox.xmin - padding * bbox.width) * w))
        x_max = min(w, int((bbox.xmin + bbox.width + padding * bbox.width) * w))
        y_min = max(0, int((bbox.ymin - padding * bbox.height) * h))
        y_max = min(h, int((bbox.ymin + bbox.height + padding * bbox.height) * h))
        
        # Extract just the face region
        face_region = frame[y_min:y_max, x_min:x_max]
        
        if face_region.size > 0:
            # Resize face to fill entire screen
            face_resized = cv2.resize(face_region, (SCREEN_WIDTH, SCREEN_HEIGHT))
            
            # Convert to RGB for pygame
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Create pygame surface
            face_surface = pygame.surfarray.make_surface(face_rgb.swapaxes(0, 1))
            
            # Apply REALLY low opacity
            face_surface.set_alpha(self.face_background_opacity)
            
            # Store for drawing
            self.face_background_surface = face_surface
    
    def start_game_countdown(self):
        """Start the Ready, Set, Go countdown for game start"""
        self.game_countdown_active = True
        self.game_countdown_timer = 0
        self.game_countdown_stage = 0
    
    def start_level_countdown(self):
        """Start quick Ready, Set, Go countdown for level change (0.5 seconds)"""
        self.level_countdown_active = True
        self.level_countdown_timer = 0
        self.level_countdown_stage = 0
    
    def update_game_countdown(self):
        """Update game start countdown"""
        if not self.game_countdown_active:
            return
        
        self.game_countdown_timer += 1
        
        if self.game_countdown_timer == 60:
            self.game_countdown_stage = 1
        elif self.game_countdown_timer == 120:
            self.game_countdown_stage = 2
        elif self.game_countdown_timer == 180:
            self.game_countdown_active = False
            self.game_started = True
            self.start_background_music()
    
    def update_level_countdown(self):
        """Update level change countdown (0.5 seconds total)"""
        if not self.level_countdown_active:
            return
        
        self.level_countdown_timer += 1
        
        if self.level_countdown_timer == 10:
            self.level_countdown_stage = 1
        elif self.level_countdown_timer == 20:
            self.level_countdown_stage = 2
        elif self.level_countdown_timer == 30:
            self.level_countdown_active = False
    
    def draw_camera_feed(self):
        """Draw camera feed with low opacity"""
        if self.camera_surface:
            camera_x = SCREEN_WIDTH - 220
            camera_y = 120
            
            camera_with_alpha = self.camera_surface.copy()
            camera_with_alpha.set_alpha(100)
            
            pygame.draw.rect(self.screen, WHITE, (camera_x - 3, camera_y - 3, 206, 156))
            pygame.draw.rect(self.screen, BLACK, (camera_x - 1, camera_y - 1, 202, 152))
            
            self.screen.blit(camera_with_alpha, (camera_x, camera_y))
            
            you_text = self.font_small.render("YOU", True, WHITE)
            you_rect = you_text.get_rect(center=(camera_x + 100, camera_y - 15))
            pygame.draw.rect(self.screen, BLACK, you_rect.inflate(10, 5))
            self.screen.blit(you_text, you_rect)
    
    def draw_countdown(self, is_level_change=False):
        """Draw Ready, Set, Go countdown"""
        if is_level_change and not self.level_countdown_active:
            return
        elif not is_level_change and not self.game_countdown_active:
            return
        
        countdown_texts = ["READY", "SET", "GO!"]
        countdown_colors = [YELLOW, YELLOW, GREEN]
        
        if is_level_change:
            stage = self.level_countdown_stage
            if stage == 0:
                text = f"LEVEL {self.current_level} - READY"
            else:
                text = countdown_texts[stage]
        else:
            stage = self.game_countdown_stage
            text = countdown_texts[stage]
        
        color = countdown_colors[stage]
        
        countdown_surface = self.font_large.render(text, True, color)
        countdown_rect = countdown_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
        
        pygame.draw.rect(self.screen, BLACK, countdown_rect.inflate(40, 20))
        pygame.draw.rect(self.screen, color, countdown_rect.inflate(40, 20), 4)
        
        self.screen.blit(countdown_surface, countdown_rect)
    
    def start_background_music(self):
        """Start background music"""
        if self.background_music and not self.music_playing:
            self.background_music.play(-1)
            self.music_playing = True
    
    def find_safe_spawn_position(self):
        """Find X position that doesn't collide with existing eggs"""
        margin = 20  # Extra space between eggs
        
        # Get all current egg X positions
        existing_positions = []
        for egg in self.eggs:
            if not egg['caught']:
                existing_positions.append((egg['x'], egg['x'] + self.current_egg_width))  # Use current size
        
        # Try to find non-overlapping position
        max_attempts = 20
        for _ in range(max_attempts):
            x = random.randint(50, SCREEN_WIDTH - 100)
            
            # Check if this position overlaps with any existing egg
            collision = False
            for start_x, end_x in existing_positions:
                if (x < end_x + margin) and (x + self.current_egg_width + margin > start_x):  # Use current size
                    collision = True
                    break
            
            if not collision:
                return x
        
        # If no safe position found, use random (rare case)
        return random.randint(50, SCREEN_WIDTH - 100)
    
    def spawn_next_egg(self):
        """Spawn the next egg from the level queue"""
        if not self.level_fraction_queue:
            return None  # No more eggs for this level
        
        # Get next fraction from queue
        next_fraction_data = self.level_fraction_queue.pop(0)
        
        # Find safe spawn position
        safe_x = self.find_safe_spawn_position()
        
        egg = {
            'x': safe_x,
            'y': -60,
            'fraction': next_fraction_data['fraction'],  # For equivalence checking
            'display_num': next_fraction_data['display_num'],  # ORIGINAL numbers for display
            'display_den': next_fraction_data['display_den'],
            'is_correct': next_fraction_data['is_correct'],
            'sprite_stage': 1,
            'width': self.current_egg_width,   # Use current size
            'height': self.current_egg_height, # Use current size
            'caught': False,
            'breaking': False,
            'break_timer': 0
        }
        
        target = self.level_data[self.current_level]['target']
        status = "✓ CORRECT" if egg['is_correct'] else "✗ WRONG"
        display_fraction = f"{egg['display_num']}/{egg['display_den']}"
        print(f"🥚 Spawned: {display_fraction} (Target: {target}) - {status}")
        
        return egg
    
    def update_eggs(self):
        """Update eggs with FIXED LOGIC"""
        if self.level_countdown_active:
            return
            
        eggs_to_remove = []
        
        for i, egg in enumerate(self.eggs):
            if not egg['caught'] and not egg['breaking']:
                # Move egg down
                egg['y'] += self.egg_speed
                
                # Check collision with pan
                if self.check_collision_with_pan(egg):
                    self.handle_pan_collision(egg)
                
                # Check if egg hits ground
                elif egg['y'] >= SCREEN_HEIGHT - 80:
                    self.handle_ground_collision(egg)
            
            # Handle breaking animation
            elif egg['breaking']:
                egg['break_timer'] += 1
                
                if egg['break_timer'] == 10:
                    egg['sprite_stage'] = 2
                elif egg['break_timer'] == 20:
                    egg['sprite_stage'] = 3
                elif egg['break_timer'] == 30:
                    egg['sprite_stage'] = 4
                elif egg['break_timer'] >= 60:
                    eggs_to_remove.append(i)
        
        # Remove finished eggs
        for i in reversed(eggs_to_remove):
            del self.eggs[i]
    
    def handle_pan_collision(self, egg):
        """FIXED: Handle pan collision with correct logic"""
        if self.collision_sound:
            self.collision_sound.play()
        
        egg['breaking'] = True
        egg['break_timer'] = 0
        egg['caught'] = True
        
        target = self.level_data[self.current_level]['target']
        display_fraction = f"{egg['display_num']}/{egg['display_den']}"
        
        if egg['is_correct']:
            # Caught CORRECT fraction - GOOD!
            self.score += 10
            if self.success_sound:
                pygame.time.set_timer(pygame.USEREVENT + 1, 200)
            print(f"🎉 EXCELLENT! Caught {display_fraction} = {target} (+10 points)")
        else:
            # Caught WRONG fraction - BAD!
            self.health -= 1
            if self.wrong_sound:
                pygame.time.set_timer(pygame.USEREVENT + 2, 200)
            print(f"💔 OOPS! Caught wrong fraction {display_fraction} ≠ {target} (-1 heart)")
    
    def handle_ground_collision(self, egg):
        """FIXED: Handle ground collision with correct logic"""
        if self.collision_sound:
            self.collision_sound.play()
        
        egg['breaking'] = True
        egg['break_timer'] = 0
        egg['y'] = SCREEN_HEIGHT - 80
        
        target = self.level_data[self.current_level]['target']
        display_fraction = f"{egg['display_num']}/{egg['display_den']}"
        
        if egg['is_correct']:
            # Missed CORRECT fraction - BAD!
            self.health -= 1
            print(f"💔 MISSED correct answer {display_fraction} = {target} (-1 heart)")
        else:
            # Missed WRONG fraction - GOOD! (No penalty)
            print(f"😌 Good! Avoided wrong fraction {display_fraction} ≠ {target}")
    
    def check_collision_with_pan(self, egg):
        """Check if egg collides with pan"""
        pan_rect = pygame.Rect(self.pan_x, self.pan_y, self.pan_width, self.pan_height)
        egg_rect = pygame.Rect(egg['x'], egg['y'], egg['width'], egg['height'])
        return pan_rect.colliderect(egg_rect)
    
    def check_level_progression(self):
        """Check if ready for next level"""
        # Check if all fractions for this level have been spawned and completed
        all_spawned = len(self.level_fraction_queue) == 0
        no_active_eggs = len([egg for egg in self.eggs if not egg['breaking']]) == 0
        
        if all_spawned and no_active_eggs and self.current_level < self.max_level:
            # Level complete! Move to next level
            self.current_level += 1
            self.egg_speed += 0.5
            self.egg_spawn_delay = max(40, self.egg_spawn_delay - 10)
            
            # Setup next level
            self.setup_level_fractions()
            
            # Clear any remaining eggs
            self.eggs = []
            
            # Start quick level countdown
            self.start_level_countdown()
            
            print(f"🎊 LEVEL UP! Now Level {self.current_level}")
    
    def draw_ui(self):
        """Draw game UI"""
        # Target fraction
        target = self.level_data[self.current_level]['target']
        remaining = len(self.level_fraction_queue)
        target_text = self.font_large.render(f"Level {self.current_level}: Catch = {target} ({remaining} left)", True, WHITE)
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
        for i in range(5):
            heart_x = SCREEN_WIDTH - 230 + (i * 40)
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
        
        # NEW: Face opacity indicator
        opacity_text = self.font_small.render(f"👻 Face Opacity: {self.face_background_opacity}% (↑↓ to adjust)", True, (150, 255, 150))
        opacity_rect = opacity_text.get_rect(topleft=(20, 95))
        pygame.draw.rect(self.screen, BLACK, opacity_rect.inflate(10, 5))
        self.screen.blit(opacity_text, opacity_rect)
        
        # NEW: Egg size indicator
        egg_size_text = self.font_small.render(f"🥚 Egg Size: {self.current_egg_width}x{self.current_egg_height} (+/- to adjust)", True, (255, 200, 100))
        egg_size_rect = egg_size_text.get_rect(topleft=(20, 120))
        pygame.draw.rect(self.screen, BLACK, egg_size_rect.inflate(10, 5))
        self.screen.blit(egg_size_text, egg_size_rect)
        
        # Instructions
        if not self.game_started and not self.game_countdown_active:
            instructions = [
                "🍳 HEAD-CONTROLLED PAN GAME!",
                "CATCH correct fractions (= target) for points",
                "AVOID wrong fractions (≠ target) or lose hearts",
                "👻 Your face is the ghostly background!",
                "↑↓ Arrow keys: Adjust face opacity",
                "🥚 +/- keys: Adjust egg size",
                "Press SPACE for Ready, Set, Go!"
            ]
            
            for i, instruction in enumerate(instructions):
                color = YELLOW if i == 0 else WHITE
                if i == 3:  # Highlight the face feature
                    color = (150, 255, 150)  # Light green
                elif i == 5:  # Highlight egg size feature
                    color = (255, 200, 100)  # Orange
                text = self.font_medium.render(instruction, True, color)
                text_rect = text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 120 + i*25))
                pygame.draw.rect(self.screen, BLACK, text_rect.inflate(20, 8))
                if i == 0:
                    pygame.draw.rect(self.screen, YELLOW, text_rect.inflate(20, 8), 2)
                elif i == 3:
                    pygame.draw.rect(self.screen, (150, 255, 150), text_rect.inflate(20, 8), 2)
                elif i == 5:
                    pygame.draw.rect(self.screen, (255, 200, 100), text_rect.inflate(20, 8), 2)
                self.screen.blit(text, text_rect)
    
    def draw_eggs(self):
        """Draw all eggs with ORIGINAL fractions (not simplified)"""
        for egg in self.eggs:
            if not egg['caught']:
                sprite_index = egg['sprite_stage'] - 1
                self.screen.blit(self.egg_images[sprite_index], (egg['x'], egg['y']))
                
                if egg['sprite_stage'] < 4:
                    # Display ORIGINAL numbers, not simplified
                    fraction_text = f"{egg['display_num']}/{egg['display_den']}"
                    text_surface = self.font_medium.render(fraction_text, True, BLACK)
                    text_rect = text_surface.get_rect(center=(egg['x'] + egg['width']//2, egg['y'] + egg['height']//2))
                    
                    # Simple white background - NO COLOR CODING
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
            subtitle = "You completed all levels!"
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
        self.health = 5
        self.score = 0
        self.game_over = False
        self.game_started = False
        self.game_countdown_active = False
        self.game_countdown_timer = 0
        self.game_countdown_stage = 0
        self.level_countdown_active = False
        self.level_countdown_timer = 0
        self.level_countdown_stage = 0
        self.eggs = []
        self.egg_spawn_timer = 0
        self.egg_speed = 3
        self.egg_spawn_delay = 80
        
        # Reset egg size to default
        self.current_egg_width = EGG_WIDTH
        self.current_egg_height = EGG_HEIGHT
        self.reload_egg_sprites()
        
        self.setup_level_fractions()
    
    def run(self):
        """Main game loop"""
        print("🍳 HEAD-CONTROLLED PAN FRACTION GAME - GHOSTLY FACE!")
        print("✅ Catch correct fractions = +10 points")
        print("❌ Catch wrong fractions = -1 heart")
        print("💔 Miss correct fractions = -1 heart")
        print("😌 Miss wrong fractions = No penalty")
        print("🎯 ALL fractions from each level will fall!")
        print("📊 Shows ORIGINAL numbers like 14/35, not simplified 2/5!")
        print("👻 Your face appears as giant transparent background!")
        print("↑↓ Use arrow keys to adjust face opacity!")
        print("🥚 Use +/- keys to adjust egg size during gameplay!")
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not self.game_started and not self.game_over and not self.game_countdown_active:
                        self.start_game_countdown()
                    elif event.key == pygame.K_r and self.game_over:
                        self.reset_game()
                    elif event.key == pygame.K_q:
                        running = False
                    # NEW: Adjust face background opacity
                    elif event.key == pygame.K_UP:
                        self.face_background_opacity = min(100, self.face_background_opacity + 5)
                        print(f"👻 Face opacity: {self.face_background_opacity}")
                    elif event.key == pygame.K_DOWN:
                        self.face_background_opacity = max(5, self.face_background_opacity - 5)
                        print(f"👻 Face opacity: {self.face_background_opacity}")
                    
                    # NEW: Adjust egg size during gameplay
                    elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:  # + key
                        self.current_egg_width = min(100, self.current_egg_width + 5)
                        self.current_egg_height = min(120, self.current_egg_height + 6)
                        self.reload_egg_sprites()
                        print(f"🥚 Egg size: {self.current_egg_width}x{self.current_egg_height}")
                    elif event.key == pygame.K_MINUS:  # - key
                        self.current_egg_width = max(30, self.current_egg_width - 5)
                        self.current_egg_height = max(36, self.current_egg_height - 6)
                        self.reload_egg_sprites()
                        print(f"🥚 Egg size: {self.current_egg_width}x{self.current_egg_height}")
                elif event.type == pygame.USEREVENT + 1:  # Delayed success sound
                    if self.success_sound:
                        self.success_sound.play()
                    pygame.time.set_timer(pygame.USEREVENT + 1, 0)
                elif event.type == pygame.USEREVENT + 2:  # Delayed wrong sound
                    if self.wrong_sound:
                        self.wrong_sound.play()
                    pygame.time.set_timer(pygame.USEREVENT + 2, 0)
            
            # Head control
            self.detect_face_and_control_pan()
            
            # Handle countdowns
            if self.game_countdown_active:
                self.update_game_countdown()
            if self.level_countdown_active:
                self.update_level_countdown()
            
            # Game logic
            if self.game_started and not self.game_over and not self.level_countdown_active:
                # Spawn eggs from queue
                self.egg_spawn_timer += 1
                if self.egg_spawn_timer >= self.egg_spawn_delay and self.level_fraction_queue:
                    new_egg = self.spawn_next_egg()
                    if new_egg:
                        self.eggs.append(new_egg)
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
            
            # Draw everything in the correct order
            # 1. FIRST: Giant transparent face background (behind everything)
            if self.face_background_surface:
                self.screen.blit(self.face_background_surface, (0, 0))
            
            # 2. SECOND: Regular background image (on top of face)
            self.screen.blit(self.background, (0, 0))
            
            self.draw_camera_feed()
            
            if self.game_started:
                self.draw_eggs()
                self.screen.blit(self.pan_image, (self.pan_x, self.pan_y))
            
            self.draw_ui()
            
            if self.game_countdown_active:
                self.draw_countdown(is_level_change=False)
            elif self.level_countdown_active:
                self.draw_countdown(is_level_change=True)
            
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
    print("🍳 Fraction Egg Game - GHOSTLY FACE + ADJUSTABLE EGGS!")
    print("🎯 ALL fractions fall from each level")
    print("⚡ 0.5-second level transitions")
    print("✅ CORRECT game logic implemented")
    print("🔍 No collision spawning system")
    print("📊 Shows 14/35, 18/45, etc. - NOT simplified!")
    print("👻 YOUR FACE AS GIANT TRANSPARENT BACKGROUND!")
    print("↑↓ Arrow keys adjust face opacity!")
    print("🥚 +/- keys adjust egg size in real-time!")
    
    try:
        game = FractionEggGame()
        game.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Install: pip install pygame opencv-python mediapipe")