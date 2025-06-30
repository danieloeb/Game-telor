import cv2
import numpy as np
import random
import time
from fractions import Fraction

class MathEggCatchingGame:
    def __init__(self):  # FIXED: was _init_ (missing double underscores)
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        # Load face detection classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Game variables
        self.score = 0
        self.lives = 3
        self.current_level = 1
        self.max_level = 5
        self.game_over = False
        self.game_started = False
        self.level_complete = False
        
        # Egg spawning properties
        self.eggs = []
        self.egg_speed = 3
        self.egg_spawn_rate = 0.015  # Probability of spawning egg each frame
        self.last_spawn_time = time.time()
        self.spawn_delay = 1.5  # Minimum seconds between spawns
        
        # Bowl properties
        self.bowl_width = 100
        self.bowl_height = 50
        self.bowl_x = 0
        self.bowl_y = 0
        
        # Asset loading
        self.load_assets()
        
        # Level definitions - target fractions for each level
        self.level_targets = {
            1: Fraction(1, 2),    # 1/2
            2: Fraction(2, 3),    # 2/3  
            3: Fraction(3, 4),    # 3/4
            4: Fraction(4, 5),    # 4/5
            5: Fraction(5, 6)     # 5/6
        }
        
        # Animation properties
        self.cracking_eggs = []  # For cracking animation
        
    def load_assets(self):
        """Load game assets - you'll replace these with your actual images"""
        try:
            # Try to load your custom assets
            self.bowl_img = cv2.imread('assets/bowl.png', cv2.IMREAD_UNCHANGED)
            self.egg_img = cv2.imread('assets/egg.png', cv2.IMREAD_UNCHANGED)
            self.cracked_egg_img = cv2.imread('assets/cracked_egg.png', cv2.IMREAD_UNCHANGED)
            
            # If assets don't exist, we'll draw shapes instead
            if self.bowl_img is None:
                print("Bowl image not found, using drawn shape")
                self.use_drawn_bowl = True
            else:
                self.use_drawn_bowl = False
                self.bowl_img = cv2.resize(self.bowl_img, (self.bowl_width, self.bowl_height))
            
            if self.egg_img is None:
                print("Egg image not found, using drawn shape")
                self.use_drawn_egg = True
            else:
                self.use_drawn_egg = False
                self.egg_img = cv2.resize(self.egg_img, (40, 50))
            
            if self.cracked_egg_img is None:
                print("Cracked egg image not found, using drawn shape")
                self.use_drawn_cracked = True
            else:
                self.use_drawn_cracked = False
                self.cracked_egg_img = cv2.resize(self.cracked_egg_img, (40, 50))
                
        except Exception as e:
            print(f"Asset loading error: {e}")
            self.use_drawn_bowl = True
            self.use_drawn_egg = True
            self.use_drawn_cracked = True
    
    def generate_wrong_fractions(self, target_fraction, count=2):
        """Generate fractions that are NOT equivalent to target"""
        wrong_fractions = []
        
        while len(wrong_fractions) < count:
            # Generate random fractions
            num = random.randint(1, 20)
            den = random.randint(num + 1, 25)  # Ensure proper fraction
            frac = Fraction(num, den)
            
            # Make sure it's not equivalent to target
            if frac != target_fraction and frac not in wrong_fractions:
                wrong_fractions.append(frac)
        
        return wrong_fractions
    
    def create_single_egg(self, frame_width):
        """Create a single falling egg with a fraction"""
        target = self.level_targets[self.current_level]
        
        # 60% chance for correct fraction, 40% chance for wrong
        if random.random() < 0.6:
            # Generate equivalent fraction
            multiplier = random.randint(2, 8)
            fraction = Fraction(target.numerator * multiplier, target.denominator * multiplier)
            is_correct = True
        else:
            # Generate wrong fraction
            num = random.randint(1, 20)
            den = random.randint(num + 1, 25)
            fraction = Fraction(num, den)
            # Make sure it's actually wrong
            if fraction == target:
                fraction = Fraction(num + 1, den) if num + 1 < den else Fraction(num, den + 1)
            is_correct = False
        
        egg = {
            'x': random.randint(50, frame_width - 90),  # Random position
            'y': 0,
            'width': 40,
            'height': 50,
            'fraction': fraction,
            'is_correct': is_correct,
            'caught': False,
            'cracking': False,
            'crack_timer': 0
        }
        
        return egg
    
    def draw_image_with_alpha(self, background, image, x, y):
        """Draw image with transparency support"""
        if image.shape[2] == 4:  # Has alpha channel
            # Extract alpha channel
            alpha = image[:, :, 3] / 255.0
            
            # Get the region where image will be placed
            h, w = image.shape[:2]
            bg_h, bg_w = background.shape[:2]
            
            # Ensure we don't go out of bounds
            x = max(0, min(x, bg_w - w))
            y = max(0, min(y, bg_h - h))
            
            # Blend the images
            for c in range(3):  # RGB channels
                background[y:y+h, x:x+w, c] = (
                    alpha * image[:, :, c] + 
                    (1 - alpha) * background[y:y+h, x:x+w, c]
                )
        else:
            # No alpha channel, just overlay
            h, w = image.shape[:2]
            bg_h, bg_w = background.shape[:2]
            x = max(0, min(x, bg_w - w))
            y = max(0, min(y, bg_h - h))
            background[y:y+h, x:x+w] = image
    
    def draw_egg(self, frame, egg):
        """Draw an egg with its fraction"""
        if egg['cracking']:
            # Draw cracking animation
            if not self.use_drawn_cracked:
                self.draw_image_with_alpha(frame, self.cracked_egg_img, int(egg['x']), int(egg['y']))
            else:
                # Draw cracked egg shape
                center = (int(egg['x'] + egg['width']//2), int(egg['y'] + egg['height']//2))
                cv2.ellipse(frame, center, (egg['width']//2, egg['height']//2), 0, 0, 360, (100, 100, 100), -1)
                # Add crack lines
                cv2.line(frame, (center[0]-10, center[1]-5), (center[0]+10, center[1]+5), (0, 0, 0), 2)
                cv2.line(frame, (center[0]-5, center[1]-10), (center[0]+5, center[1]+10), (0, 0, 0), 2)
        else:
            # Draw normal egg
            if not self.use_drawn_egg:
                self.draw_image_with_alpha(frame, self.egg_img, int(egg['x']), int(egg['y']))
            else:
                # Draw egg shape (oval)
                center = (int(egg['x'] + egg['width']//2), int(egg['y'] + egg['height']//2))
                cv2.ellipse(frame, center, (egg['width']//2, egg['height']//2), 0, 0, 360, (0, 255, 255), -1)
                cv2.ellipse(frame, center, (egg['width']//2, egg['height']//2), 0, 0, 360, (0, 200, 200), 2)
        
        # Draw fraction on egg
        fraction_text = f"{egg['fraction'].numerator}/{egg['fraction'].denominator}"
        text_size = cv2.getTextSize(fraction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = int(egg['x'] + (egg['width'] - text_size[0]) // 2)
        text_y = int(egg['y'] + egg['height'] // 2 + text_size[1] // 2)
        
        # Draw text background
        cv2.rectangle(frame, (text_x - 2, text_y - text_size[1] - 2), 
                     (text_x + text_size[0] + 2, text_y + 2), (255, 255, 255), -1)
        cv2.putText(frame, fraction_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    def draw_bowl(self, frame):
        """Draw the bowl above the detected face"""
        if not self.use_drawn_bowl:
            self.draw_image_with_alpha(frame, self.bowl_img, int(self.bowl_x), int(self.bowl_y))
        else:
            # Draw bowl shape
            bowl_points = np.array([
                [self.bowl_x, self.bowl_y + self.bowl_height],
                [self.bowl_x + 15, self.bowl_y + self.bowl_height - 10],
                [self.bowl_x + self.bowl_width - 15, self.bowl_y + self.bowl_height - 10],
                [self.bowl_x + self.bowl_width, self.bowl_y + self.bowl_height],
                [self.bowl_x + self.bowl_width - 10, self.bowl_y + self.bowl_height - 20],
                [self.bowl_x + 10, self.bowl_y + self.bowl_height - 20]
            ], np.int32)
            
            cv2.fillPoly(frame, [bowl_points], (139, 69, 19))
            cv2.polylines(frame, [bowl_points], True, (101, 67, 33), 3)
    
    def check_collision(self, egg):
        """Check if egg collides with bowl"""
        egg_center_x = egg['x'] + egg['width'] // 2
        egg_bottom = egg['y'] + egg['height']
        
        bowl_left = self.bowl_x
        bowl_right = self.bowl_x + self.bowl_width
        bowl_top = self.bowl_y
        bowl_bottom = self.bowl_y + self.bowl_height
        
        if (bowl_left < egg_center_x < bowl_right and 
            bowl_top < egg_bottom < bowl_bottom + 20):
            return True
        return False
    
    def spawn_egg(self, frame_width):
        """Spawn new eggs randomly over time"""
        current_time = time.time()
        if (random.random() < self.egg_spawn_rate and 
            current_time - self.last_spawn_time > self.spawn_delay):
            new_egg = self.create_single_egg(frame_width)
            self.eggs.append(new_egg)
            self.last_spawn_time = current_time
    
    def update_eggs(self, frame_height):
        """Update egg positions and check for collisions"""
        eggs_to_remove = []
        
        for i, egg in enumerate(self.eggs):
            if not egg['caught']:
                if egg['cracking']:
                    # Handle cracking animation
                    egg['crack_timer'] += 1
                    if egg['crack_timer'] > 30:  # Show crack for 30 frames
                        eggs_to_remove.append(i)
                else:
                    # Move egg down
                    egg['y'] += self.egg_speed
                    
                    # Check collision with bowl
                    if self.check_collision(egg):
                        egg['cracking'] = True
                        egg['crack_timer'] = 0
                        
                        if egg['is_correct']:
                            self.score += 20
                            print(f"Correct! {egg['fraction']} = {self.level_targets[self.current_level]}")
                        else:
                            self.lives -= 1
                            print(f"Wrong! {egg['fraction']} ≠ {self.level_targets[self.current_level]}")
                    
                    # Check if egg hit bottom (missed)
                    elif egg['y'] > frame_height:
                        eggs_to_remove.append(i)
                        if egg['is_correct']:
                            self.lives -= 1  # Lose life for missing correct answer
        
        # Remove eggs that are done
        for i in reversed(eggs_to_remove):
            del self.eggs[i]
    
    def check_level_progression(self):
        """Check if player should advance to next level"""
        # Advance level every 100 points (5 correct catches)
        target_score = self.current_level * 100
        if self.score >= target_score and self.current_level < self.max_level:
            self.current_level += 1
            # Increase difficulty slightly
            self.egg_speed += 0.5
            self.egg_spawn_rate += 0.005
            print(f"Level {self.current_level}! Target: {self.level_targets[self.current_level]}")
            print(f"Eggs are falling faster now!")
            
            # Check if game is complete
            if self.current_level > self.max_level:
                print("Congratulations! You mastered all levels!")
                self.game_over = True
    
    def draw_ui(self, frame):
        """Draw game UI"""
        height, width = frame.shape[:2]
        
        # Draw semi-transparent overlay for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Current target fraction
        target = self.level_targets[self.current_level]
        cv2.putText(frame, f"Level {self.current_level}: Catch = {target}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Game stats
        cv2.putText(frame, f"Score: {self.score}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Lives: {self.lives}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Next level progress
        next_level_score = self.current_level * 100
        if self.current_level < self.max_level:
            cv2.putText(frame, f"Next level: {next_level_score} pts", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if not self.game_started:
            cv2.putText(frame, "Press SPACE to start!", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Game over screen
        if self.game_over:
            cv2.rectangle(frame, (width//4, height//3), (3*width//4, 2*height//3), (0, 0, 0), -1)
            if self.lives <= 0:
                cv2.putText(frame, "GAME OVER!", (width//4 + 50, height//2 - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "CONGRATULATIONS!", (width//4 + 10, height//2 - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                cv2.putText(frame, "All levels complete!", (width//4 + 30, height//2 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Final Score: {self.score}", (width//4 + 30, height//2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Press R to restart", (width//4 + 50, height//2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def detect_face_and_position_bowl(self, frame):
        """Detect face and position bowl above it"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Optional: draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Position bowl above the face
            self.bowl_x = x + w//2 - self.bowl_width//2
            self.bowl_y = max(10, y - 70)
            break
    
    def reset_game(self):
        """Reset game to initial state"""
        self.score = 0
        self.lives = 3
        self.current_level = 1
        self.game_over = False
        self.game_started = False
        self.eggs = []
        self.egg_speed = 3
        self.egg_spawn_rate = 0.015
    
    def run(self):
        """Main game loop"""
        print("🥚 Math Egg Catching Game Started!")
        print("Instructions:")
        print("- Each level has a target fraction (like 1/2)")
        print("- Catch eggs with equivalent fractions (like 2/4, 3/6)")
        print("- Avoid wrong fractions or lose lives!")
        print("- Press SPACE to start, R to restart, Q to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read from camera")
                break
            
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            
            # Detect face and position bowl
            self.detect_face_and_position_bowl(frame)
            
            # Game logic
            if self.game_started and not self.game_over:
                # Spawn eggs continuously
                self.spawn_egg(width)
                
                # Update eggs
                self.update_eggs(height)
                
                # Check level progression
                self.check_level_progression()
                
                # Check game over condition
                if self.lives <= 0:
                    self.game_over = True
            
            # Draw game elements
            if self.game_started:
                # Draw eggs
                for egg in self.eggs:
                    self.draw_egg(frame, egg)
            
            # Draw bowl
            self.draw_bowl(frame)
            
            # Draw UI
            self.draw_ui(frame)
            
            # Display frame
            cv2.imshow('Math Egg Catching Game', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and not self.game_started and not self.game_over:
                self.game_started = True
                print(f"Level {self.current_level} Started! Target: {self.level_targets[self.current_level]}")
            elif key == ord('r') and self.game_over:
                self.reset_game()
                print("Game Reset!")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

# Run the game
if __name__ == "__main__":  # FIXED: was "_main_" (missing double underscores)
    try:
        game = MathEggCatchingGame()
        game.run()
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a camera connected and OpenCV installed")