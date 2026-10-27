import random
import time

# ==========================================
# 🏛️ PART 1: THE BLUEPRINT (Parent Class)
# Concept: Class, Object, __init__, self
# ==========================================
class VoltageSensor:
    """
    A standard voltage sensor.
    Represents the 'Base Model' hardware.
    """
    def __init__(self, sensor_id, location):
        # PUBLIC ATTRIBUTES (Accessible by anyone)
        self.sensor_id = sensor_id
        self.location = location
        
        # PROTECTED ATTRIBUTE (Accessible by Child Classes)
        # One underscore (_) tells devs: "Don't touch unless you are a subclass."
        self._status = "Active"

        # PRIVATE ATTRIBUTE (Encapsulation - The 'Black Box')
        # Double underscore (__) hides this. It simulates a factory calibration
        # that users should NOT mess with directly.
        self.__calibration_offset = 0.05  

    def read_voltage(self):
        """Simulates reading a pin value (0-5V) + calibration noise."""
        # Using 'self' to access the private offset specific to THIS object
        raw_val = random.uniform(0, 5) + self.__calibration_offset
        return round(raw_val, 3)

    def set_calibration(self, new_offset):
        """
        Public method to safely modify private data.
        This is a 'Setter' - it controls how data changes.
        """
        if -0.1 <= new_offset <= 0.1:
            self.__calibration_offset = new_offset
            print(f"✅ [{self.sensor_id}] Calibration updated to {new_offset}")
        else:
            print(f"❌ [{self.sensor_id}] Error: Offset too high! Rejected.")

# ==========================================
# 🚀 PART 2: THE UPGRADE (Inheritance)
# Concept: Child Class, super(), Extending Logic
# ==========================================
class SmartSensor(VoltageSensor):
    """
    A 'Pro' version of the sensor.
    Inherits everything from VoltageSensor but adds averaging logic.
    """
    def __init__(self, sensor_id, location, buffer_size=5):
        # 1. Initialize the Parent first (The Foundation)
        # super() calls the __init__ of VoltageSensor
        super().__init__(sensor_id, location)
        
        # 2. Add New Attributes (The Upgrade)
        self.buffer_size = buffer_size
        self.memory_buffer = []

    def get_smooth_reading(self):
        """
        New Method: Takes multiple readings and averages them.
        This is functionality the basic sensor DOES NOT have.
        """
        print(f"🔎 [{self.sensor_id}] Sampling {self.buffer_size} times...")
        total = 0
        for _ in range(self.buffer_size):
            # We can use 'read_voltage' because we INHERITED it!
            val = self.read_voltage()
            total += val
            time.sleep(0.1) # Simulate hardware delay
        
        avg = round(total / self.buffer_size, 3)
        return avg

# ==========================================
# 🧪 PART 3: THE APPLICATION EXERCISE
# Concept: Instantiation, Testing Encapsulation
# ==========================================
if __name__ == "__main__":
    print("\n--- 1. INSTANTIATION (Creating Objects) ---")
    # Creating a basic object
    basic_sensor = VoltageSensor("V-101", "Lab Bench")
    # Creating an advanced object (Child)
    ai_sensor = SmartSensor("AI-200", "Field Robot", buffer_size=3)

    print("\n--- 2. INHERITANCE IN ACTION ---")
    # The Child (AI Sensor) can use Parent methods
    print(f"Basic Reading: {basic_sensor.read_voltage()} V")
    print(f"Smart Reading: {ai_sensor.read_voltage()} V (Inherited Method)")
    
    # The Child has its OWN methods too
    avg_val = ai_sensor.get_smooth_reading()
    print(f"Smoothed AI Reading: {avg_val} V")

    print("\n--- 3. ENCAPSULATION TEST (The Security Check) ---")
    # Try to change private data efficiently using the Setter
    basic_sensor.set_calibration(0.08)

    # Try to 'hack' the private variable directly
    print("🕵️ Attempting to hack private calibration data...")
    try:
        print(basic_sensor.__calibration_offset)
    except AttributeError:
        print("🛡️ ACCESS DENIED: Python protected the private variable!")

    # EXERCISE FOR YOU:
    # 1. Create a new class called 'WirelessSensor' that inherits from 'SmartSensor'.
    # 2. Add a method called 'upload_data()' that just prints "Uploading [value] to Cloud...".
    # 3. Create an instance and call: read -> smooth -> upload.