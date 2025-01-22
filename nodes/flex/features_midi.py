from .features import  BaseFeature
import numpy as np

class MIDIFeature(BaseFeature):
    ATTRIBUTE_MAP = {
        "Velocity": "velocity",
        "Pitch": "pitch",
        "Note On/Off": "on_off",
        "Note Duration": "duration",
        "Note Density": "density",
        "Pitchbend": "pitchbend",
        "Aftertouch": "aftertouch",
        "Poly Pressure": "poly_pressure",
        "Modulation (CC1)": "cc1",
        "Breath (CC2)": "cc2",
        "Foot Controller (CC4)": "cc4",
        "Volume (CC7)": "cc7",
        "Balance (CC8)": "cc8",
        "Pan (CC10)": "cc10",
        "Expression (CC11)": "cc11",
        "Sustain (CC64)": "cc64"
    }

    @classmethod
    def get_extraction_methods(cls):
        """Return a list of parameter names that can be modulated."""
        return list(cls.ATTRIBUTE_MAP.keys())

    def __init__(self, name, midi_data, attribute, frame_rate, frame_count, width, height, notes=None, chord_only=False):
        super().__init__(name, "midi", frame_rate, frame_count, width, height)
        self.midi_data = midi_data
        self.attribute = attribute
        self.notes = set(notes) if notes is not None else set()
        self.chord_only = chord_only
        self.total_time = self.calculate_total_time()
        self.data = None
        self.time_points = None
        self.cc_data = {cc: [] for cc in [1, 2, 4, 7, 8, 10, 11, 64]}
        self.pitchbend_data = []
        self.aftertouch_data = []
        self.poly_pressure_data = []
        

    def extract(self):
        try:
            self.data = self.extract_attribute()
        except ValueError as e:
            print(f"Warning: {str(e)}. Using default values.")
            self.data = np.zeros(self.frame_count)
        return self.normalize()

    def calculate_total_time(self):
        total_time = sum(msg.time for msg in self.midi_data if not msg.is_meta)
        return max(total_time, self.frame_count / self.frame_rate)

    def extract_attribute(self):
        attribute_values = []
        time_points = []
        current_time = 0
        note_on_times = {}
        active_notes = set()
        note_counts = np.zeros(self.frame_count)
        modulation_data = []

        for msg in self.midi_data:
            if msg.is_meta:
                continue

            current_time += msg.time
            frame_index = int(current_time * self.frame_rate)
            # Remove the following lines to prevent early termination
            # if frame_index >= self.frame_count:
            #     break

            value = self.process_message(msg, current_time, note_on_times, active_notes, note_counts, modulation_data)
            if value is not None:
                attribute_values.append(value)
                time_points.append(current_time)

        return self.finalize_attribute(attribute_values, time_points, note_counts, modulation_data)
    def process_message(self, msg, current_time, note_on_times, active_notes, note_counts, modulation_data):
        frame_index = int(current_time * self.frame_rate)

        if msg.type == 'control_change':
            if msg.control in self.cc_data:
                self.cc_data[msg.control].append((current_time, msg.value / 127.0))
            if msg.control == 1:
                modulation_data.append((current_time, msg.value / 127.0))

        elif msg.type == 'pitchwheel':
            self.pitchbend_data.append((current_time, (msg.pitch + 8192) / 16383.0))

        elif msg.type == 'aftertouch':
            self.aftertouch_data.append((current_time, msg.value / 127.0))

        elif msg.type == 'polytouch':
            self.poly_pressure_data.append((current_time, msg.value / 127.0))

        if msg.type == 'note_on' and (not self.notes or msg.note in self.notes):
            note_on_times[msg.note] = current_time
            active_notes.add(msg.note)
            if frame_index < self.frame_count:
                note_counts[frame_index:] += 1

            if not self.chord_only or (self.chord_only and active_notes == self.notes):
                return self.get_attribute_value_for_message(msg, 'on')

        elif msg.type == 'note_off' and msg.note in note_on_times:
            if msg.note in active_notes:
                active_notes.remove(msg.note)
            if frame_index < self.frame_count:
                note_counts[frame_index:] -= 1
            duration = current_time - note_on_times[msg.note]
            del note_on_times[msg.note]

            if not self.chord_only or (self.chord_only and len(active_notes) == 0):
                return self.get_attribute_value_for_message(msg, 'off', duration)

        return None

    def get_attribute_value_for_message(self, msg, state, duration=None):
        if self.attribute == 'velocity' and state == 'on':
            return msg.velocity / 127.0
        elif self.attribute == 'pitch' and state == 'on':
            return msg.note / 127.0
        elif self.attribute == 'on_off':
            return 1.0 if state == 'on' else 0.0
        elif self.attribute == 'duration' and state == 'off':
            return duration
        elif self.attribute == 'pitchbend':
            return (msg.pitch + 8192) / 16383.0 if msg.type == 'pitchwheel' else None
        elif self.attribute == 'aftertouch':
            return msg.value / 127.0 if msg.type == 'aftertouch' else None
        elif self.attribute == 'poly_pressure':
            return msg.value / 127.0 if msg.type == 'polytouch' else None
        elif self.attribute.startswith('cc'):
            cc_number = int(self.attribute[2:])
            return msg.value / 127.0 if msg.type == 'control_change' and msg.control == cc_number else None
        return None

    def finalize_attribute(self, attribute_values, time_points, note_counts, modulation_data):
        if not time_points:
            return np.zeros(self.frame_count)

        self.time_points = np.array(time_points)
        attribute_values = np.array(attribute_values)

        if len(self.time_points) < 2:
            self.time_points = np.array([0, self.total_time])
            attribute_values = np.array([0, 0])

        frame_times = np.linspace(0, self.total_time, self.frame_count)

        if self.attribute == 'density':
            max_count = np.max(note_counts)
            return note_counts / max_count if max_count > 0 else np.zeros(self.frame_count)
        elif self.attribute.startswith('cc'):
            cc_number = int(self.attribute[2:])
            return self.interpolate_cc_data(cc_number, frame_times)
        elif self.attribute == 'pitchbend':
            return self.interpolate_data(self.pitchbend_data, frame_times)
        elif self.attribute == 'aftertouch':
            return self.interpolate_data(self.aftertouch_data, frame_times)
        elif self.attribute == 'poly_pressure':
            return self.interpolate_data(self.poly_pressure_data, frame_times)
        elif self.attribute == 'modulation':
            if modulation_data:
                mod_times, mod_values = zip(*modulation_data)
                return np.interp(frame_times, mod_times, mod_values)
            return np.zeros(self.frame_count)
        else:
            interpolated = np.interp(frame_times, self.time_points, attribute_values)
            interpolated[note_counts == 0] = 0
            return self.apply_modulation(interpolated, frame_times, modulation_data)

    def interpolate_cc_data(self, cc_number, frame_times):
        if self.cc_data[cc_number]:
            cc_times, cc_values = zip(*self.cc_data[cc_number])
            return np.interp(frame_times, cc_times, cc_values)
        return np.zeros(self.frame_count)

    def interpolate_data(self, data, frame_times):
        if data:
            times, values = zip(*data)
            return np.interp(frame_times, times, values)
        return np.zeros(self.frame_count)

    def apply_modulation(self, interpolated, frame_times, modulation_data):
        if modulation_data and self.attribute != 'modulation':
            mod_times, mod_values = zip(*modulation_data)
            modulation = np.interp(frame_times, mod_times, mod_values)
            interpolated = interpolated * (1 + modulation * 0.2)
            return np.clip(interpolated, 0, 1)
        return interpolated

    def normalize(self):
        if self.data is not None and len(self.data) > 0:
            min_val = np.min(self.data)
            max_val = np.max(self.data)
            if max_val > min_val:
                self.data = (self.data - min_val) / (max_val - min_val)
            else:
                self.data = np.zeros_like(self.data)
        return self

    @classmethod
    def get_attribute_names(cls):
        return list(cls.ATTRIBUTE_MAP.keys())

    @classmethod
    def get_attribute_value(cls, friendly_name):
        return cls.ATTRIBUTE_MAP.get(friendly_name)