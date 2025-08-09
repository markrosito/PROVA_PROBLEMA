import json
import os
#from solvers.RVNS_solver import *

class Loader:

    def __init__(self, path: str):
        self.path = path
        self.data = self.get_data()
        self.days = self.get_days()
        self.skill_levels = self.get_skill_levels()
        self.shift_types = self.get_shift_types()
        self.age_groups = self.get_age_groups()
        self.occupants = self.get_occupants()
        self.patients = self.get_patients()
        self.surgeons = self.get_surgeons()
        self.operating_theaters = self.get_operating_theaters()
        self.rooms = self.get_rooms()
        self.nurses = self.get_nurses()
        
        self.occupant_count = len(self.occupants)
        self.patient_count = len(self.patients)
        self.room_count = len(self.rooms)
        self.nurse_count = len(self.nurses)
        self.surgeon_count = len(self.surgeons)
        
        self.occupant_dict = {occupant['id']: occupant for occupant in self.occupants} #occupants as dict with id as key
        self.room_dict = {room['id']: room for room in self.rooms} #rooms as dict with id as key
        self.nurses_dict = {nurse['id']: nurse for nurse in self.nurses} #nurses as dict with id as key
        self.surgeon_dict = {surgeon['id']: surgeon for surgeon in self.surgeons}
        self.patient_dict = {patient['id']: patient for patient in self.patients} #patients as dict with id as key
        self.operating_theaters_dict = {ot['id']: ot for ot in self.operating_theaters} 
        
    def get_data(self):
        # Load data from the file
        with open(self.path, 'r') as file:
            data = json.load(file)
        return data
    
    def get_days(self):
        # Return the number of days in the data
        return self.data['days']
    
    def get_skill_levels(self):
        # Return the skill levels of the occupants
        return self.data['skill_levels']
    
    def get_shift_types(self):
        # Return the shift types
        return self.data['shift_types']
    
    def get_age_groups(self):
        # Return the age groups of the patients
        return self.data['age_groups']

    def get_occupants(self):
        # Return the list of occupants
        return self.data['occupants']
    
    def get_occupant(self, id: str):
        # Return the occupant with the given ID
        for occupant in self.data['occupants']:
            if occupant['id'] == id:
                return occupant
        return None
    
    def get_surgeons(self):
        # Return the list of surgeons
        return self.data['surgeons']
    
    def get_operating_theaters(self):
        # Return the list of operating theaters
        return self.data['operating_theaters']
        
    def get_rooms(self):
        # Return the list of rooms
        return self.data['rooms']
    
    def get_room(self, id: str):
        # Return the room with the given ID
        for room in self.data['rooms']:
            if room['id'] == id:
                return room
        return None
    
    def get_nurses(self):
        # Return the list of nurses
        return self.data['nurses']

    def get_patients(self):
        # Return the list of patients
        return self.data['patients']
    
    def get_patient(self, id: str):
        # Return the patient with the given ID
        for patient in self.data['patients']:
            if patient['id'] == id:
                return patient
        return None


    # === REPLACE THE OLD HELPER METHODS WITH THIS CORRECTED BLOCK ===

    def get_patient_by_id(self, patient_id: str):
        """
        Retrieves a patient or occupant's data dictionary by their ID using the pre-built dictionaries.
        """
        if patient_id in self.patient_dict:
            return self.patient_dict[patient_id]
        if patient_id in self.occupant_dict:
            return self.occupant_dict[patient_id]
        return None

    def get_solution_patient_by_id(self, solution: dict, patient_id: str):
        """
        Finds a patient's specific data within the solution dictionary.
        """
        for p_sol in solution.get('patients', []):
            if p_sol['id'] == patient_id:
                return p_sol
        return None

    def get_all_patients_in_rooms(self, solution: dict):
        """
        Generator function that yields all patients (admitted and occupants)
        in a room for each day of their stay.
        Yields: (day, room_id, patient_object)
        """
        # Process newly admitted patients from the solution
        for p_sol in solution.get('patients', []):
            patient = self.get_patient_by_id(p_sol['id'])
            if not patient:
                continue

            admission_day = p_sol.get('admission_day')
            room_id = p_sol.get('room')
            
            if admission_day is not None and room_id is not None:
                # Create a copy to avoid modifying the original data
                patient_copy = patient.copy()
                patient_copy['admission_day'] = admission_day
                for day_offset in range(patient_copy['length_of_stay']):
                    current_day = admission_day + day_offset
                    if current_day < self.days:
                        yield (current_day, room_id, patient_copy)

        # Process occupants who were already in the hospital
        # This is the line that caused the error. We now iterate over the list directly.
        for occupant in self.occupants:
            room_id = occupant.get('room_id')
            if room_id:
                for day_offset in range(occupant['length_of_stay']):
                    current_day = day_offset
                    if current_day < self.days:
                         yield (current_day, room_id, occupant)

    def get_nurse_assignments(self, solution: dict, id_only=False):
        """
        Creates a quick-lookup dictionary for nurse assignments from the solution.
        Uses the pre-built nurses_dict for efficiency.
        """
        assignments = {}
        for n_sol in solution.get('nurses', []):
            # Use the nurses_dict for quick lookup
            nurse = self.nurses_dict.get(n_sol['id'])
            if not nurse:
                continue
            for assignment in n_sol.get('assignments', []):
                for room_id in assignment['rooms']:
                    key = (assignment['day'], assignment['shift'], room_id)
                    assignments[key] = nurse['id'] if id_only else nurse['skill_level']
        return assignments

    def get_nurse_max_load(self, nurse_id: str, day: int, shift_name: str):
        """
        Gets the maximum workload for a specific nurse on a given day and shift.
        """
        nurse = self.nurses_dict.get(nurse_id)
        if not nurse:
            return 0
        for shift in nurse.get('working_shifts', []):
            if shift['day'] == day and shift['shift'] == shift_name:
                return shift.get('max_load', 0)
        return 0
    
    def get_nurse_by_id(self, nurse_id: str):
        """
        Retrieves a nurse's data dictionary by their ID using the pre-built nurses_dict.
        """
        return self.nurses_dict.get(nurse_id, None)

class Occupant:

    def __init__(self, 
                 id: str,
                 gender: str,
                 age_group: str,
                 length_of_stay: int,
                 workload_produced: list,
                 skill_level_required: list,
                 room_id: str):
        
       
        self.id = id
        self.gender = gender
        self.age_group = age_group
        self.length_of_stay = length_of_stay
        self.workload_produced = workload_produced
        self.skill_level_required = skill_level_required
        self.room_id = room_id


class Patient:
    def __init__(self,
                 id: str,
                 mandatory: bool,
                 gender: str,
                 age_group: str,
                 length_of_stay: int,
                 surgery_release_day: int,
                 surgery_duration: int,
                 surgeon_id: str,
                 incompatible_room_ids: list,
                 workload_produced: list,
                 skill_level_required: list,
                 surgery_due_day: int = None):
        
        self.id = id
        self.mandatory = mandatory
        self.gender = gender
        self.age_group = age_group
        self.length_of_stay = length_of_stay
        self.surgery_release_day = surgery_release_day
        self.surgery_due_day = surgery_due_day # Optional, can be None
        self.surgery_duration = surgery_duration
        self.surgeon_id = surgeon_id
        self.incompatible_room_ids = incompatible_room_ids
        self.workload_produced = workload_produced
        self.skill_level_required = skill_level_required

class Surgeon:
    def __init__(self, id: str, max_surgery_time: list):
        self.id = id
        self.max_surgery_time = max_surgery_time

class OperatingTheater:
    def __init__(self, id: str, availability: list):
        self.id = id
        self.availability = availability

class Room:
    def __init__(self, id: str, capacity: int):
        self.id = id
        self.capacity = capacity
        self.occupants = []

class Nurse:
    def __init__(self, id: str, skill_level: str, working_shifts: list):
        self.id = id
        self.skill_level = skill_level
        self.working_shifts = working_shifts
