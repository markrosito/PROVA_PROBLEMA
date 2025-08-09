
# A large constant to heavily penalize hard constraint violations
HARD_VIOLATION_PENALTY = 1000000

# ==================================
# Patient Admission Scheduling (PAS) Constraints
# ==================================

def h1_no_gender_mix(solution, hospital):
    """H1: No gender mix in rooms."""
    violations = 0
    room_daily_genders = {}
    all_patients_in_rooms = hospital.get_all_patients_in_rooms(solution)
    for day, room_id, patient in all_patients_in_rooms:
        key = (day, room_id)
        if key not in room_daily_genders:
            room_daily_genders[key] = set()
        room_daily_genders[key].add(patient['gender'])
    for genders in room_daily_genders.values():
        if len(genders) > 1:
            violations += 1
    return violations * HARD_VIOLATION_PENALTY

# In solvers/constraints.py

def h2_compatible_rooms(solution, hospital):
    """H2: Patients must be in compatible rooms."""
    violations = 0
    for p_sol in solution.get('patients', []):
        patient = hospital.get_patient_by_id(p_sol['id'])
        # Add a check to ensure the patient object was found before using it.
        if patient: 
            if 'incompatible_room_ids' in patient and p_sol['room'] in patient['incompatible_room_ids']:
                violations += 1
        # If patient is None, we simply ignore it, as it's not a valid entry.

    return violations * HARD_VIOLATION_PENALTY


def h7_room_capacity(solution, hospital):
    """H7: Room capacity cannot be exceeded."""
    violations = 0
    room_occupancy = {} # {(day, room_id): count}
    all_patients_in_rooms = hospital.get_all_patients_in_rooms(solution)

    for day, room_id, _ in all_patients_in_rooms:
        key = (day, room_id)
        room_occupancy[key] = room_occupancy.get(key, 0) + 1

    for (day, r_id), occupancy in room_occupancy.items():
        # --- THIS IS THE FIX ---
        # Use the room_dict for string-based lookup instead of the room list.
        if occupancy > hospital.room_dict[r_id]['capacity']:
            violations += 1
            
    return violations * HARD_VIOLATION_PENALTY

def s1_mixed_age_penalty(solution, hospital, weight):
    """S1: Minimize age difference in rooms."""
    cost = 0
    room_daily_ages = {}
    age_map = {age: i for i, age in enumerate(hospital.age_groups)}
    all_patients_in_rooms = hospital.get_all_patients_in_rooms(solution)
    for day, room_id, patient in all_patients_in_rooms:
        key = (day, room_id)
        if key not in room_daily_ages:
            room_daily_ages[key] = []
        room_daily_ages[key].append(age_map[patient['age_group']])
    for ages in room_daily_ages.values():
        if len(ages) > 1:
            cost += max(ages) - min(ages)
    return cost * weight

# ==================================
# Nurse-to-Room Assignment (NRA) Constraints
# ==================================

def s2_minimum_skill_level(solution, hospital, weight):
    """S2: Nurse skill level must meet patient requirements."""
    cost = 0
    nurse_assignments = hospital.get_nurse_assignments(solution)
    all_patients_in_rooms = hospital.get_all_patients_in_rooms(solution)
    for day, room_id, patient in all_patients_in_rooms:
        for shift_idx, shift_name in enumerate(hospital.shift_types):
            day_offset = day - patient.get('admission_day', 0)
            req_idx = day_offset * 3 + shift_idx
            if 0 <= req_idx < len(patient['skill_level_required']):
                required_skill = patient['skill_level_required'][req_idx]
                key = (day, shift_name, room_id)
                if key in nurse_assignments and nurse_assignments[key] < required_skill:
                    cost += required_skill - nurse_assignments[key]
    return cost * weight


def s3_continuity_of_care(solution, hospital, weight):
    """S3: Minimize the number of distinct nurses per patient."""
    cost = 0
    nurse_assignments = hospital.get_nurse_assignments(solution, id_only=True)
    
    admitted_patient_ids = [p['id'] for p in solution.get('patients', [])]
    occupant_ids = [o['id'] for o in hospital.occupants]
    all_patient_ids = admitted_patient_ids + occupant_ids

    for p_id in set(all_patient_ids):
        distinct_nurses = set()
        patient = hospital.get_patient_by_id(p_id)
        
        # --- THIS IS THE FIX ---
        # Add a check to ensure the patient object exists before using it.
        if not patient:
            continue # Skip to the next ID if this one isn't valid

        p_sol = hospital.get_solution_patient_by_id(solution, p_id)
        
        admission_day = p_sol.get('admission_day', 0) if p_sol else 0
        room_id = p_sol.get('room') if p_sol else patient.get('room_id')

        if not room_id: 
            continue

        for day_offset in range(patient['length_of_stay']):
            day = admission_day + day_offset
            for shift_name in hospital.shift_types:
                key = (day, shift_name, room_id)
                if key in nurse_assignments:
                    distinct_nurses.add(nurse_assignments[key])
        
        if distinct_nurses:
            cost += len(distinct_nurses)
            
    return cost * weight

def s4_maximum_workload(solution, hospital, weight):
    """S4: Nurse workload should not exceed maximum."""
    cost = 0
    nurse_workload = {}
    nurse_assignments = hospital.get_nurse_assignments(solution, id_only=True)
    all_patients_in_rooms = hospital.get_all_patients_in_rooms(solution)
    for day, room_id, patient in all_patients_in_rooms:
        for shift_idx, shift_name in enumerate(hospital.shift_types):
            key = (day, shift_name, room_id)
            if key in nurse_assignments:
                nurse_id = nurse_assignments[key]
                day_offset = day - patient.get('admission_day', 0)
                workload_idx = day_offset * 3 + shift_idx
                if 0 <= workload_idx < len(patient['workload_produced']):
                    workload = patient['workload_produced'][workload_idx]
                    n_key = (nurse_id, day, shift_name)
                    nurse_workload[n_key] = nurse_workload.get(n_key, 0) + workload
    for (n_id, day, shift), load in nurse_workload.items():
        max_load = hospital.get_nurse_max_load(n_id, day, shift)
        if load > max_load:
            cost += load - max_load
    return cost * weight

# ==================================
# Surgical Case Planning (SCP) Constraints
# ==================================

def h3_surgeon_overtime(solution, hospital):
    """H3: Surgeon's daily surgery time must not be exceeded."""
    violations = 0
    surgeon_load = {} # {(day, surgeon_id): total_minutes}

    for p_sol in solution.get('patients', []):
        # First, retrieve the patient object from the hospital data.
        patient = hospital.get_patient_by_id(p_sol['id'])

        # Now, check if the patient was found before using it.
        if patient:
            key = (p_sol['admission_day'], patient['surgeon_id'])
            surgeon_load[key] = surgeon_load.get(key, 0) + patient['surgery_duration']

    for (day, s_id), load in surgeon_load.items():
        # Use the new surgeon_dict for a safe and fast lookup.
        surgeon = hospital.surgeon_dict.get(s_id)
        if surgeon and load > surgeon['max_surgery_time'][day]:
            violations += 1
            
    return violations * HARD_VIOLATION_PENALTY


def h4_ot_overtime(solution, hospital):
    """H4: OT daily capacity must not be exceeded."""
    violations = 0
    ot_load = {} # {(day, ot_id): total_minutes}
    for p_sol in solution.get('patients', []):
        # --- FIX Part 1: Get the patient object ---
        patient = hospital.get_patient_by_id(p_sol['id'])

        # --- FIX Part 2: Check if patient exists before using it ---
        if patient:
            key = (p_sol['admission_day'], p_sol['operating_theater'])
            ot_load[key] = ot_load.get(key, 0) + patient['surgery_duration']

    for (day, ot_id), load in ot_load.items():
        # --- FIX Part 3: Use the new dictionary for safe lookup ---
        ot = hospital.operating_theaters_dict.get(ot_id)
        if ot and load > ot['availability'][day]:
            violations += 1
            
    return violations * HARD_VIOLATION_PENALTY

def s5_open_ots(solution, hospital, weight):
    """S5: Minimize the number of open OTs per day."""
    open_ots = {(p['admission_day'], p['operating_theater']) for p in solution.get('patients', [])}
    return len(open_ots) * weight


def s6_surgeon_transfer(solution, hospital, weight):
    """S6: Minimize the number of OTs a surgeon uses per day."""
    cost = 0
    surgeon_ots = {}  # {(day, surgeon_id): set_of_OTs}
    
    for p_sol in solution.get('patients', []):
        # --- FIX Part 1: Get the patient object ---
        patient = hospital.get_patient_by_id(p_sol['id'])

        # --- FIX Part 2: Check if the patient was found before using it ---
        if patient:
            key = (p_sol['admission_day'], patient['surgeon_id'])
            if key not in surgeon_ots:
                surgeon_ots[key] = set()
            surgeon_ots[key].add(p_sol['operating_theater'])

    for ots in surgeon_ots.values():
        if len(ots) > 1:
            cost += len(ots) - 1
            
    return cost * weight

# ==================================
# Global Constraints
# ==================================

def h5_mandatory_unscheduled(solution, hospital):
    """H5: All mandatory patients must be admitted."""
    admitted_ids = {p['id'] for p in solution.get('patients', [])}
    violations = 0
    # Iterate over the patient_dict which supports .items()
    for p_id, patient in hospital.patient_dict.items():
        if patient['mandatory'] and p_id not in admitted_ids:
            violations += 1
            
    return violations * HARD_VIOLATION_PENALTY


def h6_admission_day(solution, hospital):
    """H6: Admission must be within release and due dates."""
    violations = 0
    for p_sol in solution.get('patients', []):
        patient = hospital.get_patient_by_id(p_sol['id'])
        if patient:
            if p_sol['admission_day'] < patient['surgery_release_day']:
                violations += 1
            if patient.get('mandatory') and p_sol['admission_day'] > patient['surgery_due_day']:
                violations += 1
                
    return violations * HARD_VIOLATION_PENALTY


def s7_admission_delay(solution, hospital, weight):
    """S7: Minimize patient admission delay."""
    cost = 0
    for p_sol in solution.get('patients', []):
        patient = hospital.get_patient_by_id(p_sol['id'])
        if patient:
            delay = p_sol['admission_day'] - patient['surgery_release_day']
            if delay > 0:
                cost += delay
                
    return cost * weight

# In solvers/constraints.py


def s8_unscheduled_optional(solution, hospital, weight):
    """S8: Minimize the number of unscheduled optional patients."""
    admitted_ids = {p['id'] for p in solution.get('patients', [])}
    cost = 0

    # Iterate over the patient_dict which supports .items()
    for p_id, patient in hospital.patient_dict.items():
        if not patient.get('mandatory') and p_id not in admitted_ids:
            cost += 1
            
    return cost * weight