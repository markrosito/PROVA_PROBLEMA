from instances.Hospital import *
import random
import copy
import time  
from solvers import constraints

class PenaltyWeights:
    """
    Loads and provides access to the penalty weights from the instance file.
    """
    def __init__(self, hospital_instance: Loader):
        self.hospital = hospital_instance
        self.weights = self.hospital.data['weights']
        self.room_mixed_age = self.weights['room_mixed_age']
        self.room_nurse_skill = self.weights['room_nurse_skill']
        self.continuity_of_care = self.weights['continuity_of_care']
        self.nurse_eccessive_workload = self.weights['nurse_eccessive_workload']
        self.open_operating_theater = self.weights['open_operating_theater']
        self.surgeon_transfer = self.weights['surgeon_transfer']
        self.patient_delay = self.weights['patient_delay']
        self.unscheduled_optional = self.weights['unscheduled_optional']


class PAS(PenaltyWeights):
    """
    Wrapper for Patient Admission Scheduling (PAS) constraints.
    """
    def __init__(self, hospital_instance: Loader):
        super().__init__(hospital_instance)

    def h1_no_gender_mix(self, solution: dict):
        return constraints.h1_no_gender_mix(solution, self.hospital)

    def h2_compatible_rooms(self, solution: dict):
        return constraints.h2_compatible_rooms(solution, self.hospital)

    def h7_room_capacity(self, solution: dict):
        return constraints.h7_room_capacity(solution, self.hospital)

    def s1_mixed_age_penalty(self, solution: dict):
        return constraints.s1_mixed_age_penalty(solution, self.hospital, self.room_mixed_age)


class NRA(PenaltyWeights):
    """
    Wrapper for Nurse-to-Room Assignment (NRA) constraints.
    """
    def __init__(self, hospital_instance: Loader):
        super().__init__(hospital_instance)

    def s2_minimum_skill_level(self, solution: dict):
        return constraints.s2_minimum_skill_level(solution, self.hospital, self.room_nurse_skill)

    def s3_continuity_of_care(self, solution: dict):
        return constraints.s3_continuity_of_care(solution, self.hospital, self.continuity_of_care)

    def s4_maximum_workload(self, solution: dict):
        return constraints.s4_maximum_workload(solution, self.hospital, self.nurse_eccessive_workload)


class SCP(PenaltyWeights):
    """
    Wrapper for Surgical Case Planning (SCP) constraints.
    """
    def __init__(self, hospital_instance: Loader):
        super().__init__(hospital_instance)

    def h3_surgeon_overtime(self, solution: dict):
        return constraints.h3_surgeon_overtime(solution, self.hospital)

    def h4_ot_overtime(self, solution: dict):
        return constraints.h4_ot_overtime(solution, self.hospital)

    def s5_open_ots(self, solution: dict):
        return constraints.s5_open_ots(solution, self.hospital, self.open_operating_theater)

    def s6_surgeon_transfer(self, solution: dict):
        return constraints.s6_surgeon_transfer(solution, self.hospital, self.surgeon_transfer)

class GlobalPenalty(PenaltyWeights):
    """
    Wrapper for global and cross-problem constraints.
    """
    def __init__(self, hospital_instance: Loader):
        super().__init__(hospital_instance)

    def h5_mandatory_unscheduled(self, solution: dict):
        return constraints.h5_mandatory_unscheduled(solution, self.hospital)

    def h6_admission_day(self, solution: dict):
        return constraints.h6_admission_day(solution, self.hospital)

    def s7_admission_delay(self, solution: dict):
        return constraints.s7_admission_delay(solution, self.hospital, self.patient_delay)

    def s8_unscheduled_optional(self, solution: dict):
        return constraints.s8_unscheduled_optional(solution, self.hospital, self.unscheduled_optional)

class RVNS:
    """
    Main solver class that orchestrates the RVNS algorithm and solution evaluation.
    """
    def __init__(self, path: str, time_limit_seconds=60):
        print(f"Initializing solver for instance: {path}")
        self.hospital = Loader(path) # Assicurati che il nome della classe sia Loader
        
        # Inizializza i gestori dei vincoli
        self.pas_penalties = PAS(self.hospital)
        self.nra_penalties = NRA(self.hospital)
        self.scp_penalties = SCP(self.hospital)
        self.global_penalties = GlobalPenalty(self.hospital)

        # Parametri dell'algoritmo
        self.time_limit = time_limit_seconds
        self.start_time = None
        self.best_solution = {}
        self.best_cost = float('inf')
        
        # Definisci le strutture di vicinato
        self.neighborhoods = [
            self._neighborhood_change_patient_room,
            self._neighborhood_change_patient_day,
            self._neighborhood_reschedule_unscheduled,
            self._neighborhood_change_nurse_assignment,
            self._neighborhood_change_patient_ot
        ]
        self.k_max = len(self.neighborhoods)

    def evaluate_solution(self, solution: dict):
        """
        Calculates the total cost of a solution by calling the wrapper methods.
        """
        costs = {
            # PAS Constraints
            'H1': self.pas_penalties.h1_no_gender_mix(solution),
            'H2': self.pas_penalties.h2_compatible_rooms(solution),
            'H7': self.pas_penalties.h7_room_capacity(solution),
            'S1': self.pas_penalties.s1_mixed_age_penalty(solution),

            # NRA Constraints
            'S2': self.nra_penalties.s2_minimum_skill_level(solution),
            'S3': self.nra_penalties.s3_continuity_of_care(solution),
            'S4': self.nra_penalties.s4_maximum_workload(solution),
            
            # SCP Constraints
            'H3': self.scp_penalties.h3_surgeon_overtime(solution),
            'H4': self.scp_penalties.h4_ot_overtime(solution),
            'S5': self.scp_penalties.s5_open_ots(solution),
            'S6': self.scp_penalties.s6_surgeon_transfer(solution),

            # Global Constraints
            'H5': self.global_penalties.h5_mandatory_unscheduled(solution),
            'H6': self.global_penalties.h6_admission_day(solution),
            'S7': self.global_penalties.s7_admission_delay(solution),
            'S8': self.global_penalties.s8_unscheduled_optional(solution),
        }
        total_cost = sum(costs.values())
        return total_cost, costs


    # --- 1. Generazione della Soluzione Iniziale (Greedy) ---
    # In RVNS_solver.py

    def _generate_initial_solution(self):
        """
        Genera una soluzione iniziale valida, provando a schedulare prima tutti i pazienti
        obbligatori e poi aggiungendo avidamente i pazienti opzionali.
        """
        print("Generating enhanced initial solution...")
        solution = {"patients": [], "nurses": []}

        # Ordina i pazienti: prima gli obbligatori (per data di scadenza), poi gli opzionali (per data di rilascio)
        mandatory_patients = sorted([p for p in self.hospital.patient_dict.values() if p.get('mandatory')], 
                                    key=lambda p: p.get('surgery_due_day', float('inf')))
        optional_patients = sorted([p for p in self.hospital.patient_dict.values() if not p.get('mandatory')], 
                                key=lambda p: p['surgery_release_day'])
        
        patients_to_schedule = mandatory_patients + optional_patients

        for patient in patients_to_schedule:
            found_valid_assignment = False
            
            # Definisci il range di giorni possibili
            release_day = patient['surgery_release_day']
            # Per gli opzionali, il due_day non esiste. Usiamo un limite ragionevole.
            due_day = patient.get('surgery_due_day', self.hospital.days - patient['length_of_stay'])
            
            # Assicurati che il range sia valido
            if release_day > due_day:
                continue
                
            days_range = range(release_day, due_day + 1)
            rooms_to_try = [r['id'] for r in self.hospital.rooms if r['id'] not in patient.get('incompatible_room_ids', [])]
            ots_to_try = [ot['id'] for ot in self.hospital.operating_theaters]

            random.shuffle(rooms_to_try)
            random.shuffle(ots_to_try)
            
            for day in days_range:
                for room_id in rooms_to_try:
                    for ot_id in ots_to_try:
                        
                        temp_solution = copy.deepcopy(solution)
                        temp_solution['patients'].append({
                            "id": patient['id'],
                            "admission_day": day,
                            "room": room_id,
                            "operating_theater": ot_id
                        })
                        
                        # Controlla se la nuova assegnazione paziente viola i vincoli hard
                        if not constraints.h1_no_gender_mix(temp_solution, self.hospital) and \
                        not constraints.h2_compatible_rooms(temp_solution, self.hospital) and \
                        not constraints.h7_room_capacity(temp_solution, self.hospital) and \
                        not constraints.h3_surgeon_overtime(temp_solution, self.hospital) and \
                        not constraints.h4_ot_overtime(temp_solution, self.hospital) and \
                        not constraints.h6_admission_day(temp_solution, self.hospital):
                            
                            solution = temp_solution
                            found_valid_assignment = True
                            break
                    if found_valid_assignment:
                        break
                if found_valid_assignment:
                    break
            
            if not found_valid_assignment and patient['mandatory']:
                print(f"Warning: Could not find a valid assignment for mandatory patient {patient['id']}")

        # 2. Assegnazione Iniziale Infermieri (INVARIATA)
        nurse_assignments_by_id = {}
        all_patients_in_rooms = self.hospital.get_all_patients_in_rooms(solution) # Ricrea la mappa per l'assegnazione
        
        # Raggruppa le richieste per (giorno, turno, stanza)
        room_shift_requirements = {}
        for day, room_id, patient in all_patients_in_rooms:
            for shift_idx, shift_name in enumerate(self.hospital.shift_types):
                day_offset = day - patient.get('admission_day', 0)
                req_idx = day_offset * len(self.hospital.shift_types) + shift_idx

                if 0 <= req_idx < len(patient['skill_level_required']):
                    required_skill = patient['skill_level_required'][req_idx]
                    key = (day, shift_name, room_id)
                    current_max_skill = room_shift_requirements.get(key, -1)
                    room_shift_requirements[key] = max(current_max_skill, required_skill)

        # Assegna gli infermieri
        for (day, shift_name, room_id), required_skill in room_shift_requirements.items():
            available_nurses = [
                n for n in self.hospital.nurses if 
                any(ws['day'] == day and ws['shift'] == shift_name for ws in n['working_shifts']) and
                n['skill_level'] >= required_skill
            ]
            
            if available_nurses:
                nurse_to_assign = random.choice(available_nurses)
                nurse_id = nurse_to_assign['id']
                
                if nurse_id not in nurse_assignments_by_id:
                    nurse_assignments_by_id[nurse_id] = []
                    
                found_assignment = False
                for assignment in nurse_assignments_by_id[nurse_id]:
                    if assignment['day'] == day and assignment['shift'] == shift_name:
                        if room_id not in assignment['rooms']:
                            assignment['rooms'].append(room_id)
                        found_assignment = True
                        break
                
                if not found_assignment:
                    nurse_assignments_by_id[nurse_id].append({
                        "day": day,
                        "shift": shift_name,
                        "rooms": [room_id]
                    })

        solution['nurses'] = [{"id": n_id, "assignments": assigns} for n_id, assigns in nurse_assignments_by_id.items()]

        return solution

    # --- 2. Strutture di Vicinato ---
    def _neighborhood_change_patient_room(self, solution):
        s_prime = copy.deepcopy(solution)
        if not s_prime['patients']:
            return s_prime

        p_sol = random.choice(s_prime['patients'])
        patient = self.hospital.get_patient_by_id(p_sol['id'])

        available_rooms = [
            r for r in self.hospital.room_dict.keys() 
            if r not in patient.get('incompatible_room_ids', []) 
            and r != p_sol['room']
        ]
        
        if available_rooms:
            original_room = p_sol['room']
            random.shuffle(available_rooms)
            
            for new_room in available_rooms:
                p_sol['room'] = new_room
                
                # Controlla i vincoli hard che potrebbero essere violati
                if not constraints.h1_no_gender_mix(s_prime, self.hospital) and \
                not constraints.h2_compatible_rooms(s_prime, self.hospital) and \
                not constraints.h7_room_capacity(s_prime, self.hospital):
                    # La mossa è valida, esci dal ciclo
                    return s_prime
            
            # Se nessun'altra stanza è valida, ripristina la stanza originale
            p_sol['room'] = original_room

        return s_prime

    def _neighborhood_change_patient_day(self, solution):
        s_prime = copy.deepcopy(solution)
        if not s_prime['patients']:
            return s_prime

        p_sol = random.choice(s_prime['patients'])
        patient = self.hospital.get_patient_by_id(p_sol['id'])

        due_day = patient.get('surgery_due_day', self.hospital.days - patient['length_of_stay'])
        original_day = p_sol['admission_day']

        days_to_try = list(range(patient['surgery_release_day'], due_day + 1))
        days_to_try.remove(original_day)
        random.shuffle(days_to_try)

        for new_day in days_to_try:
            p_sol['admission_day'] = new_day

            # Controlla i vincoli hard che potrebbero essere violati
            if not constraints.h6_admission_day(s_prime, self.hospital) and \
            not constraints.h3_surgeon_overtime(s_prime, self.hospital) and \
            not constraints.h4_ot_overtime(s_prime, self.hospital):
                return s_prime
            
        # Se nessun giorno è valido, ripristina il giorno originale
        p_sol['admission_day'] = original_day
        
        return s_prime
        
    def _neighborhood_reschedule_unscheduled(self, solution):
        s_prime = copy.deepcopy(solution)
        admitted_ids = {p['id'] for p in s_prime['patients']}
        unscheduled_optionals = [
            p for p_id, p in self.hospital.patient_dict.items() 
            if not p['mandatory'] and p_id not in admitted_ids
        ]
        
        if unscheduled_optionals:
            patient = random.choice(unscheduled_optionals)
            
            days_to_try = list(range(patient['surgery_release_day'], self.hospital.days - patient['length_of_stay']))
            rooms_to_try = list(self.hospital.room_dict.keys())
            ots_to_try = list(self.hospital.operating_theaters_dict.keys())
            
            random.shuffle(days_to_try)
            random.shuffle(rooms_to_try)
            random.shuffle(ots_to_try)
            
            for day in days_to_try:
                for room in rooms_to_try:
                    for ot in ots_to_try:
                        new_patient_assignment = {
                            "id": patient['id'], "admission_day": day, 
                            "room": room, "operating_theater": ot
                        }
                        
                        # Crea una copia temporanea con la nuova assegnazione
                        temp_solution = copy.deepcopy(s_prime)
                        temp_solution['patients'].append(new_patient_assignment)
                        
                        # Controlla i vincoli hard
                        if not constraints.h1_no_gender_mix(temp_solution, self.hospital) and \
                        not constraints.h2_compatible_rooms(temp_solution, self.hospital) and \
                        not constraints.h7_room_capacity(temp_solution, self.hospital) and \
                        not constraints.h3_surgeon_overtime(temp_solution, self.hospital) and \
                        not constraints.h4_ot_overtime(temp_solution, self.hospital) and \
                        not constraints.h6_admission_day(temp_solution, self.hospital):
                            
                            s_prime['patients'].append(new_patient_assignment)
                            return s_prime
                            
        return s_prime

    def _neighborhood_change_nurse_assignment(self, solution):
        s_prime = copy.deepcopy(solution)
        if not s_prime['nurses'] or len(s_prime['nurses']) < 2:
            return s_prime

        # Trova un'assegnazione casuale (assegnazione A)
        nurse_a_obj = random.choice([n for n in s_prime['nurses'] if n['assignments']])
        assignment_a = random.choice(nurse_a_obj['assignments'])
        
        candidates_b = []
        for nurse_b_obj in s_prime['nurses']:
            if nurse_b_obj['id'] == nurse_a_obj['id']:
                continue
            
            for assignment_b in nurse_b_obj['assignments']:
                if assignment_b['day'] == assignment_a['day'] and \
                assignment_b['shift'] == assignment_a['shift'] and \
                set(assignment_b['rooms']).isdisjoint(set(assignment_a['rooms'])):
                    candidates_b.append((nurse_b_obj, assignment_b))

        if candidates_b:
            nurse_b_obj, assignment_b = random.choice(candidates_b)
            
            rooms_a = assignment_a['rooms']
            rooms_b = assignment_b['rooms']
            
            assignment_a['rooms'] = rooms_b
            assignment_b['rooms'] = rooms_a
            
            # Controlla se la mossa ha violato qualche vincolo hard
            # Se la validazione passa, restituisci la soluzione modificata
            if not constraints.h1_no_gender_mix(s_prime, self.hospital) and \
            not constraints.h7_room_capacity(s_prime, self.hospital):
                return s_prime

        return solution
    
    def _neighborhood_change_patient_ot(self, solution):
        s_prime = copy.deepcopy(solution)
        if not s_prime['patients']:
            return s_prime
        
        p_sol = random.choice(s_prime['patients'])
        original_ot = p_sol['operating_theater']
        
        available_ots = [ot for ot in self.hospital.operating_theaters_dict.keys() if ot != original_ot]

        if available_ots:
            random.shuffle(available_ots)
            
            for new_ot in available_ots:
                p_sol['operating_theater'] = new_ot
                
                # Controlla i vincoli hard che potrebbero essere violati
                if not constraints.h3_surgeon_overtime(s_prime, self.hospital) and \
                not constraints.h4_ot_overtime(s_prime, self.hospital):
                    return s_prime
                    
            p_sol['operating_theater'] = original_ot
            
        return s_prime

    # --- 3. Shake ---
    def _shake(self, solution, k):
        s_prime = copy.deepcopy(solution)
        neighborhood_func = self.neighborhoods[k-1] # k è 1-based
        # Applica la mossa k volte per una perturbazione più forte
        for _ in range(k):
             s_prime = neighborhood_func(s_prime)
        return s_prime

    # --- 4. Local Search (VND) ---
    def _local_search(self, solution):
        s_current = copy.deepcopy(solution)
        cost_current, _ = self.evaluate_solution(s_current)
        
        k = 0
        while k < self.k_max:
            neighborhood_func = self.neighborhoods[k]
            # Esplora il vicinato k per trovare il *primo* miglioramento (first improvement)
            # Una strategia "best improvement" sarebbe più lenta ma potenzialmente più efficace
            s_best_neighbor = neighborhood_func(copy.deepcopy(s_current))
            cost_neighbor, _ = self.evaluate_solution(s_best_neighbor)
            
            if cost_neighbor < cost_current:
                s_current = s_best_neighbor
                cost_current = cost_neighbor
                k = 0 # Torna al primo vicinato
            else:
                k += 1
        return s_current

    # --- 5. Metodo Principale SOLVE ---
    def solve(self):
        self.start_time = time.time()
        
        # Genera una soluzione iniziale
        s = self._generate_initial_solution()
        self.best_solution = s
        self.best_cost, _ = self.evaluate_solution(self.best_solution)
        print(f"Initial solution cost: {self.best_cost}")

        iteration = 0
        while time.time() - self.start_time < self.time_limit:
            iteration += 1
            k = 1
            while k <= self.k_max:
                # 1. Shake: perturba la soluzione per uscire dall'ottimo locale
                s_prime = self._shake(self.best_solution, k)
                
                # 2. Local Search: migliora la soluzione perturbata
                s_second = self._local_search(s_prime)
                cost_second, _ = self.evaluate_solution(s_second)

                # 3. Confronta e aggiorna se necessario
                if cost_second < self.best_cost:
                    self.best_solution = s_second
                    self.best_cost = cost_second
                    print(f"Iteration {iteration}: New best solution found with cost {self.best_cost:.2f} (k={k})")
                    k = 1 # Ritorna al primo vicinato
                else:
                    k += 1
        
        print("\n--- RVNS Finished ---")
        print(f"Time limit reached. Final best cost: {self.best_cost}")
        return self.best_solution