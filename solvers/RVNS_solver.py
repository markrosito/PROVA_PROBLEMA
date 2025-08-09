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
    def _generate_initial_solution(self):
        print("Generating initial greedy solution...")
        solution = {"patients": [], "nurses": []}
        
        # Prova a schedulare tutti i pazienti obbligatori
        mandatory_patients = [p for p in self.hospital.patient_dict.values() if p['mandatory']]
        for patient in sorted(mandatory_patients, key=lambda p: p['surgery_due_day']):
            # Trova il primo posto valido
            for day in range(patient['surgery_release_day'], patient['surgery_due_day'] + 1):
                # Semplificazione: prova ad assegnare alla prima sala e stanza disponibili
                room = next((r for r in self.hospital.room_dict.keys() if r not in patient.get('incompatible_room_ids', [])), None)
                ot = next(iter(self.hospital.operating_theaters_dict.keys()), None)
                if room and ot:
                    solution['patients'].append({
                        "id": patient['id'], "admission_day": day, "room": room, "operating_theater": ot
                    })
                    break # Passa al paziente successivo
        
        # Assegnazione infermieri semplificata: assegna il primo infermiere disponibile a tutte le stanze occupate
        occupied_rooms_per_shift = {}
        for p_sol in solution['patients']:
            patient = self.hospital.get_patient_by_id(p_sol['id'])
            for day_offset in range(patient['length_of_stay']):
                day = p_sol['admission_day'] + day_offset
                for shift_name in self.hospital.shift_types:
                    key = (day, shift_name)
                    if key not in occupied_rooms_per_shift: occupied_rooms_per_shift[key] = set()
                    occupied_rooms_per_shift[key].add(p_sol['room'])
        
        nurse_assignments = {}
        for (day, shift_name), rooms in occupied_rooms_per_shift.items():
            # Trova un infermiere che lavora in quel turno
            available_nurse = next((n_id for n_id, n in self.hospital.nurses_dict.items() if any(ws['day'] == day and ws['shift'] == shift_name for ws in n['working_shifts'])), None)
            if available_nurse:
                if available_nurse not in nurse_assignments: nurse_assignments[available_nurse] = []
                nurse_assignments[available_nurse].append({"day": day, "shift": shift_name, "rooms": list(rooms)})
        
        solution['nurses'] = [{"id": n_id, "assignments": assigns} for n_id, assigns in nurse_assignments.items()]
        
        return solution

    # --- 2. Strutture di Vicinato ---
    def _neighborhood_change_patient_room(self, solution):
        if not solution['patients']: return solution
        p_sol = random.choice(solution['patients'])
        patient = self.hospital.get_patient_by_id(p_sol['id'])
        
        available_rooms = [r for r in self.hospital.room_dict.keys() if r not in patient.get('incompatible_room_ids', []) and r != p_sol['room']]
        if available_rooms:
            p_sol['room'] = random.choice(available_rooms)
        return solution

    def _neighborhood_change_patient_day(self, solution):
        if not solution['patients']: return solution
        p_sol = random.choice(solution['patients'])
        patient = self.hospital.get_patient_by_id(p_sol['id'])
        
        due_day = patient.get('surgery_due_day', self.hospital.days - patient['length_of_stay'])
        new_day = random.randint(patient['surgery_release_day'], due_day)
        p_sol['admission_day'] = new_day
        return solution
        
    def _neighborhood_reschedule_unscheduled(self, solution):
        # Prova a schedulare un paziente opzionale non schedulato
        admitted_ids = {p['id'] for p in solution['patients']}
        unscheduled_optionals = [p for p_id, p in self.hospital.patient_dict.items() if not p['mandatory'] and p_id not in admitted_ids]
        
        if unscheduled_optionals:
            patient = random.choice(unscheduled_optionals)
            # Logica simile a quella iniziale per trovare un posto
            day = random.randint(patient['surgery_release_day'], self.hospital.days - patient['length_of_stay'])
            room = random.choice(list(self.hospital.room_dict.keys()))
            ot = random.choice(list(self.hospital.operating_theaters_dict.keys()))
            solution['patients'].append({"id": patient['id'], "admission_day": day, "room": room, "operating_theater": ot})
        return solution

    def _neighborhood_change_nurse_assignment(self, solution):
        """
        Operatore di vicinato: Riassegnazione Infermiere.
        Sceglie un'assegnazione infermiere-paziente e prova a cambiare il paziente
        con un altro paziente valido per lo stesso infermiere nello stesso turno.
        Modifica la soluzione "in-place".
        """
        # 1. Controllo di sicurezza: se non ci sono assegnazioni, esci
        if not solution['nurses'] or not any(nurse.get('assignments') for nurse in solution['nurses']):
            return solution

        # 2. Scegli casualmente un'assegnazione da modificare
        #    'assignment_sol' è un riferimento a un dizionario dentro la lista della soluzione -à
        nurse = random.choice([n for n in solution['nurses'] if n.get('assignments')])
        assignment_sol = random.choice(nurse['assignments'])

        available_patients = []
        
        # Estrai le informazioni necessarie dall'assegnazione scelta
        nurse_id = nurse['id']
        shift = assignment_sol['shift']
        #current_patient_id = assignment_sol.get('patient_id', None)
        
        # 3. Trova i pazienti alternativi e validi
        #    Questa è la parte più importante e richiede delle assunzioni sul tuo modello "hospital"
        
        # Assunzione 1: Esiste un metodo che restituisce gli ID dei pazienti che richiedono
        # cure in un determinato turno.
        for day in range(1, self.hospital.days + 1):
            for shift in self.hospital.shift_types:
                candidate_patients = list(self.hospital.get_all_patients_in_rooms(solution))
                # Filtra i pazienti per il giorno corrente
                filtered_patients = [c for c in candidate_patients if c[0] == day-1]  # and assignment_sol['shift'] == shift
                if filtered_patients:
                    candidate_patient_ids = [p[2]['id'] for p in filtered_patients]
                    #print(f"1. Candidate patients for nurse {nurse_id} in shift {shift}: {candidate_patient_ids}")
                  
        # Assunzione 2: Esiste un metodo per verificare se un infermiere ha le competenze
        # necessarie per un determinato paziente.                
                for p_id in candidate_patient_ids:
                    for p in list(self.hospital.occupants) + solution['patients']:
                        if p in solution['patients'] and p['id'] == p_id:
                            admission_day = p['admission_day']
                            skill_index = 3 * (day-1) - admission_day + self.hospital.shift_types.index(shift)
                            skill_levels = self.hospital.get_patient_by_id(p_id)['skill_level_required']
                            if 0 <= skill_index < len(skill_levels):
                                if self.hospital.get_nurse_by_id(nurse_id)['skill_level'] >= skill_levels[skill_index] and p_id not in available_patients:
                                    available_patients.append(p_id)
                                    print(f"2. Checking patient {p_id} for nurse {nurse_id} on day {day}, shift {shift}: skill index {skill_index}")
                                available_patients.append(p_id)
                                print(f"2. Checking patient {p_id} for nurse {nurse_id} on day {day}, shift {shift}: skill index {skill_index}")
                        elif p['id'] == p_id:
                            skill_index = 3 * (day-1) + self.hospital.shift_types.index(shift)
                            print(f"skill index: {skill_index}")
                            skill_levels = self.hospital.get_patient_by_id(p_id)['skill_level_required']
                            if 0 <= skill_index < len(skill_levels):
                                if self.hospital.get_nurse_by_id(nurse_id)['skill_level'] >= skill_levels[skill_index] and p_id not in available_patients:
                                    available_patients.append(p_id)
                                    print(f"2. Checking patient {p_id} for nurse {nurse_id} on day {day}, shift {shift}: skill index {skill_index}")
                                
        assignment_sol['patient_id'] = random.choice(available_patients)
        available_patients = []  # Reset per il prossimo ciclo
        
        # 4. Se esistono alternative valide, scegline una e modifica la soluzione
        # Modifica direttamente l'oggetto 'assignment_sol', che aggiorna la soluzione generale

        # 5. Restituisci la soluzione (modificata o meno)
        return solution

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