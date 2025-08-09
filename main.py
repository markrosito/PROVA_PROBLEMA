import os, json
from solvers.RVNS_solver import RVNS 

def main():
    # --- PASSO 1: CONFIGURAZIONE ---
    # Definisci quale istanza risolvere
    instance_name = 'test01.json'
    instance_path = os.path.join('data/ihtc2024_test_dataset', instance_name)

    # Definisci un limite di tempo per la ricerca (in secondi)
    # Per un test rapido, 10-30 secondi sono sufficienti.
    # Per una ricerca seria, aumenta a 600 (10 minuti) come da competizione.
    time_limit = 20 

    if not os.path.exists(instance_path):
        print(f"Errore: File istanza non trovato in {instance_path}")
        return

    # --- PASSO 2: INIZIALIZZAZIONE DEL SOLVER ---
    # Crea un'istanza del solver con il percorso del file e il limite di tempo.
    solver = RVNS(instance_path, time_limit_seconds=time_limit)
    
    # --- PASSO 3: ESECUZIONE DELL'ALGORITMO ---
    # Chiama il metodo solve() per avviare l'RVNS.
    # Questo metodo si occuperà di tutto:
    # 1. Creare una soluzione iniziale.
    # 2. Avviare il ciclo di ricerca (Shake e Local Search).
    # 3. Restituire la migliore soluzione trovata entro il limite di tempo.
    print(f"\n--- Avvio del Solver RVNS per {time_limit} secondi ---")
    best_solution_found = solver.solve()
    
    # --- PASSO 4: ANALISI DEI RISULTATI ---
    print("\n--- Analisi della Soluzione Finale ---")
    if not best_solution_found:
        print("Nessuna soluzione valida è stata trovata.")
    else:
        # Valuta la soluzione finale per ottenere un resoconto dettagliato
        final_cost, costs_breakdown = solver.evaluate_solution(best_solution_found)
        
        print(f"Costo Totale della Migliore Soluzione: {final_cost}")
        print("Dettaglio Costi/Violazioni Finali:")
        for constraint, value in costs_breakdown.items():
                print(f"  - {constraint}: {value}")
    
    print("-------------------------------------")
    with open('output.json', 'w') as output_file:
        json.dump(best_solution_found, output_file, indent=4)


if __name__ == '__main__':
    main()