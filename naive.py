import numpy as np
import random
import json

# -----------------------------
# PARAMETRI DEL PROBLEMA
# -----------------------------

NUM_PAZIENTI = 10
NUM_STANZE = 5
NUM_SALE = 3
NUM_INFERMIERI = 6
NUM_TURNI = 3
NUM_GIORNI = 5

# -----------------------------
# FUNZIONI DI SUPPORTO
# -----------------------------

def costo_soluzione(soluzione):
    """Calcola il costo della soluzione basato su penalità per vincoli soft"""
    costo = 0
    
    for s in range(NUM_STANZE):
        for t in range(NUM_TURNI):
            if np.sum(soluzione["stanze"][:, t, s]) > 2:
                costo += 10  
    
    for n in range(NUM_INFERMIERI):
        workload = np.sum(soluzione["infermiere"][n, :, :])
        if workload > (NUM_TURNI * NUM_STANZE) / NUM_INFERMIERI:
            costo += 5  

    return costo

def soluzione_iniziale():
    """Genera una soluzione iniziale casuale"""
    sol = {
        "stanze": np.zeros((NUM_PAZIENTI, NUM_TURNI, NUM_STANZE), dtype=int),
        "sale_operatorie": np.zeros((NUM_PAZIENTI, NUM_GIORNI, NUM_SALE), dtype=int),
        "infermiere": np.zeros((NUM_INFERMIERI, NUM_TURNI, NUM_STANZE), dtype=int),
    }
    
    for p in range(NUM_PAZIENTI):
        for t in range(NUM_TURNI):
            stanza = random.randint(0, NUM_STANZE - 1)
            sol["stanze"][p, t, stanza] = 1

    for p in range(NUM_PAZIENTI):
        giorno = random.randint(0, NUM_GIORNI - 1)
        sala = random.randint(0, NUM_SALE - 1)
        sol["sale_operatorie"][p, giorno, sala] = 1

    for n in range(NUM_INFERMIERI):
        for t in range(NUM_TURNI):
            stanza = random.randint(0, NUM_STANZE - 1)
            sol["infermiere"][n, t, stanza] = 1
    
    return sol

def genera_vicinato(soluzione, k):
    """Genera una soluzione vicina modificando casualmente k elementi"""
    nuova_sol = soluzione.copy()
    
    for _ in range(k):
        p = random.randint(0, NUM_PAZIENTI - 1)
        t = random.randint(0, NUM_TURNI - 1)
        s = random.randint(0, NUM_STANZE - 1)
        nuova_sol["stanze"][p, t, :] = 0  
        nuova_sol["stanze"][p, t, s] = 1  
    
    return nuova_sol

# -----------------------------
# RVNS ALGORITHM
# -----------------------------

def RVNS(max_iter=100, max_k=5):
    """Implementazione della metaeuristica Reduced Variable Neighborhood Search (RVNS)"""
    soluzione = soluzione_iniziale()
    costo_migliore = costo_soluzione(soluzione)

    for iterazione in range(max_iter):
        k = 1
        while k <= max_k:
            nuova_soluzione = genera_vicinato(soluzione, k)
            nuovo_costo = costo_soluzione(nuova_soluzione)
            
            if nuovo_costo < costo_migliore:
                soluzione = nuova_soluzione
                costo_migliore = nuovo_costo
                k = 1  
            else:
                k += 1  

        print(f"Iterazione {iterazione}: Costo migliore = {costo_migliore}")
    
    return soluzione

# -----------------------------
# SALVATAGGIO DELLA SOLUZIONE
# -----------------------------

def salva_soluzione(soluzione, filename="soluzione_ihtc2024.json"):
    """Salva la soluzione in un file JSON"""
    soluzione_json = {
        "stanze": soluzione["stanze"].tolist(),
        "sale_operatorie": soluzione["sale_operatorie"].tolist(),
        "infermiere": soluzione["infermiere"].tolist()
    }
    with open(filename, "w") as file:
        json.dump(soluzione_json, file, indent=4)
    print(f"Soluzione salvata in {filename}")

# -----------------------------
# ESECUZIONE
# -----------------------------

soluzione_finale = RVNS(max_iter=50, max_k=3)
salva_soluzione(soluzione_finale)
