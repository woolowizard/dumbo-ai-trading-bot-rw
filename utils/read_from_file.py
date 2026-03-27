import os
import json

def read_log_file(file_path):
    """
    Legge un file .txt e restituisce una lista di righe
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    
    return lines

def extract_execution_results(data: list):
    
    """
    Recupera dal file di log la riga della decisione i-esima
    e la normalizza per inserirla nel DB. Skippa le righe in 
    errore.
    """
    
    normalized = []

    for row in data:
        agent = row.get("agent_result")
        if not agent:
            continue

        exec_res = agent.get("execution_result")
        if not exec_res:
            continue

        decision = exec_res.get("decision", {})

        normalized.append({
            "timestamp": row.get("timestamp"),
            "ticker": row.get("ticker"),

            "status": exec_res.get("status"),

            # decision fields
            "action": decision.get("action"),
            "confidence": decision.get("confidence"),
            "size_pct": decision.get("size_pct"),
            "time_in_force": decision.get("time_in_force"),
            "reasoning": decision.get("reasoning"),

            # optional execution fields
            "order_id": exec_res.get("order_id"),
            "notional": exec_res.get("notional"),
        })

    return normalized

def extract_and_normalize(file_path):
    
    """
    Estrae le righe dal file delle decisioni e le normalizza
    Return: flatten array for DB insert
    """
    
    lines = read_log_file(file_path=file_path)[2:]
    json_lines = [json.loads(row) for row in lines]
    normalized = extract_execution_results(json_lines)
    
    return normalized
    

if __name__ == "__main__":
    
    path = "./logs/decision_log.txt"
    
    lines = read_log_file(path)[2:] # Salto le intestazioni
    
    json_lines = [json.loads(row) for row in lines]
    
    normalized = extract_execution_results(json_lines)