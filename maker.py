import os
import pandas as pd
from datetime import datetime

# ==========================================
# CONFIGURATION SETTINGS
# ==========================================
TARGET_FOLDER_NAME = "ENTER_FOLDER_NAME_HERE"
OUTPUT_FILE_NAME = "generated_output.txt"

# Excel Column Headers
COL_PRIMARY_ID = "ENTER_PRIMARY_ID_COLUMN_HERE"
COL_SECONDARY_ID = "ENTER_SECONDARY_ID_COLUMN_HERE"

# Query Template
# Use {secondary_id} and {primary_id} as placeholders.
QUERY_TEMPLATE = "ENTER_YOUR_QUERY_TEMPLATE_HERE"
# ==========================================

def generate_queries():
    cwd = os.getcwd()
    target_dir = os.path.join(cwd, TARGET_FOLDER_NAME)

    if not os.path.exists(target_dir):
        print(f"[ERROR] Directory '{TARGET_FOLDER_NAME}' not found in {cwd}.")
        return

    excel_files = [f for f in os.listdir(target_dir) if f.endswith(".xlsx") or f.endswith(".xls")]
    
    if not excel_files:
        print(f"[WARNING] No Excel files found in '{TARGET_FOLDER_NAME}'.")
        return
        
    print(f"[INFO] Found {len(excel_files)} Excel file(s) to process.")

    queries_by_group = {}
    total_queries_generated = 0
    
    for file in excel_files:
        filepath = os.path.join(target_dir, file)
        print(f"[PROCESS] Reading file: {file}")
        
        try:
            df = pd.read_excel(filepath)
            
            df.columns = df.columns.str.strip()
            
            if COL_PRIMARY_ID not in df.columns or COL_SECONDARY_ID not in df.columns:
                print(f"  -> [ERROR] Missing required columns in {file}. Skipping.")
                continue
                
            df = df.dropna(subset=[COL_PRIMARY_ID, COL_SECONDARY_ID])
            
            count = 0
            for index, row in df.iterrows():
                primary_val = str(row[COL_PRIMARY_ID]).strip()
                secondary_val = str(row[COL_SECONDARY_ID]).strip()
                
                query = QUERY_TEMPLATE.format(secondary_id=secondary_val, primary_id=primary_val)
                
                if secondary_val not in queries_by_group:
                    queries_by_group[secondary_val] = []
                queries_by_group[secondary_val].append(query)
                
                count += 1
                total_queries_generated += 1
                
            print(f"  -> [SUCCESS] Extracted {count} records from {file}.")
            
        except Exception as e:
            print(f"  -> [ERROR] Failed processing {file}. Reason: {e}")
            
    if queries_by_group:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        master_run_folder = os.path.join(target_dir, f"Run_{timestamp_str}")
        os.makedirs(master_run_folder, exist_ok=True)
        
        print(f"\n[INFO] Creating main timestamp folder: {f'Run_{timestamp_str}'}")
        print(f"[INFO] Saving queries for {len(queries_by_group)} unique groups...")
        
        for secondary_val, queries in queries_by_group.items():
            safe_folder_name = "".join([c for c in secondary_val if c.isalnum() or c in ('-', '_')]).strip()
            
            group_folder_path = os.path.join(master_run_folder, safe_folder_name)
            os.makedirs(group_folder_path, exist_ok=True)
            
            output_path = os.path.join(group_folder_path, OUTPUT_FILE_NAME)
            
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    for q in queries:
                        f.write(q + "\n\n") 
                print(f"  -> [SAVED] {len(queries)} queries to '{safe_folder_name}/{OUTPUT_FILE_NAME}'")
            except Exception as e:
                print(f"  -> [ERROR] Failed to write file for {safe_folder_name}: {e}")
                
        print(f"\n[SUCCESS] Operation complete. Processed {total_queries_generated} total queries across {len(queries_by_group)} folders inside Run_{timestamp_str}.")
    else:
        print("\n[INFO] No queries were generated.")

if __name__ == "__main__":
    generate_queries()

