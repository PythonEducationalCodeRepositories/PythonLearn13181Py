import os
import pandas as pd

# ==========================================
# CONFIGURATION SETTINGS - FILL THESE IN LATER
# ==========================================
TARGET_FOLDER_NAME = "ENTER_FOLDER_NAME_HERE"
OUTPUT_FILE_NAME = "generated_queries.txt"

# Excel Column Headers
COL_R_OBJECT_ID = "ENTER_ID_COLUMN_NAME_HERE"
COL_EDMS_NUMBER = "ENTER_EDMS_COLUMN_NAME_HERE"

# Query Template
# Use {edms_no} and {r_object_id} as placeholders where the script should inject the values.
# Example: "select * from table where id='{r_object_id}' and num='{edms_no}'"
QUERY_TEMPLATE = "ENTER_YOUR_QUERY_TEMPLATE_HERE"
# ==========================================

def generate_queries():
    cwd = os.getcwd()
    target_dir = os.path.join(cwd, TARGET_FOLDER_NAME)

    # 1. Directory and File Checks
    if not os.path.exists(target_dir):
        print(f"[ERROR] Directory '{TARGET_FOLDER_NAME}' not found in {cwd}.")
        return

    excel_files = [f for f in os.listdir(target_dir) if f.endswith(".xlsx") or f.endswith(".xls")]
    
    if not excel_files:
        print(f"[WARNING] No Excel files found in '{TARGET_FOLDER_NAME}'.")
        return
        
    print(f"[INFO] Found {len(excel_files)} Excel file(s) to process.")

    all_queries = []
    
    # 2. Process each Excel file
    for file in excel_files:
        filepath = os.path.join(target_dir, file)
        print(f"[PROCESS] Reading file: {file}")
        
        try:
            df = pd.read_excel(filepath)
            
            # Clean up column names (remove leading/trailing spaces)
            df.columns = df.columns.str.strip()
            
            # Verify required columns exist
            if COL_R_OBJECT_ID not in df.columns or COL_EDMS_NUMBER not in df.columns:
                print(f"  -> [ERROR] Missing required columns in {file}. Skipping.")
                continue
                
            # Drop rows where either identifier is missing
            df = df.dropna(subset=[COL_R_OBJECT_ID, COL_EDMS_NUMBER])
            
            # 3. Generate queries row by row
            count = 0
            for index, row in df.iterrows():
                r_obj_id = str(row[COL_R_OBJECT_ID]).strip()
                edms_num = str(row[COL_EDMS_NUMBER]).strip()
                
                # Format the template with the extracted row data
                query = QUERY_TEMPLATE.format(edms_no=edms_num, r_object_id=r_obj_id)
                all_queries.append(query)
                count += 1
                
            print(f"  -> [SUCCESS] Generated {count} queries from {file}.")
            
        except Exception as e:
            print(f"  -> [ERROR] Failed processing {file}. Reason: {e}")
            
    # 4. Write all generated queries to the output text file
    if all_queries:
        output_path = os.path.join(target_dir, OUTPUT_FILE_NAME)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for q in all_queries:
                    f.write(q + "\n\n") # Double newline for readability
            print(f"\n[SUCCESS] All queries saved to: {os.path.relpath(output_path, cwd)}")
        except Exception as e:
            print(f"\n[ERROR] Failed to write output file: {e}")
    else:
        print("\n[INFO] No queries were generated.")

if __name__ == "__main__":
    generate_queries()
