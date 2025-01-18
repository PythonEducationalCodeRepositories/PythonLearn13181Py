# PythonLearn13181Py
python code course 


import csv
from connection import get_session, get_sentence_embedding

def create_nodes_and_relationships(tx, row):
    # Generate embeddings for each field
    embedding_site = get_sentence_embedding(row['At what site did this occur or what site is responsible?'])
    embedding_location = get_sentence_embedding(row['Site Location'])
    embedding_company = get_sentence_embedding(row['At what Company did this occur or what J&J Company is responsible?'])
    embedding_sector = get_sentence_embedding(row['Sector'])
    embedding_what_happened = get_sentence_embedding(row['What Happened?'])
    embedding_cause = get_sentence_embedding(row['What do you think caused this Incident?'])
    embedding_actions_taken = get_sentence_embedding(row['What action(s) did you take?'])
    embedding_suggestions = get_sentence_embedding(row['What action(s) do you suggest?'])
    embedding_business_unit = get_sentence_embedding(row['EH&S Business Unit'])
    embedding_sub_business_unit = get_sentence_embedding(row['EH&S Sub-Business Unit'])

    # Cypher query to create nodes and relationships
    query = """
    MERGE (site:Site {
        name: $site_name,
        location: $site_location,
        embedding_name: $embedding_site,
        embedding_location: $embedding_location
    })
    MERGE (company:Company {
        name: $company_name,
        sector: $sector,
        embedding_name: $embedding_company,
        embedding_sector: $embedding_sector
    })
    MERGE (incident:Incident {
        what_happened: $what_happened,
        cause: $cause,
        embedding_what_happened: $embedding_what_happened,
        embedding_cause: $embedding_cause
    })
    MERGE (action:Action {
        actions_taken: $actions_taken,
        suggestions: $suggestions,
        embedding_actions_taken: $embedding_actions_taken,
        embedding_suggestions: $embedding_suggestions
    })
    MERGE (business_unit:BusinessUnit {
        business_unit: $business_unit,
        sub_business_unit: $sub_business_unit,
        embedding_business_unit: $embedding_business_unit,
        embedding_sub_business_unit: $embedding_sub_business_unit
    })
    MERGE (legal_entity:LegalEntity {
        t_mrc: $t_mrc,
        t_legal_entity: $t_legal_entity
    })
    MERGE (incident)-[:OCCURRED_AT]->(site)
    MERGE (incident)-[:RESPONSIBLE_FOR]->(company)
    MERGE (company)-[:BELONGS_TO]->(business_unit)
    MERGE (site)-[:MANAGED_BY]->(business_unit)
    MERGE (incident)-[:HAS_ACTION]->(action)
    MERGE (incident)-[:LINKED_TO]->(legal_entity)
    """
    tx.run(query,
           site_name=row['At what site did this occur or what site is responsible?'],
           site_location=row['Site Location'],
           embedding_site=embedding_site,
           embedding_location=embedding_location,
           company_name=row['At what Company did this occur or what J&J Company is responsible?'],
           sector=row['Sector'],
           embedding_company=embedding_company,
           embedding_sector=embedding_sector,
           what_happened=row['What Happened?'],
           cause=row['What do you think caused this Incident?'],
           embedding_what_happened=embedding_what_happened,
           embedding_cause=embedding_cause,
           actions_taken=row['What action(s) did you take?'],
           suggestions=row['What action(s) do you suggest?'],
           embedding_actions_taken=embedding_actions_taken,
           embedding_suggestions=embedding_suggestions,
           business_unit=row['EH&S Business Unit'],
           sub_business_unit=row['EH&S Sub-Business Unit'],
           embedding_business_unit=embedding_business_unit,
           embedding_sub_business_unit=embedding_sub_business_unit,
           t_mrc=row['T_MRC'],
           t_legal_entity=row['T_LEGAL_ENTITY'])

def add_data_to_neo4j(file_path):
    with get_session() as session:
        with open(file_path, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                session.write_transaction(create_nodes_and_relationships, row)

# Call the function with the CSV file path
add_data_to_neo4j('path_to_your_csv_file.csv')
