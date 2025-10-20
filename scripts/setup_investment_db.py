#!/usr/bin/env python3
"""
Script to set up the banco_itau database for the investment agent.
Flattens the mutual funds JSON and loads it into PostgreSQL.
"""
import json
import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import pandas as pd
from flatten_json import flatten
from sqlalchemy import create_engine



def create_database(db_host='127.0.0.1', db_port='5432'):
    """Create the banco_itau database if it doesn't exist."""
    print("Creating database...")
    
    # Connect to PostgreSQL server (default postgres database)
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        user='postgres',
        password='postgres',
        database='postgres'
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'banco_itau'")
    exists = cursor.fetchone()
    
    if not exists:
        cursor.execute('CREATE DATABASE banco_itau')
        print("Database 'banco_itau' created")
    else:
        print("Database 'banco_itau' already exists")
    
    cursor.close()
    conn.close()


def create_user(db_host='127.0.0.1', db_port='5432'):
    """Create select_user if it doesn't exist."""
    print("Creating database user...")
    
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        user='postgres',
        password='postgres',
        database='banco_itau'
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = 'select_user'")
    exists = cursor.fetchone()
    
    if not exists:
        cursor.execute("CREATE USER select_user WITH PASSWORD 'select_user'")
        print("User 'select_user' created")
    else:
        print("User 'select_user' already exists")
    
    cursor.close()
    conn.close()


def create_table_and_load_data(json_file, db_host='127.0.0.1', db_port='5432'):
    """Create fidelity_funds table and load flattened data."""
    print(f"Loading data from {json_file}...")
    
    # Load JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    funds = data.get('funds', [])
    print(f"Found {len(funds)} funds")
    
    # Flatten all funds using flatten-json library with recursive flattening
    flattened_funds = []
    for fund in funds:
        flattened = flatten(fund)
        
        # Check if any values are still dicts or lists and flatten them recursively
        needs_reflattening = True
        while needs_reflattening:
            needs_reflattening = False
            new_flattened = {}
            for key, value in flattened.items():
                if isinstance(value, dict):
                    # Flatten this nested dict with the parent key as prefix
                    for nested_key, nested_value in flatten(value).items():
                        new_flattened[f"{key}_{nested_key}"] = nested_value
                    needs_reflattening = True
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    # Flatten list of dicts
                    for i, item in enumerate(value):
                        for nested_key, nested_value in flatten(item).items():
                            new_flattened[f"{key}_{i}_{nested_key}"] = nested_value
                    needs_reflattening = True
                else:
                    new_flattened[key] = value
            flattened = new_flattened
        
        flattened_funds.append(flattened)
    
    # Convert to DataFrame using pandas
    df = pd.DataFrame(flattened_funds)
    print(f"Flattened to {len(df.columns)} columns")
    
    # Use SQLAlchemy for everything to avoid connection/transaction conflicts
    engine = create_engine(f'postgresql://postgres:postgres@{db_host}:{db_port}/banco_itau')
    
    # pandas handles the table creation and data insertion
    print("Creating table and inserting data...")
    df.to_sql('fidelity_funds', engine, if_exists='replace', index=False)
    print(f"Inserted {len(df)} funds using pandas")
    
    # Now grant permissions using psycopg2
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        user='postgres',
        password='postgres',
        database='banco_itau'
    )
    cursor = conn.cursor()
    cursor.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO select_user")
    conn.commit()
    print("Granted SELECT permissions to select_user")
    
    cursor.close()
    conn.close()
    engine.dispose()
    print("\nDatabase setup complete!")


def main():
    """Main setup function."""
    print("=" * 60)
    print("Investment Agent Database Setup")
    print("=" * 60)
    
    # Get environment variables for database connection
    db_host = os.getenv('DB_HOST', '127.0.0.1')
    db_port = os.getenv('DB_PORT', '5432')
    
    # Path to the mutual funds JSON file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, 'mutual_funds_scraper', 'mutual_funds.json')
    
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found!")
        return
    
    try:
        create_database(db_host, db_port)
        create_user(db_host, db_port)
        create_table_and_load_data(json_file, db_host, db_port)
    except psycopg2.Error as e:
        print(f"\nDatabase error: {e}")
        print("\nMake sure PostgreSQL is installed and running:")
        print("  sudo service postgresql start")
        print("\nAnd that you have a postgres superuser with password 'postgres'")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == '__main__':
    main()

