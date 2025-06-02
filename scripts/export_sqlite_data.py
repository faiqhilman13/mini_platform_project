#!/usr/bin/env python3
"""
SQLite to Supabase Data Export Script
Exports all data from the current SQLite database to JSON format for migration.
"""
import sqlite3
import json
import os
from datetime import datetime

def export_sqlite_data():
    """Export all data from SQLite to JSON files"""
    db_path = 'mini_idp.db'
    
    if not os.path.exists(db_path):
        print(f"❌ Database file {db_path} not found!")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get all table names
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        all_tables = [row[0] for row in cursor.fetchall()]
        print(f"📊 Found tables: {all_tables}")
        
        # Tables we want to export (in dependency order)
        tables_to_export = [
            'uploadedfilelog',
            'pipelinerun', 
            'dataprofiling',
            'ml_pipeline_run',
            'ml_experiment',
            'ml_model'
        ]
        
        # Filter to only export tables that exist
        existing_tables = [table for table in tables_to_export if table in all_tables]
        print(f"📋 Exporting tables: {existing_tables}")
        
        export_data = {}
        total_records = 0
        
        for table in existing_tables:
            print(f"\n📤 Exporting table: {table}")
            
            # Get table schema
            cursor = conn.execute(f"PRAGMA table_info({table})")
            schema = cursor.fetchall()
            columns = [col[1] for col in schema]
            
            # Get all data
            cursor = conn.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            
            # Convert to dict format
            table_data = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                table_data.append(row_dict)
            
            export_data[table] = {
                'schema': schema,
                'columns': columns,
                'rows': table_data,
                'count': len(table_data)
            }
            
            total_records += len(table_data)
            print(f"  ✅ Exported {len(table_data)} records")
        
        # Add metadata
        export_data['_metadata'] = {
            'export_timestamp': datetime.now().isoformat(),
            'source_database': db_path,
            'total_tables': len(existing_tables),
            'total_records': total_records,
            'tables_exported': existing_tables
        }
        
        # Save to JSON file
        output_file = 'data_export.json'
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        conn.close()
        
        print(f"\n🎉 Export completed successfully!")
        print(f"📁 Output file: {output_file}")
        print(f"📊 Total tables: {len(existing_tables)}")
        print(f"📈 Total records: {total_records}")
        
        # Show summary
        print(f"\n📋 Export Summary:")
        for table in existing_tables:
            count = export_data[table]['count']
            print(f"  • {table}: {count} records")
            
    except Exception as e:
        print(f"❌ Error during export: {e}")
        if 'conn' in locals():
            conn.close()

def preview_export_data():
    """Preview exported data without actually exporting"""
    db_path = 'mini_idp.db'
    
    if not os.path.exists(db_path):
        print(f"❌ Database file {db_path} not found!")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get all table names and record counts
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print("📊 Database Preview:")
        print(f"Database: {db_path}")
        print(f"Tables found: {len(tables)}")
        
        total_records = 0
        for table in tables:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            total_records += count
            print(f"  • {table}: {count} records")
        
        print(f"\nTotal records across all tables: {total_records}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error during preview: {e}")
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        preview_export_data()
    else:
        export_sqlite_data() 