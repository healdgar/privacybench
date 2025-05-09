import json
import os
import pandas as pd
from pathlib import Path

def main():
    try:
        print("Starting script execution...")
        
        # Get absolute paths
        base_dir = Path(__file__).parent
        input_path = base_dir / "output_results.json"
        output_path = base_dir / "bias_ratings_table.html"
        
        print(f"Input file path: {input_path}")
        print(f"Output file path: {output_path}")
        
        # Verify input file exists
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found at {input_path}")
            
        # Load data
        print("Loading input data...")
        with open(input_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        # Process data
        print("Processing data...")
        rows = []
        for summary in results.get("summarization_results", []):
            for rating in summary.get("bias_ratings", []):
                rows.append({
                    "Model": summary.get("summarizer_model"),
                    "Criteria": rating.get("criteria"),
                    "Rating": rating.get("rating_numeric")
                })
                
        df = pd.DataFrame(rows)
        
        # Pivot data to show criteria as columns, models as rows
        print("Pivoting data...")
        pivot_df = df.pivot_table(
            index='Model',
            columns='Criteria',
            values='Rating',
            aggfunc='mean'
        ).reset_index()
        
        # Generate HTML
        print("Generating HTML...")
        html_content = pivot_df.to_html(classes='sortable', index=False)
        
        # Write output
        print(f"Writing output to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"<!DOCTYPE html><html><body>{html_content}</body></html>")
            
        print(f"Success! HTML file created at: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Script failed to complete")

if __name__ == "__main__":
    main()
