import json
import csv
import argparse


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract relationship data from EDC output JSON to CSV format')
    
    parser.add_argument('--input', type=str, 
                        default='edc/output/webnlg_simple_target_alignment/iter0/result_at_each_stage.json',
                        help='Path to input JSON file from EDC')
    parser.add_argument('--output', type=str, 
                        default='preprocessed_edc_result/edc_output.csv',
                        help='Path for output CSV file')
    
    # Predefined configurations
    parser.add_argument('--config', type=str, choices=['webnlg_simple', 'webnlg', 'wiki-nre', 'rebel'],
                        help='Use predefined configuration instead of individual arguments')
    
    args = parser.parse_args()
    
    # Handle predefined configurations
    if args.config:
        if args.config == 'webnlg_simple':
            args.input = 'edc/output/webnlg_simple_target_alignment/iter0/result_at_each_stage.json'
            args.output = 'preprocessed_edc_result/edc_webnlg_simple.csv'
        elif args.config == 'webnlg':
            args.input = 'edc/output/webnlg_target_alignment/iter0/result_at_each_stage.json'
            args.output = 'preprocessed_edc_result/edc_webnlg.csv'
        elif args.config == 'wiki-nre':
            args.input = 'edc/output/wiki-nre_target_alignment/iter0/result_at_each_stage.json'
            args.output = 'preprocessed_edc_result/edc_wiki-nre.csv'
        elif args.config == 'rebel':
            args.input = 'edc/output/rebel_target_alignment/iter0/result_at_each_stage.json'
            args.output = 'preprocessed_edc_result/edc_rebel.csv'
    
    # Print the configuration being used
    print(f"\nUsing configuration:")
    print(f"  Input JSON: {args.input}")
    print(f"  Output CSV: {args.output}\n")
    
    # Read data from JSON file
    try:
        with open(args.input, mode='r', encoding='utf-8') as infile:
            data = json.load(infile)
    except FileNotFoundError:
        print(f"Error: JSON file '{args.input}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: JSON file '{args.input}' is invalid. Details: {e}")
        return
    except Exception as e:
        print(f"Unidentified error reading JSON file '{args.input}': {e}")
        return

    print(f"Successfully read data from '{args.input}'.")

    # Extract EDC output and convert to CSV format
    try:
        with open(args.output, mode='w', encoding='utf-8', newline='') as outfile:
            csv_writer = csv.writer(outfile)

            # Write header row
            header = ['sentence', 'extracted_subject', 'extracted_object', 'extracted_relation', 'extracted_relation_definition']
            csv_writer.writerow(header)

            # Process each dictionary in the data list
            rows_written = 0
            for item in data:
                sentence = item.get('input_text', '').replace('\n', ' ').strip()

                # Get the list of triplets from 'oie'
                oie_list = item.get('oie', [])
                if not isinstance(oie_list, list):
                    print(f"Warning: 'oie' field is not a list in item with index {item.get('index', 'N/A')}. Skipping this item.")
                    continue

                # Get the schema definition dictionary
                definitions = item.get('schema_definition', {})
                if not isinstance(definitions, dict):
                    print(f"Warning: 'schema_definition' field is not a dictionary in item with index {item.get('index', 'N/A')}. Using empty dict.")
                    definitions = {}

                # Process each triplet in the 'oie' list
                for triplet in oie_list:
                    if isinstance(triplet, list) and len(triplet) == 3:
                        raw_subject, relation, raw_object = triplet

                        subject = str(raw_subject).replace('_', ' ') if raw_subject is not None else ''
                        object_ = str(raw_object).replace('_', ' ') if raw_object is not None else ''

                        definition = definitions.get(relation, '')

                        csv_writer.writerow([sentence, subject, object_, relation, definition])
                        rows_written += 1
                    else:
                        print(f"Warning: Invalid triplet {triplet} in item with index {item.get('index', 'N/A')}. Skipping this triplet.")

        print(f"Processing complete. {rows_written} rows written to '{args.output}'.")

    except Exception as e:
        print(f"An error occurred while writing the CSV file: {e}")


if __name__ == "__main__":
    main()