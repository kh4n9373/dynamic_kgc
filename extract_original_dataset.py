import csv
import ast
import argparse


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract and process relationship triplets from dataset files')
    
    parser.add_argument('--schema', type=str, default='edc/schemas/webnlg_simple_schema.csv',
                        help='Path to schema CSV file with relation definitions')
    parser.add_argument('--triplets', type=str, default='edc/evaluate/references/webnlg_simple.txt',
                        help='Path to file containing relation triplets')
    parser.add_argument('--sentences', type=str, default='edc/datasets/webnlg.txt',
                        help='Path to file containing sentences')
    parser.add_argument('--output_csv', type=str, default='preprocessed_dataset/webnlg_simple.csv',
                        help='Path for output CSV file')
    parser.add_argument('--output_sentences', type=str, default='preprocessed_documents/webnlg_simple.txt',
                        help='Path for output file with aligned unique sentences')
    
    # Predefined configurations for convenience
    parser.add_argument('--config', type=str, choices=['webnlg_simple', 'webnlg', 'wiki-nre', 'rebel'],
                        help='Use predefined configuration instead of individual arguments')
    
    args = parser.parse_args()
    
    # Handle predefined configurations
    if args.config:
        if args.config == 'webnlg_simple':
            args.schema = 'edc/schemas/webnlg_simple_schema.csv'
            args.triplets = 'edc/evaluate/references/webnlg_simple.txt'
            args.sentences = 'edc/datasets/webnlg.txt'
            args.output_csv = 'preprocessed_dataset/webnlg_simple.csv'
            args.output_sentences = 'preprocessed_documents/webnlg_simple.txt'
        elif args.config == 'webnlg':
            args.schema = 'edc/schemas/webnlg_simple_schema.csv'
            args.triplets = 'edc/evaluate/references/webnlg.txt'
            args.sentences = 'edc/datasets/webnlg.txt'
            args.output_csv = 'preprocessed_dataset/webnlg.csv'
            args.output_sentences = 'preprocessed_documents/webnlg.txt'
        elif args.config == 'wiki-nre':
            args.schema = 'edc/schemas/wiki-nre_simple_schema.csv'
            args.triplets = 'edc/evaluate/references/wiki-nre.txt'
            args.sentences = 'edc/datasets/wiki-nre.txt'
            args.output_csv = 'preprocessed_dataset/wiki-nre.csv'
            args.output_sentences = 'preprocessed_documents/wiki-nre.txt'
        elif args.config == 'rebel':
            args.schema = 'edc/schemas/rebel_simple_schema.csv'
            args.triplets = 'edc/evaluate/references/rebel.txt'
            args.sentences = 'edc/datasets/rebel.txt'
            args.output_csv = 'preprocessed_dataset/rebel.csv'
            args.output_sentences = 'preprocessed_documents/rebel.txt'
    
    # Print the configuration being used
    print(f"\nUsing configuration:")
    print(f"  Schema file:       {args.schema}")
    print(f"  Triplets file:     {args.triplets}")
    print(f"  Sentences file:    {args.sentences}")
    print(f"  Output CSV:        {args.output_csv}")
    print(f"  Output sentences:  {args.output_sentences}\n")
    
    # Process the schema definitions
    schema_definitions = {}
    try:
        with open(args.schema, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            for row in reader:
                if len(row) == 2:
                    relation, definition = row
                    schema_definitions[relation.strip()] = definition.strip()
    except FileNotFoundError:
        print(f"Error: Schema file '{args.schema}' not found.")
        return
    except Exception as e:
        print(f"Error reading schema file '{args.schema}': {e}")
        return

    print(f"Read {len(schema_definitions)} definitions from '{args.schema}'.")

    # Read sentences
    sentences = []
    try:
        with open(args.sentences, mode='r', encoding='utf-8') as infile:
            sentences = [line.strip() for line in infile]
    except FileNotFoundError:
        print(f"Error: Sentences file '{args.sentences}' not found.")
        return
    except Exception as e:
        print(f"Error reading sentences file '{args.sentences}': {e}")
        return

    print(f"Read {len(sentences)} sentences from '{args.sentences}'.")

    # Process triplets and create dataset CSV
    ordered_unique_sentences = []
    seen_sentences = set()

    try:
        with open(args.triplets, mode='r', encoding='utf-8') as infile_triplets, \
             open(args.output_csv, mode='w', encoding='utf-8', newline='') as outfile_csv:

            csv_writer = csv.writer(outfile_csv)
            csv_writer.writerow(['line_number', 'subject', 'object', 'relation', 'relation_definition', 'sentence'])

            line_number = 0
            for line in infile_triplets:
                line_number += 1
                line = line.strip()

                if not line:
                    continue

                current_sentence = ""
                if 0 <= line_number - 1 < len(sentences):
                    current_sentence = sentences[line_number - 1]
                else:
                    print(f"Warning: No corresponding sentence found for line {line_number} in '{args.sentences}'.")

                try:
                    triplets_on_line = ast.literal_eval(line)

                    if not isinstance(triplets_on_line, list):
                        print(f"Warning: Line {line_number} in '{args.triplets}' does not contain a valid list: {line[:100]}...")
                        continue

                    found_valid_triplet_on_line = False
                    for triplet in triplets_on_line:
                        if isinstance(triplet, list) and len(triplet) == 3:
                            subject, relation, obj = triplet

                            if relation in schema_definitions:
                                definition = schema_definitions[relation]
                                csv_writer.writerow([line_number, subject.replace('_', ' '), obj.replace('_', ' '), relation, definition, current_sentence])
                                found_valid_triplet_on_line = True

                                # Add to unique sentences list
                                if current_sentence and current_sentence not in seen_sentences:
                                    ordered_unique_sentences.append(current_sentence)
                                    seen_sentences.add(current_sentence)

                except (SyntaxError, ValueError) as e:
                    print(f"Warning: Cannot parse line {line_number} in '{args.triplets}': {line[:100]}... Error: {e}")
                    continue

        print(f"CSV processing complete. Data written to '{args.output_csv}'.")

        # Write unique sentences to file for EDC processing
        try:
            with open(args.output_sentences, mode='w', encoding='utf-8') as outfile_sentences:
                for sentence in ordered_unique_sentences:
                    outfile_sentences.write(sentence + '\n')
            print(f"Unique sentences in order written to '{args.output_sentences}'.")
        except Exception as e:
            print(f"Error writing unique sentences file '{args.output_sentences}': {e}")

    except FileNotFoundError:
        print(f"Error: Triplets file '{args.triplets}' not found.")
    except Exception as e:
        print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    main()