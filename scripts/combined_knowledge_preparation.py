import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--correct_path', type=str)
    parser.add_argument('--incorrect_path', type=str)
    parser.add_argument('--ambiguous_path', type=str)
    args = parser.parse_args()

    with open(args.correct_path, 'r', encoding='utf-8') as f:
        correct_lines = [q.strip()[1:] if q.strip().startswith('#') else q.strip() for q in f.readlines()]
    with open(args.incorrect_path, 'r', encoding='utf-8') as f:
        incorrect_lines = [q.strip()[1:] if q.strip().startswith('#') else q.strip() for q in f.readlines()]

    ambiguous_lines = []
    for correct, incorrect in zip(correct_lines, incorrect_lines):
        ambiguous_lines.append("Knowledge1: " + correct + " [sep] Knowledge2: " + incorrect)
    with open(args.ambiguous_path, 'w', encoding='utf-8') as f:
        f.write('#')
        f.write('\n#'.join(ambiguous_lines))

if __name__ == '__main__':
    main()