"""
Find Problematic Language in Paper

Scans the LaTeX files for language that needs to be revised:
- "derive" → should be "parameterize" or "model"
- "first to" → overstatement
- "novel" → needs justification
- "prove" → needs to be actual proof
- "game-theoretic" → may need to be "equilibrium"

Run: python scripts/find_problematic_language.py
"""

import re
from pathlib import Path
from collections import defaultdict


def find_patterns(text, patterns, filename):
    """Find all pattern matches with context."""
    results = []
    lines = text.split('\n')

    for pattern_name, pattern in patterns.items():
        for i, line in enumerate(lines):
            matches = list(re.finditer(pattern, line, re.IGNORECASE))
            for match in matches:
                results.append({
                    'file': filename,
                    'line': i + 1,
                    'pattern': pattern_name,
                    'match': match.group(),
                    'context': line.strip()[:100]
                })

    return results


def main():
    base_path = Path(__file__).parent.parent
    tex_dir = base_path / "jmlr_submission"

    patterns = {
        'derive': r'\bderiv(e|ed|es|ation|ing)\b',
        'first_to': r'first\s+to\b',
        'novel': r'\bnovel\b',
        'prove': r'\b(we\s+)?prov(e|ed|es|ing)\b',
        'game_theoretic': r'game[- ]?theoretic',
        'mechanism': r'\bmechanism(s|atic)?\b',
        'guarantee': r'\bguarantee[sd]?\b',
        'establish': r'\bestablish(ed|es|ing)?\b',
        'show_that': r'\bshow\s+that\b',
    }

    all_results = []

    # Scan main tex file
    main_tex = tex_dir / "main.tex"
    if main_tex.exists():
        text = main_tex.read_text()
        all_results.extend(find_patterns(text, patterns, "main.tex"))

    # Scan section files
    sections_dir = tex_dir / "sections"
    if sections_dir.exists():
        for tex_file in sections_dir.glob("*.tex"):
            text = tex_file.read_text()
            all_results.extend(find_patterns(text, patterns, f"sections/{tex_file.name}"))

    # Scan appendix files
    appendices_dir = tex_dir / "appendices"
    if appendices_dir.exists():
        for tex_file in appendices_dir.glob("*.tex"):
            text = tex_file.read_text()
            all_results.extend(find_patterns(text, patterns, f"appendices/{tex_file.name}"))

    # Group by pattern
    by_pattern = defaultdict(list)
    for r in all_results:
        by_pattern[r['pattern']].append(r)

    # Print report
    print("\n" + "="*70)
    print("PROBLEMATIC LANGUAGE AUDIT")
    print("="*70)

    priority_patterns = ['derive', 'first_to', 'prove', 'game_theoretic', 'guarantee']

    for pattern in priority_patterns:
        matches = by_pattern.get(pattern, [])
        print(f"\n{'='*60}")
        print(f"PATTERN: '{pattern}' ({len(matches)} matches)")
        print('='*60)

        if matches:
            for m in matches[:10]:  # Show first 10
                print(f"\n  File: {m['file']}, Line {m['line']}")
                print(f"  Match: '{m['match']}'")
                print(f"  Context: {m['context']}")

            if len(matches) > 10:
                print(f"\n  ... and {len(matches) - 10} more matches")

            # Recommendations
            recommendations = {
                'derive': "→ Consider: 'model', 'parameterize', 'characterize', 'specify'",
                'first_to': "→ Consider: 'We contribute by...', 'Our approach...', or remove",
                'prove': "→ Verify these are actual proofs. If not, use 'show', 'demonstrate'",
                'game_theoretic': "→ Consider: 'equilibrium model', 'competitive equilibrium'",
                'guarantee': "→ If coverage guarantee, verify assumption. Consider: 'empirical coverage'"
            }
            if pattern in recommendations:
                print(f"\n  RECOMMENDATION: {recommendations[pattern]}")
        else:
            print("  No matches found ✓")

    # Summary stats
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nTotal issues found: {len(all_results)}")
    print("\nBy pattern:")
    for pattern in patterns:
        count = len(by_pattern.get(pattern, []))
        if count > 0:
            print(f"  {pattern}: {count}")

    # Files with most issues
    by_file = defaultdict(int)
    for r in all_results:
        by_file[r['file']] += 1

    print("\nBy file:")
    for f, count in sorted(by_file.items(), key=lambda x: -x[1])[:10]:
        print(f"  {f}: {count} issues")

    # Save detailed results
    output_path = base_path / "results" / "language_audit.txt"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("PROBLEMATIC LANGUAGE AUDIT\n")
        f.write("="*70 + "\n\n")

        for r in all_results:
            f.write(f"File: {r['file']}, Line {r['line']}\n")
            f.write(f"Pattern: {r['pattern']}\n")
            f.write(f"Match: {r['match']}\n")
            f.write(f"Context: {r['context']}\n")
            f.write("-"*40 + "\n")

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
