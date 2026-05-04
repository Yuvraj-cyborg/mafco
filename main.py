"""Entry point: delegates to the MAFCO CLI.

Run with:
    uv run main.py run "your question" --context-file path/to/notes.md
    uv run main.py demo
    uv run main.py memory show /tmp/m.json
"""

from mafco.cli import main

if __name__ == "__main__":
    main()
