"""
src/visualise_app.py

Renders the ContentBlitz LangGraph layout as a PNG and opens it automatically.

Usage:
    python3 -m src.visualise_app

Output:
    graph.png  — saved in the project root, opened via macOS 'open'.
    Falls back to ASCII in the terminal if Mermaid PNG render fails.
"""

import os
import subprocess
import sys

from src.workflow.langgraph_workflow import visualise_graph


def main() -> None:
    print("Rendering ContentBlitz graph...")

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "graph.png",
    )

    saved_path = visualise_graph(output_path=output_path)

    if saved_path:
        print(f"Graph saved → {saved_path}")
        try:
            subprocess.run(["open", saved_path], check=True)
            print("Opened graph.png")
        except Exception as exc:
            print(f"Could not auto-open file: {exc}")
            print(f"Open manually: open {saved_path}")
    else:
        print("PNG render failed — ASCII graph printed above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
