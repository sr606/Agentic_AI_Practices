from pathlib import Path

from lineage_mcp.toolbox import run_lineage_pipeline


def main() -> None:
    input_dir = Path("lineage_mcp/data/uploads")
    output_dir = Path("lineage_mcp/data/output")

    input_files = []
    input_files.extend(input_dir.glob("*.txt"))
    input_files.extend(input_dir.glob("*.xml"))
    input_files.extend(input_dir.glob("*.json"))

    results = []
    for input_file in sorted(input_files):
        result = run_lineage_pipeline(
            input_path=str(input_file),
            output_dir=str(output_dir),
        )
        results.append(result)

    print("Pipeline finished")
    print(results)


if __name__ == "__main__":
    main()
