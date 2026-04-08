import os
import re
# import shutil
from html import unescape
from pathlib import Path

import httpx
from dotenv import load_dotenv
from graphviz import Source
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

from loguru import logger
from logging_config import setup_logger
from uuid import uuid4


load_dotenv()

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"
SUPPORTED_EXTENSIONS = {".txt", ".pseudo", ".dsx", ".log", ".md"}

SYSTEM_PROMPT = """You are a data pipeline analyst and Graphviz DOT code generator.
Return only a single valid Graphviz DOT digraph.
Do not return SVG markup directly.
Do not use markdown fences.
Do not add explanations, headings, or extra text outside the DOT."""


def get_llm() -> AzureChatOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if not all([endpoint, deployment, api_key]):
        raise ValueError(
            "Missing Azure OpenAI env vars: "
            "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY"
        )

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_key=SecretStr(api_key),
        api_version="2024-08-01-preview",
        temperature=0.0,
        streaming=False,
        http_client=httpx.Client(timeout=httpx.Timeout(300.0, read=600.0)),
    )


def build_extraction_prompt(pseudocode: str) -> str:
    return f"""You are a DataStage lineage extraction engine.

Input: Full DataStage pseudocode
Output: Accurate Graphviz lineage diagram

---

## STEP 1 — EXTRACT EDGES

From DATA_FLOW_SUMMARY only:
* Format: SOURCE → TARGET [link_name]
* Do not infer or skip edges
* Extract every edge, even if same target appears multiple times

---

## STEP 2 — DETECT TERMINALS

Check each node's STAGE definition block. Mark as TERMINAL if ANY:
* Target Table (TableName / Target_Table_Config)
* File or Dataset output (File Name / Target File / Hashed file)
* Connector performing write (INSERT / TRUNCATE / GenerateSQL)

Terminal rules:
* Terminals come from stage properties, not just edges
* Same terminal can appear in multiple branches — this is valid
* Only the FINAL writer in a chain connects to terminal
  → If chain is A → B → TARGET, only B → TARGET (not A → TARGET too)
* If terminal missing from edges → ADD: FINAL_WRITER → TERMINAL

---

## STEP 3 — ENUMERATE ALL PATHS (MANDATORY)

This step is NON-NEGOTIABLE. Do NOT skip.

Using DFS from every root node (node with no incoming edges):
Trace every complete route to every terminal.

Write them out explicitly before building anything:

  PATH 1: A → B → C → TERMINAL_1
  PATH 2: A → B → D → TERMINAL_2
  PATH 3: A → B → D → E → TERMINAL_2
  PATH 4: A → F → TERMINAL_3

Rules:
* Every path must start at a root
* Every path must end at a confirmed terminal
* Never stop a path at an intermediate node
* Same terminal appearing in 2 paths = 2 separate path entries
* Do NOT merge or collapse any paths
* If a path cannot reach a terminal → check stage block → fix or remove node

---

## STEP 4 — BUILD & VALIDATE GRAPH

Using ONLY the paths from Step 3:
* Convert each path into edges
* Each node appears exactly once in the graph
* Each terminal gets one incoming edge per branch that writes to it
* No dangling nodes, no orphans, no intermediate terminals
* Confirm: every path from Step 3 is represented in the graph

---

## STEP 5 — RENDER

Output ONLY valid Graphviz DOT. No explanation. No comments.

### CRITICAL — HTML LABELS ARE MANDATORY FOR EVERY NODE

* ALL nodes MUST use HTML TABLE labels — no exceptions
* Plain labels like [label="node_name"] are STRICTLY FORBIDDEN
* Applies to every graph, even simple 2-node ones
* Default any unclassified node to MDL styling
* Before outputting: scan every node — if plain label found → rewrite as HTML TABLE

### GRAPH SETTINGS

digraph G {{
  rankdir=LR;
  splines=curved;
  nodesep=0.6;
  ranksep=1.2;
  fontname=Helvetica;
  node [shape=plain];
}}

### NODE CLASSIFICATION

* SRC → source tables / raw inputs
* MDL → all other nodes (transforms, connectors, files, targets)

### NODE STYLE (MANDATORY)

node_id [label=
  <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">
    <TR>
      <TD BGCOLOR="TAG_COLOR"><B>TAG</B></TD>
      <TD BGCOLOR="BODY_COLOR">node_name</TD>
    </TR>
  </TABLE>
>];

Colors:
* SRC → TAG_COLOR="#bfe3ea", BODY_COLOR="#f5f7f9"
* MDL → TAG_COLOR="#d6eaf8", BODY_COLOR="#ffffff"

Node ID rules:
* Alphanumeric + underscore only
* Replace "." and special characters with "_"
* Keep original name in the label cell

### EDGE RULES

* Simple directed edges: A -> B;
* Lookup edges → [style=dashed]
* All other edges → solid

---

## OUTPUT FORMAT

Return in this order:
1. Extracted paths (one per line)
2. Graphviz DOT code block

---

PSEUDOCODE:
{pseudocode}
"""


def _extract_digraph_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    search_from = 0

    while search_from < len(text):
        match = re.search(r"digraph\s+\w+\s*\{", text[search_from:])
        if not match:
            break

        block_start = search_from + match.start()
        brace_depth = 0
        pos = block_start

        while pos < len(text):
            if text[pos] == "{":
                brace_depth += 1
            elif text[pos] == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    blocks.append(text[block_start : pos + 1])
                    search_from = pos + 1
                    break
            pos += 1
        else:
            break

    return blocks


def parse_llm_response(response: str) -> str:
    dot_block = ""
    blocks = _extract_digraph_blocks(response)
    if blocks:
        dot_block = blocks[0].strip()
    else:
        dot_block = re.sub(r"```dot|```", "", response).strip()
    return dot_block


def sanitize_dot(dot_code: str) -> str:
    cleaned = dot_code.replace("\r\n", "\n").strip()

    # Remove markdown fences
    cleaned = re.sub(r"```dot|```", "", cleaned).strip()

    # Fix arrow encoding
    cleaned = cleaned.replace("→", "->").replace("â†'", "->")

    # Fix shape
    cleaned = cleaned.replace(
        "node [shape=plaintext]",
        "node [shape=plain]"
    )

    # ✅ CORE FIX: Join label= separated from <TABLE by whitespace/newline
    # LLM generates:   label=\n  <TABLE   → Graphviz syntax error
    # This fixes to:   label=<\n  <TABLE  → valid HTML label
    cleaned = re.sub(r'label=\s*\n\s*<', 'label=<\n  <', cleaned)

    # Fix label on edges → xlabel
    cleaned = re.sub(
        r'(->\s*\w+\s*)\[label=',
        r'\1[xlabel=',
        cleaned
    )

    # Fix bare edge attribute tokens
    cleaned = re.sub(
        r'(\s*->\s*[^\[;\n]+)\[([A-Za-z_][A-Za-z0-9_]*)\];',
        r'\1[xlabel="\2"];',
        cleaned,
    )

    return cleaned


def validate_dot(dot_code: str, label: str) -> bool:
    if not dot_code:
        # print(f"    [{label}] Empty DOT output")
        logger.error(f"[{label}] Empty DOT output")
        return False
    if not dot_code.strip().startswith("digraph"):
        print(f"    [{label}] Does not start with 'digraph'")
        return False
    if "{" not in dot_code or "}" not in dot_code:
        print(f"    [{label}] Missing braces")
        return False
    return True

def render_dot(dot_code: str, output_path: str, label: str) -> None:
    logger.info("{} Writing DOT file", label)

    dot_file = f"{output_path}.dot"
    Path(dot_file).write_text(dot_code, encoding="utf-8")
    logger.debug("{} DOT saved to {}", label, dot_file)

    try:
        logger.info("{} Rendering SVG output", label)
        src = Source(dot_code)
        src.render(output_path, format="svg", cleanup=False)
        logger.info("{} SVG rendered to {}.svg", label, output_path)

    except Exception:
        logger.exception("{} Graphviz rendering failed", label)
        logger.warning("{} DOT file preserved at {}", label, dot_file)


@logger.catch(reraise=True)
def call_llm(prompt: str, llm: AzureChatOpenAI) -> str:
    logger.info("Calling Azure OpenAI model")
    logger.debug("Prompt size: {} characters", len(prompt))

    response = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", prompt),
        ]
    ).content.strip()

    logger.debug("LLM response size: {} characters", len(response))
    return response

def generate_lineage_from_pseudocode(
    pseudocode: str,
    stem: str,
    llm: AzureChatOpenAI,
    output_dir: Path,
    log,
) -> None:
    log.info("Generating lineage for file: {}", stem)

    try:
        log.info("Building extraction prompt")
        prompt = build_extraction_prompt(pseudocode)

        log.info("Calling LLM")
        response = call_llm(prompt, llm)

        log.info("Parsing and sanitizing DOT output")
        lineage_dot = sanitize_dot(parse_llm_response(response))

        log.info("Validating DOT output")
        if not validate_dot(lineage_dot, "LINEAGE"):
            log.error("DOT validation failed")
            return

        log.info("Rendering Graphviz output")
        render_dot(
            dot_code=lineage_dot,
            output_path=str(output_dir / f"{stem}_lineage"),
            label="LINEAGE",
        )

        log.info("Lineage generation completed successfully")

    except Exception:
        log.exception("Failed to generate lineage for file: {}", stem)
        raise

def run_pipeline() -> None:
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    logger.info("Initializing pipeline")
    logger.debug("Input dir: {}", input_dir)
    logger.debug("Output dir: {}", output_dir)

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not input_files:
        logger.error("No supported files found in {}", INPUT_DIR)
        return

    logger.info("Found {} input file(s)", len(input_files))
    llm = get_llm()

    for file_path in input_files:
        logger.info("===============================================")
        logger.info("Processing file: {}", file_path.name)

        try:
            pseudocode = file_path.read_text(encoding="utf-8", errors="replace")

            if not pseudocode.strip():
                logger.warning("File is empty — skipping: {}", file_path.name)
                continue

            generate_lineage_from_pseudocode(
                pseudocode=pseudocode,
                stem=file_path.stem,
                llm=llm,
                output_dir=output_dir,
                log=logger,
            )

        except Exception:
            logger.exception("Pipeline failed for file: {}", file_path.name)

    logger.info("Pipeline run completed")

if __name__ == "__main__":
    setup_logger()
    logger.info("Pipeline started")
    run_pipeline()
    logger.info("Pipeline finished")