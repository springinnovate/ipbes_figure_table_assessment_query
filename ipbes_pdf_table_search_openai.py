from datetime import datetime
from collections import defaultdict
import argparse
import yaml
from pathlib import Path
import os
import glob
import logging

from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

load_dotenv()
os.environ["OPENAI_API_KEY"]


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return {
        "search_path": config["search_path"],
        "questions": config["questions"],
        "output_directory": config["output_directory"],
        "preamble": config["preamble"],
    }


def load_documents(search_path):
    logging.info("Initializing OpenAI client")
    client = OpenAI()
    logging.info("Listing existing files from OpenAI")
    existing_files = {
        f.filename: f.id for f in client.files.list().data if f.filename
    }
    filename_to_openai_id = {}
    logging.info("Scanning ./data for PDF files")
    for pdf_path in glob.glob(search_path):
        pdf_name = Path(pdf_path).name
        if pdf_name in existing_files:
            logging.info(f"File already uploaded, skipping: {pdf_name}")
            filename_to_openai_id[pdf_name] = existing_files[pdf_name]
            continue
        logging.info(f"Uploading file: {pdf_name}")
        uploaded_file = client.files.create(
            file=open(pdf_path, "rb"), purpose="assistants"
        )
        logging.info(
            f"Finished uploading file: {pdf_name} (id={uploaded_file.id})"
        )
        filename_to_openai_id[pdf_name] = uploaded_file.id

    if len(filename_to_openai_id) == 0:
        raise ValueError(
            f"The search path '{search_path}'' resulted in 0 files."
        )

    logging.info("Finished processing all documents")
    return filename_to_openai_id


def format_results(results):
    out = []
    for question, file_dict in results.items():
        out.append(f"QUESTION: {question}")
        for filename, answer in file_dict.items():
            out.append(f"  FILE: {filename}")
            out.append(f"    {answer}")
        out.append("")
    return "\n".join(out)


def write_results(output_dir, text):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path = Path(output_dir) / f"results_{ts}.txt"
    with open(path, "w") as f:
        f.write(text)
    return path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Path to a YAML configuration file. The YAML must define:\n"
            "\n"
            'search_path: "<glob pattern for PDFs>"\n'
            "questions:\n"
            '  - "<question 1>"\n'
            '  - "<question 2>"\n'
            "  ...\n"
            'output_directory: "<directory for results>"\n'
            "\n"
            "Example:\n"
            '  search_path: "./data/*.pdf"\n'
            "  questions:\n"
            "    - 'Find any figures or tables in this assessment that classifies or categorizes different aspects of \"quality of life\"'\n"
            "    - 'Find any figures or tables in this assessment that classifies or categorizes different aspects of \"nature\"'\n"
            "    - 'Find any figures or tables in this assessment that classifies or categorizes different aspects of \"nature's contributions to people or ecosystem services\"'\n"
            "    - 'Find any figures or tables in this assessment that classifies or categorizes different aspects of \"indirect drivers and/or direct drivers\"'\n"
            "    - 'Find any figures or tables in this assessment that classifies or categorizes different aspects of \"scenarios or futures\"'\n"
            "    - 'Find any figures or tables in this assessment that classifies or categorizes different aspects of \"options for action or response options\"'\n"
            '  output_directory: "./output"\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("config", help="Path to YAML configuration file.")
    args = parser.parse_args()
    config = load_config(args.config)

    logging.info("Starting main workflow")
    logging.debug(config)
    filename_to_openai_id = load_documents(config["search_path"])
    logging.info(f"Loaded {len(filename_to_openai_id)} documents")

    client = OpenAI()
    logging.info("Creating assistant")
    assistant = client.beta.assistants.create(
        name="IPBES PDF QA", model="gpt-4o", tools=[{"type": "file_search"}]
    )
    logging.info(f"Assistant created with id={assistant.id}")

    results = defaultdict(dict)

    for base_question in config["questions"]:
        logging.info(f"Starting question: {base_question}")
        full_question = config["preamble"] + "\n\n" + base_question
        logging.info(full_question)
        for filename, file_id in filename_to_openai_id.items():
            logging.info(f"Asking question for file: {filename}")
            response = client.responses.create(
                model="gpt-4o",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "file_id": file_id,
                            },
                            {
                                "type": "input_text",
                                "text": full_question,
                            },
                        ],
                    }
                ],
            )

            results[base_question][filename] = response.output_text

            logging.info(f"Got response for file={filename}, printing answer")
            logging.info(f"Finished question for file={filename}")
            break
        logging.info("Finished this question for one file")
        break

    logging.info("All questions processed")

    formatted_results = format_results(results)
    write_results(config["output_directory"], formatted_results)


if __name__ == "__main__":
    main()
