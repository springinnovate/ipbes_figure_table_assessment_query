from datetime import datetime
import json
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
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [line %(lineno)d] %(message)s",
)

load_dotenv()
os.environ["OPENAI_API_KEY"]
logging.info("Initializing OpenAI client")
CLIENT = OpenAI()


def format_results(filename, result_obj):
    try:
        lines = []
        results = result_obj.get("results")
        if not isinstance(results, list):
            raise ValueError("results not a list")

        for res in results:
            question = res.get("question", "<missing question>")
            lines.append(f"QUESTION: {question}")
            lines.append(f"  FILE: {filename}")

            matches = res.get("matches")
            if isinstance(matches, list):
                for m in matches:
                    lines.append(f'    ID: {m.get("id", "<missing id>")}')
                    lines.append(f'      Section: {m.get("section", None)}')
                    lines.append(f'      Page: {m.get("page", None)}')
                    lines.append(
                        f'      Caption: {m.get("caption", "<missing caption>")}'
                    )
                    lines.append(
                        f'      Explanation: {m.get("explanation", "<missing explanation>")}'
                    )
            else:
                lines.append(f"    MATCHES: {matches}")
        return "\n".join(lines)
    except Exception:
        return "UNEXPECTED FORMAT:\n" + repr(result_obj)


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
    logging.info("Listing existing vector stores from OpenAI")
    existing_vs = CLIENT.vector_stores.list()
    name_to_vs_id = {vs.name: vs.id for vs in existing_vs.data}

    filename_to_openai_id = {}
    logging.info(f"Scanning {search_path} for PDF files")
    for pdf_str_path in glob.glob(search_path):
        pdf_path = Path(pdf_str_path)
        store_name = f"vs_{pdf_path.name}"

        if store_name in name_to_vs_id:
            vector_store_id = name_to_vs_id[store_name]
        else:
            vector_store = CLIENT.vector_stores.create(name=store_name)
            vector_store_id = vector_store.id
            file_response = CLIENT.files.create(
                file=open(pdf_path, "rb"),
                purpose="assistants",
            )
            attach_response = CLIENT.vector_stores.files.create_and_poll(
                vector_store_id=vector_store_id,
                file_id=file_response.id,
            )
            logging.info(f"attach reponse: {attach_response}")

        filename_to_openai_id[pdf_path.name] = vector_store_id

    if len(filename_to_openai_id) == 0:
        raise ValueError(f"The search path '{search_path}'' found 0 files.")
    logging.info(f"Finished uploading all files from {search_path}")
    return filename_to_openai_id


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

    results = defaultdict(dict)

    for filename, vector_store_id in filename_to_openai_id.items():
        vs = CLIENT.vector_stores.retrieve(vector_store_id)
        logging.info(vs.file_counts)
        for base_question in config["questions"]:
            full_question = config["preamble"] + "\n\n" + base_question
            logging.info(
                f"Asking question for file: {filename}, with vector_store_id "
                f"{vector_store_id} of a question {len(full_question)} "
                f"characters long"
            )
            # this just does the search
            # search_results = CLIENT.vector_stores.search(
            #     vector_store_id=vector_store_id, query=full_question
            # )
            # answer = response.output[0].content[0].text
            # logging.info(search_results)
            # results[base_question][filename] = (
            #     search_results.data[0].content[0].text
            # )
            response = CLIENT.responses.create(
                model="gpt-5",
                input=full_question,
                tools=[
                    {
                        "type": "file_search",
                        "vector_store_ids": [vector_store_id],
                    }
                ],
            )
            msg = None
            for item in response.output:
                if getattr(item, "type", None) == "message":
                    msg = item
                    break

            text = msg.content[0].text

            start = text.find("{")
            end = text.rfind("}") + 1
            json_str = text[start:end]

            parsed = json.loads(json_str)
            logging.info(parsed)
            results[base_question][filename] = parsed

            logging.info(f"Finished question for file={filename}")
            break
        logging.info("Finished this question for one file")
        break

    logging.info("All questions processed")

    formatted_results = format_results(results)
    write_results(config["output_directory"], formatted_results)


if __name__ == "__main__":
    main()
