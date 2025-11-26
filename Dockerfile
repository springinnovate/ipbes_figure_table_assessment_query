#docker build -t ipbes_figure_assessment_query . && docker run --rm -it -v %CD%:/app ipbes_figure_assessment_query
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir openai python-dotenv pyyaml

CMD ["bash"]
