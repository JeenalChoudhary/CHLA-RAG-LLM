FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

COPY ./ollama /usr/local/bin/ollama

RUN chmod +x /usr/local/bin/ollama

COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]