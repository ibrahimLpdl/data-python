FROM ubuntu

WORKDIR /usr/src/app
COPY . .
RUN apt-get -y update

RUN apt-get -y install python3 \
&& apt-get -y install python3-pip \
&& pip3 install -r requirement.txt \

EXPOSE 8000
CMD ["python","algoAPI.py"]