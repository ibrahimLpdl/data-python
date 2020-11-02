FROM ubuntu

WORKDIR /usr/src/app
COPY . .
RUN apt-get -y update

RUN apt-get -y install python3 \
&& apt-get -y install python3-pip \
&& apt-get install -y git \
&& apt-get install -y vim \
&& pip3 install -r requirement.txt \
&& git config --global user.name ibrahimLpdl \
&& git config --global user.email shanks91.ib@gnail.com \
&& git clone https://github.com/ibrahimLpdl/data-python.git

EXPOSE 8000
CMD ["jupyter", "notebook", "--port=8000", "--no-browser", "--ip=0.0.0.0", "--allow-root"]