#
# Super simple example of a Dockerfile
#
FROM uttrip1993/regression:firstimage

WORKDIR /home/regression/

#RUN apt-get update \
#  && apt-get install -y python3-pip python3-dev \
#  && cd /usr/local/bin \
#  && ln -s /usr/bin/python3 python \
#  && pip3 install --upgrade pip
RUN apt-get update -y
RUN apt-get install sudo -y
RUN apt-get install -y emacs
RUN apt-get install -y wget bzip2
RUN apt-get upgrade -y 
RUN apt-get install curl -y
RUN apt-get install vim -y
RUN sudo apt-get install --reinstall coreutils
RUN chmod a+rwx /home/regression/
RUN sudo -H pip3 install --upgrade pip
RUN pip3 install Flask
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN sudo pip install pickle-mixin
RUN pip3 install sklearn
RUN pip3 install statistics
RUN pip3 install xgboost
RUN pip3 install requests
RUN pip3 install responses
RUN pip3 install gunicorn

Add server.py /home/regression/
#Add request.py /home/regression/
Add diamonds.csv /home/regression/
Add BigDiamonds.csv /home/regression/
Add ODSP_Regression_final.py /home/regression/

RUN python3 ODSP_Regression_final.py

#ENTRYPOINT ["python3", "server.py"]

# Anaconda installing
#RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
#RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
#RUN rm Anaconda3-5.0.1-Linux-x86_64.sh


# Set path to conda
#ENV PATH /root/anaconda3/bin:$PATH
#RUN ENV PATH /anaconda3/bin:$PATH
#RUN export PATH=$PATH:/anaconda3/bin


# Updating Anaconda packages
#RUN conda update conda
#RUN conda update anaconda
#RUN conda update --all

# Configuring access to Jupyter
#RUN mkdir /home/regression/notebooks
#RUN jupyter notebook --generate-config --allow-root
#RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /home/ubuntu/.jupyter/jupyter_notebook_config.py

# Jupyter listens port: 8888
#EXPOSE 8888