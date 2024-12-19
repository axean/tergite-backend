FROM python:3.9-slim-bullseye

WORKDIR /code

# copy this only so as to increase the chances of the cache being used
# for the pip install step
COPY ./requirements.txt /code/requirements.txt

# Extract the core requirements that have a dependency of PyQt5; a difficult package to install
RUN grep -E '^(quantify-core|quantify-scheduler)' /code/requirements.txt >> core-requirements.txt; \
    # show core-requirements for debugging
    cat core-requirements.txt;

# Clean up code/requirements.txt file
RUN \
    sed -i '/^# dev-dependencies/q'  /code/requirements.txt; \
    # comment out the packages that may need PyQt5
    sed -i "s:quantify-core:# quantify-core:" /code/requirements.txt; \
    sed -i "s:quantify-scheduler:# quantify-scheduler:" /code/requirements.txt;

RUN apt update -y && apt install -y python3-pyqt5;
RUN pip install --no-cache-dir pipdeptree~=2.24.0; \
    pip install --upgrade --no-cache-dir -r /code/requirements.txt

# Install quantify-core and quantify-scheduler without dependencies
RUN pip install --upgrade --no-deps --no-cache-dir -r core-requirements.txt; \
    rm core-requirements.txt;

# Extract all the yet-to-be-installed required dependencies
RUN  pipdeptree -w silence -p quantify-core >> pending-requirements.txt; \
    pipdeptree -w silence -p quantify-scheduler >> pending-requirements.txt; \
    pip uninstall -y pipdeptree;

# Cleaning up the pending-requirements.txt
RUN \
    # remove indirect dependencies of quantify-core and quantify-scheduler
    sed -i "s:^│[[:space:]]*├──.*::" pending-requirements.txt; \
    sed -i "s:^│[[:space:]]*└──.*::" pending-requirements.txt; \
    # remove the names: quantify-core, quantify-scheduler
    sed -i "s:^quantify-.*::" pending-requirements.txt; \
    # remove the tree demarcators
    sed -i "s:^├── ::" pending-requirements.txt; \
    sed -i "s:^└── ::" pending-requirements.txt; \
    # remove already installed dependencies
    sed -i "s/.* installed: [0-9].*$//" pending-requirements.txt; \
    # cleanup dependencies whose versions don't matter
    sed -i "s/ \[required: Any,.*$//" pending-requirements.txt; \
    # remove the installation statuses
    sed -i "s/, installed: ?\]$//" pending-requirements.txt; \
    # clean up the version numbers
    sed -i "s/ \[required: //" pending-requirements.txt; \
    # remove pyqt5 dependency
    sed -i "s/^pyqt5[\>\<\=\~\!].*//" pending-requirements.txt; \
    # remove empty lines
    sed -i.bak "/^$/d" pending-requirements.txt; \
    # print the final output for debugging purposes
    cat pending-requirements.txt;

# Install all yet-to-be-installed dependencies except pyqt5
RUN pip install --no-cache-dir --upgrade -r pending-requirements.txt; \
    rm pending-requirements.txt

# clear pip's cache
RUN pip cache purge

# clear the build dependencies
RUN \
    # apt remove -y --auto-remove gcc g++ gfortran libopenblas-dev liblapack-dev python3-dev python3-pip; \
    apt autoremove; \
    apt clean;

COPY . /code/

RUN chmod +x /code/start_bcc.sh

LABEL org.opencontainers.image.licenses=APACHE-2.0
LABEL org.opencontainers.image.description="The Backend in the Tergite software stack of the WACQT quantum computer."

ENV ENV_FILE=".env"
ENV IS_SYSTEMD="false"

ENTRYPOINT ["/code/start_bcc.sh"]