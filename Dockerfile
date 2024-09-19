FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN echo 'APT::Install-Recommends "0" ; APT::Install-Suggests "0" ;' >> /etc/apt/apt.conf && \
echo 'Dir::Cache::pkgcache "";\nDir::Cache::srcpkgcache "";' | tee /etc/apt/apt.conf.d/00_disable-cache-files

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# enable for nnunet==1.7 compatibility
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
ENV PATH="/home/user/.local/bin:${PATH}"

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    matplotlib \
    nnunet==1.7.0 \
    cupy-cuda12x \
    cucim-cu12

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources
    
RUN python -m pip install -e /opt/app/resources/src/fracture-surfaces
ENV nnUNet_results=/opt/app/resources/models/fracture_seg

RUN cp -r resources/src/fracSegNet/* /home/user/.local/lib/python3.11/site-packages/nnunet/
ENV RESULTS_FOLDER=/opt/app/resources/models/anatomical_seg

# COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user run_inference.sh /opt/app/

# ENTRYPOINT ["python", "inference.py"]
ENTRYPOINT ["/bin/bash", "run_inference.sh"]
