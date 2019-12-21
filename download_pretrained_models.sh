set -x

for model in base large xlarge xxlarge
do
  wget https://storage.googleapis.com/albert_models/albert_${model}_v2.tar.gz
  tar -xvf albert_${model}_v2.tar.gz
  mv albert_${model} albert_${model}_v2
  rm albert_${model}_v2.tar.gz
done
