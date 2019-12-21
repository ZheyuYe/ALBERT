set -x

for model in base large xlarge xxlarge
do
  mkdir albert_${model}_v2
  wget https://tfhub.dev/google/albert_${model}/2?tf-hub-format=compressed -O albert_${model}_v2/albert_${model}_v2.tar.gz
  cd albert_${model}_v2
  tar -xvf albert_${model}_v2.tar.gz
  rm albert_${model}_v2.tar.gz
  cd ..
done
