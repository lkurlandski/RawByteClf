while :
do
  bash ./run/preprocessing.sh $1
  if $? == 0; then
    exit 0
  fi
done

