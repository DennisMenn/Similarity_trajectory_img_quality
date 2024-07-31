# Loop from 0 to 50
for i in $(seq 21 50); do
  # Print the current loop value
  echo dim $i
  # Call the Python script with the current loop value
  python3 trainer.py --dim $i --seed 0
done