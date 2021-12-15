for file in `ls $1 | grep "json$"`
do
C:/Users/q7423/anaconda3/envs/class/Scripts/labelme_json_to_dataset.exe $1"/"$file -o $2"/"$file
done