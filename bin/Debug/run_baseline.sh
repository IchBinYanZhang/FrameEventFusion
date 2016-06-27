declare -a baseline=("Baseline-FreeRoute" "Baseline-ConstrainedRoute")
for i in {7..26} 
do
	for j in "${baseline[@]}"
	do
		./FrameEventFusion /home/yzhang/Videos/SenseEmotion2/right/subject$i/"$j"/*.mp4 /home/yzhang/Videos/SenseEmotion2/right/subject$i/"$j"/*.txt /home/yzhang/Videos/SenseEmotion2/left/subject$i/"$j"/*.mp4 /home/yzhang/Videos/SenseEmotion2/left/subject$i/"$j"/*.txt traj_"$j"_subject$i.txt
	done 
done


